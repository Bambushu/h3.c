#!/usr/bin/env python3
"""Fold a MiniMax-H3 LoRA into an int8 checkpoint (tools/h3_int8_pack), in place.

The int8 counterpart of fold_turbo_lora.py, for machines that only hold the
int8 copy. For an I8 target with "<name>_int8_scale" row scales, each weight is
expanded (q * scale), the LoRA is added (W + B@A), the result is rounded to
BF16 and quantized again with h3_int8_pack's arithmetic, so it matches packing
a folded BF16 checkpoint except for the original int8 rounding. BF16 targets
are patched as fold_turbo_lora.py does. Sizes and offsets do not change.

Each shard is cloned (APFS copy-on-write), patched, and renamed over the
original. turbo_folded.json in the transformer directory records the folded
shards, so an interrupted run resumes and a finished one is never folded twice.

Usage:
  python3 tools/fold_turbo_lora_int8.py \
      --checkpoint MiniMax-H3-int8/FL2VA/transformer \
      --lora minimax_h3_turbo_v4_step600_ema.safetensors
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

SCALE_SUFFIX = "_int8_scale"


def read_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return hlen, header


def read_raw(f, info, base):
    start, end = info["data_offsets"]
    f.seek(base + start)
    return f.read(end - start)


def load_tensor(path, info, base):
    with open(path, "rb") as f:
        raw = read_raw(f, info, base)
    dtype = info["dtype"]
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (u16 << 16).view(np.float32).reshape(info["shape"])
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(info["shape"]).copy()
    raise SystemExit(f"unsupported dtype {dtype} for {path}")


def bf16_round(x):
    u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def quantize_rows(w):
    """h3_int8_pack.c quantize_rows: per-row max/127, rint, clamp to +-127."""
    max_abs = np.max(np.abs(w), axis=1).astype(np.float32)
    nonzero = max_abs > 0
    scale = np.where(nonzero, max_abs / np.float32(127), np.float32(1 / 127))
    inverse = np.where(nonzero, np.float32(127) / np.where(nonzero, max_abs, 1),
                       np.float32(127)).astype(np.float32)
    q = np.clip(np.rint(w * inverse[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def clone(src, dst):
    if subprocess.run(["cp", "-c", str(src), str(dst)],
                      capture_output=True).returncode != 0:
        shutil.copy2(src, dst)


def parity(W, folded, A, B, scale, key):
    x = np.random.default_rng(0).standard_normal(W.shape[1]).astype(np.float32)
    direct = folded @ x
    composed = W @ x + scale * (B @ (A @ x))
    rel = np.max(np.abs(direct - composed)) / (np.max(np.abs(composed)) + 1e-9)
    if rel > 1e-4:
        raise SystemExit(f"{key}: parity check failed (rel={rel:.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="int8 transformer directory, folded in place")
    ap.add_argument("--lora", required=True, type=Path,
                    help="LoRA .safetensors with <key>.lora_A/.lora_B pairs")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--only", help="fold just this shard file name (testing)")
    args = ap.parse_args()

    lhl, lora_hdr = read_header(args.lora)
    lora_base = 8 + lhl
    pairs = {}
    for key in lora_hdr:
        if key.endswith(".lora_A.weight"):
            stem = key[: -len(".lora_A.weight")]
            pairs[stem + ".weight"] = (key, stem + ".lora_B.weight")
    if not pairs:
        raise SystemExit("no .lora_A/.lora_B pairs found in the adapter")

    marker = args.checkpoint / "turbo_folded.json"
    state = json.loads(marker.read_text()) if marker.exists() else {
        "lora": args.lora.name, "scale": args.scale, "shards": []}
    if state["lora"] != args.lora.name or state["scale"] != args.scale:
        raise SystemExit(f"{marker} records a different adapter or scale")

    shards = sorted(args.checkpoint.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no shards in {args.checkpoint}")
    found = 0
    worst = 0.0
    for shard in shards:
        hlen, hdr = read_header(shard)
        todo = [k for k in hdr if k in pairs]
        found += len(todo)
        if not todo or shard.name in state["shards"]:
            continue
        if args.only and shard.name != args.only:
            continue
        base = 8 + hlen
        partial = shard.with_name(shard.name + ".partial")
        clone(shard, partial)
        with open(partial, "r+b") as f:
            for key in todo:
                info = hdr[key]
                a_key, b_key = pairs[key]
                A = load_tensor(args.lora, lora_hdr[a_key], lora_base)
                B = load_tensor(args.lora, lora_hdr[b_key], lora_base)
                if info["dtype"] == "I8":
                    rows, columns = info["shape"]
                    q = np.frombuffer(read_raw(f, info, base),
                                      dtype=np.int8).reshape(rows, columns)
                    s_info = hdr[key + SCALE_SUFFIX]
                    s = np.frombuffer(read_raw(f, s_info, base), dtype="<f4")
                    W = q.astype(np.float32) * s[:, None]
                    folded = W + args.scale * (B @ A)
                    parity(W, folded, A, B, args.scale, key)
                    folded = bf16_to_f32(bf16_round(folded))
                    q2, s2 = quantize_rows(folded)
                    err = np.max(np.abs(q2 * s2[:, None] - folded) /
                                 s2[:, None])
                    worst = max(worst, float(err))
                    for data, where in ((q2.tobytes(), info),
                                        (s2.astype("<f4").tobytes(), s_info)):
                        start, end = where["data_offsets"]
                        if len(data) != end - start:
                            raise SystemExit(f"{key}: byte count mismatch")
                        f.seek(base + start)
                        f.write(data)
                elif info["dtype"] == "BF16":
                    W = load_tensor(partial, info, base)
                    folded = W + args.scale * (B @ A)
                    parity(W, folded, A, B, args.scale, key)
                    data = bf16_round(folded).astype("<u2").tobytes()
                    start, end = info["data_offsets"]
                    if len(data) != end - start:
                        raise SystemExit(f"{key}: byte count mismatch")
                    f.seek(base + start)
                    f.write(data)
                else:
                    raise SystemExit(f"{key}: unsupported dtype {info['dtype']}")
            f.flush()
            os.fsync(f.fileno())
        os.replace(partial, shard)
        state["shards"].append(shard.name)
        marker.write_text(json.dumps(state, indent=1) + "\n")
        print(f"{shard.name}: {len(todo)} tensors folded "
              f"(worst requant error {worst:.3f} of a step)", flush=True)

    if found != len(pairs):
        raise SystemExit(f"only {found} of {len(pairs)} adapter pairs matched "
                         "checkpoint tensors - wrong checkpoint or adapter?")
    print("done")


if __name__ == "__main__":
    sys.exit(main())
