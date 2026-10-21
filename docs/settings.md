# Settings log: what this build does, 2026-10-21

The state of `integration` as built in `~/h3c` and used by the previz runners.
Only what differs from upstream antirez/h3.c is listed; upstream's own knobs
are documented in the README's implementation notes. Numbers behind every
default: `docs/bench-2026-10-21.md`.

## Defaults that differ from upstream

| Setting | This build | Upstream | Override |
| --- | --- | --- | --- |
| DiT weights | resident int8 (no `--ssd-streaming`) with the quantized cache at `~/Library/Caches/h3c/dit` (18 GiB, cached load 7 s vs 106 s) | same weights, cache opt-in | `H3_DIT_CACHE=0` disables, `H3_DIT_CACHE=<dir>` relocates, `H3_DIT_CACHE_VERIFY=1` checks it; `--ssd-streaming` for small-memory Macs |
| Video VAE precision | bf16 (63 dB against F32, 3 to 4x faster decode) | f32 | `H3_VAE_PRECISION=f32` or `int8`; `H3_VAE_F32_ATTENTION=1` keeps attention in F32 |
| VAE decoder prefetch | on (decoder loads during the denoise; bit-identical) | not in upstream | `H3_VAE_PREFETCH=0` |
| VAE weight cache | off | not in upstream | `H3_VAE_CACHE=<dir>` (4.8 GiB, bit-identical) |
| Pipelined decode tail | on | not in upstream | `H3_VAE_DECODE_LEGACY=1`; `H3_VAE_UNFUSED=1` keeps the pipeline, unfuses epilogues |
| VAE tile size | 256 px | geometry scan (320 px at 576x1024 and 768x1344, prints a 16 px grid) | `H3_VAE_TILE_PIXELS=N` |
| VAE encoder | arena allocation | per-block | `H3_DISABLE_VAE_ENCODER_ARENA=1`, `H3_VAE_ENCODER_CONV2D=1` |
| H.264 encode | `-preset slow -crf 18` | `-preset fast -crf 18` | none |
| `--first-frame` | center-cropped to the render aspect | stretched | none |
| Reference audio | up to 45 s | shorter bound | none |
| Process lock | flock on `$TMPDIR/h3.c.lock`, a second instance waits | none | none |
| Sampler | `--sampler euler` (default) or `res` | euler only | flag, or `!sampler` interactively |
| Scheduler | `--scheduler linear` (default), `simple`, `beta`, `beta57` | linear only | flag |
| Video sigma shift | `--shift 12` (default; audio stays 3) | fixed 12 | flag, `!shift` interactively |
| SDPA query split (PR #64) | off (no accuracy cliff measured on the M5 Pro) | not in upstream; the PR defaults it on | `--sdpa-query-block 2048` or `H3_SDPA_MAX_QUERY_ROWS=2048`; `H3_SDPA_SPLIT_MIN_ROWS` (12288) |
| GPU queue depth | upstream default | | `H3_GPU_QUEUE_DEPTH=<n>` |
| Early DiT load | off | not in upstream | `H3_DIT_EARLY_LOAD=1` (DiT coexists with the encoders; needs the memory) |
| Sol-Attn block-sparse attention | off | not in upstream | below |

Defaults the same as upstream and left alone: `--steps 20`, `--reuse 1`,
`--layers 50`, `--core-reuse 1`, `--seed 42`, 864x480, 56 frames.

## Sol-Attn (parked)

Measured on this machine: at tau 1.3 it is 3 to 13% faster than dense and
drifts the set and hands (Mike and a blind judge agreed); at tau 0.5 and 0 it
is slower than dense. Kept as a flag for other machines and for a Morton-order
experiment. Nothing in the runners sets it.

| Variable | Default | Meaning |
| --- | --- | --- |
| `H3_SOL_ATTN` | unset | `1` enables the sparse path |
| `H3_SOL_ATTN_TAU` | `-1` | routing threshold, mean + tau x std of the centroid scores; 1.3 is yshenaw's production value |
| `H3_SOL_ATTN_MIN_ROWS` | `4096` | smallest sequence that routes sparsely |
| `H3_SOL_ATTN_DENSE_STEPS` | unset | `N` keeps the first N steps dense, every later step sparse; unset keeps the 20% to 90% window. (`H3_SOL_ATTN_ALL_STEPS` was removed 2026-10-21; `0` here is the same thing) |
| `H3_SOL_ATTN_STRICT` | unset | `1` turns a missing library or buffer into an error instead of a dense fallback |
| `H3_PROFILE_SOL_ROUTES` | unset | `1` prints route density for blocks 0, 10, ... 49 on the first sparse step |
| `H3_DUMP_SOL_ATTENTION=<prefix>`, `H3_DUMP_SOL_BLOCK`, `H3_DUMP_SOL_STEP` | unset | capture post-RoPE Q/K/V of one block and step for `h3_sol_attention_bench` |

## Diagnostics added

`H3_DUMP_VIDEO_LATENT=<file>` writes the final video latent. `H3_PROFILE`
(upstream) now also prints the Sol route density line and the VAE
"delivered at" clock for a prefetched decoder.

## The production lane, as the runners call it

Tonton `ad/previz/run_h3c.sh` and Ben `ad6_samevial/previz/run_h3c*.sh`
export nothing; the compiled defaults are the lane:

```
~/h3c/h3 -d ~/h3c/fl2va_as_r2v -p "$(cat composed/ID/gen_01.txt)" \
  --width 544 --height 960 --seconds N --steps 8 --layers 50 --reuse 1 \
  --seed S --profile [--first-frame plate.png] [--ref-image ref.png] [--ref-audio vocal.wav]
```

Ben's scripts default to 576x1024; the 1 MP pass is `W=768 HH=1344` (Tonton
adds `STEPS=14 PLATES=plates/1mp`). One job at a time, behind Mike's
`~/h3-demos/local/h3lock.sh` (refuses to start under 25% free memory or above
6 GB swap in use; stale swap after a long session trips it until a reboot).

Measured on the reference shot (124 frames, 8 steps):

| Shape | DiT load (cached) | Denoise | Decode | Wall | Peak DiT |
| --- | ---: | ---: | ---: | ---: | --- |
| 544x960 | 7 s | 297 s | 40 s | about 6.5 min | 20 GiB |
| 768x1344 | 7 s | 961 to 1015 s | 86 s | about 18.5 min | 23 GiB |

Upstream on the same shot: 12 min and 27.7 min.

## Not merged, not planned unless asked

ldoroshe's graft-format checkpoint loader (conflicts with the AdaLN cache),
LachlanStuart's hybrid anchors, PR #55 SSD attention streaming, neokree's
query-blocking default, runtime LoRA loading, Morton ordering for Sol.

## Bench harness

`~/h3c-bench/`: `bin_upstream/` (the 8974cc0 binary, the only build kept;
each run dir carries `BUILD_COMMIT`), one directory per run with `run.log`,
`frames/`, `out.mp4`, `whisper.txt`; `qa_compare.sh` writes `RESULTS.md`
(timings, frame md5, PSNR against `base_upstream`, grid score, whisper);
`judge_sheets.sh` makes the blind-judge sheets in `judge/`; `review/` holds
the clips Mike watched; the `run_*.sh` one-offs are kept as the record of each
run's exact invocation; `dit_cache` is a symlink to the real cache.
