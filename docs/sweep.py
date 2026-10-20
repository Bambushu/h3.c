#!/usr/bin/env python3
"""Census of every antirez/h3.c fork: ahead_by + commit subjects per branch.
Writes census.json and census.md in the cwd. Uses `gh api` (authenticated)."""
import json, subprocess, sys, concurrent.futures as cf
from datetime import datetime

UP = "antirez/h3.c"
FREEZE = "2026-08-11T15:50"
OUT_JSON, OUT_MD = "census.json", "census.md"


def api(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def compare(owner, branch):
    d = api(f"repos/{UP}/compare/main...{owner}:{branch}")
    if not d:
        return None
    return {
        "ahead": d.get("ahead_by", 0),
        "behind": d.get("behind_by", 0),
        "files": d.get("files") and len(d["files"]),
        "subjects": [c["commit"]["message"].split("\n")[0][:110] for c in d.get("commits", [])],
        "dates": [c["commit"]["author"]["date"][:10] for c in d.get("commits", [])],
    }


def main():
    forks = []
    for line in open("forks.tsv"):
        p = line.rstrip("\n").split("\t")
        forks.append({"repo": p[0], "pushed": p[1], "stars": int(p[2]), "default": p[3],
                      "subforks": int(p[4]), "desc": p[5] if len(p) > 5 else ""})
    # sub-forks (one level)
    for f in [x for x in forks if x["subforks"] > 0]:
        for sf in api(f"repos/{f['repo']}/forks?per_page=100") or []:
            forks.append({"repo": sf["full_name"], "pushed": sf["pushed_at"], "stars": sf["stargazers_count"],
                          "default": sf["default_branch"], "subforks": 0, "desc": sf.get("description") or "",
                          "parent": f["repo"]})

    def work(f):
        owner = f["repo"].split("/")[0]
        res = {"default_branch": compare(owner, f["default"]), "branches": {}}
        if f["pushed"] > FREEZE:
            brs = api(f"repos/{f['repo']}/branches?per_page=100") or []
            for b in brs:
                if b["name"] == f["default"]:
                    continue
                c = compare(owner, b["name"])
                if c and c["ahead"] > 0:
                    res["branches"][b["name"]] = c
        f.update(res)
        return f

    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        forks = list(ex.map(work, forks))

    json.dump(forks, open(OUT_JSON, "w"), indent=1)

    def total_ahead(f):
        a = (f["default_branch"] or {}).get("ahead", 0)
        return a + sum(b["ahead"] for b in f["branches"].values())

    live = [f for f in forks if total_ahead(f) > 0]
    live.sort(key=lambda f: (-total_ahead(f), f["pushed"]), reverse=False)
    with open(OUT_MD, "w") as md:
        md.write(f"# h3.c fork census {datetime.now():%Y-%m-%d}\n\n")
        md.write(f"{len(forks)} forks scanned, {len(live)} carry commits not in upstream main.\n\n")
        for f in live:
            d = f["default_branch"] or {}
            md.write(f"## {f['repo']}  stars={f['stars']} pushed={f['pushed'][:10]} "
                     f"default `{f['default']}` ahead={d.get('ahead',0)}"
                     + (f" (sub-fork of {f['parent']})" if f.get("parent") else "") + "\n")
            if f["desc"]:
                md.write(f"_{f['desc'][:200]}_\n")
            for s, dt in zip(d.get("subjects", []), d.get("dates", [])):
                md.write(f"- {dt} {s}\n")
            for bn, b in f["branches"].items():
                md.write(f"### branch `{bn}` ahead={b['ahead']}\n")
                for s, dt in zip(b["subjects"], b["dates"]):
                    md.write(f"- {dt} {s}\n")
            md.write("\n")
    print(f"scanned {len(forks)} forks, {len(live)} ahead; wrote {OUT_MD}")


if __name__ == "__main__":
    main()
