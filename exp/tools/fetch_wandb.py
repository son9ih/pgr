"""Pull run metadata (command line, config, final summary) for the 8 `*_final`
wandb projects under entity `gda-for-orl` into exp/data/wandb_final_runs.json.

Usage:  python exp/tools/fetch_wandb.py
"""
import json
import os

import wandb

ENTITY = "gda-for-orl"
PROJECTS = [
    "quad_final", "cheetah_final", "reacher_final",
    "fingereasy_final", "fingerhard_final",
    "hopper_final", "walker_final", "half_final",
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wandb_final_runs.json")


def main():
    api = wandb.Api(timeout=180)
    out = {}
    for p in PROJECTS:
        rows = []
        for r in api.runs(f"{ENTITY}/{p}"):
            d = {
                "id": r.id, "name": r.name, "state": r.state,
                "created_at": str(r.created_at), "tags": list(r.tags), "group": r.group,
                "config": {k: v for k, v in r.config.items() if not k.startswith("_")},
            }
            m = r.metadata or {}
            d["program"] = m.get("program")
            d["args"] = m.get("args")
            d["codePath"] = m.get("codePathLocal") or m.get("codePath")
            d["git_commit"] = (m.get("git") or {}).get("commit")
            d["startedAt"] = m.get("startedAt")
            s = dict(r.summary)
            d["summary"] = {k: v for k, v in s.items()
                            if isinstance(v, (int, float, str, bool)) and not k.startswith("_")}
            d["_step"] = s.get("_step")
            d["_runtime"] = s.get("_runtime")
            rows.append(d)
        out[p] = rows
        print("done", p, len(rows), flush=True)
    with open(os.path.abspath(OUT), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
