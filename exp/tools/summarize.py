"""Aggregate exp/data/wandb_final_runs.json into a per-task / per-method table.

Usage:  python exp/tools/summarize.py            # human-readable
        python exp/tools/summarize.py --md       # markdown tables
        python exp/tools/summarize.py --cmds     # unique command lines per project
"""
import argparse
import collections
import json
import os
import shlex
import statistics as st

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wandb_final_runs.json")
ORDER = ["quad_final", "cheetah_final", "reacher_final", "fingereasy_final",
         "fingerhard_final", "hopper_final", "walker_final", "half_final"]
METRICS = [("Return", "eval/AverageTestEpRet", "{:.1f}"),
           ("DynMSE", "eval/DynMSE_mean", "{:.2f}"),
           ("StateEnt", "eval/StateEnt", "{:.2f}"),
           ("NormQBias", "eval/AverageNormQBias", "{:.2f}")]


def method_of(r):
    """Recover the method label from config + program (run names are not parseable
    for envs whose name contains '_', e.g. finger-turn_hard-v0)."""
    c, prog = r["config"], (r.get("program") or "")
    nm = c.get("novelty_measure", "curiosity")
    if "ddpm_ori" in prog:
        return f"Ours+{nm} (A={c.get('alpha_rtb')}, clip={c.get('ft_clip_grad')})"
    if c.get("synther"):
        return "SER"          # SER ignores novelty_measure
    return f"PGR+{nm}"


def agg(vals, fmt):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "-"
    m = fmt.format(st.mean(vals))
    s = fmt.format(st.stdev(vals)) if len(vals) > 1 else fmt.format(0)
    return f"{m} ± {s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--cmds", action="store_true")
    a = ap.parse_args()
    d = json.load(open(os.path.abspath(DATA)))

    if a.cmds:
        for p in ORDER:
            print(f"\n### {p}")
            groups = collections.defaultdict(list)
            for r in d[p]:
                args = list(r.get("args") or [])
                if "--seed" in args:
                    i = args.index("--seed")
                    del args[i:i + 2]
                groups[(r.get("program"), tuple(args))].append(r)
            for (prog, args), members in groups.items():
                seeds = sorted(str(m["config"].get("seed")) for m in members)
                print(f"\n<!-- {method_of(members[0])} | seeds {seeds} | "
                      f"commit {(members[0].get('git_commit') or '')[:7]} -->")
                # wandb stores argv pre-split, so re-quote (e.g. --gin_params
                # "redq_sac.cond_top_frac = 0.25" is a single token with spaces).
                print("```bash")
                print(f"python {prog} \\\n  " + " ".join(shlex.quote(a) for a in args))
                print("```")
        return

    for p in ORDER:
        rows = d[p]
        env = rows[0]["config"].get("env_name")
        ep = rows[0]["config"].get("epochs")
        g = collections.defaultdict(list)
        for r in rows:
            g[method_of(r)].append(r)
        if a.md:
            print(f"\n### {p} — `{env}` ({ep} epochs)\n")
            print("| method | n | seeds | " + " | ".join(m[0] for m in METRICS) + " | wall-clock |")
            print("|---|---|---|" + "---|" * (len(METRICS) + 1))
            for k in sorted(g):
                v = g[k]
                seeds = ",".join(sorted(str(x["config"].get("seed")) for x in v))
                cells = [agg([x["summary"].get(key) for x in v], fmt) for _, key, fmt in METRICS]
                h = [x.get("_runtime") for x in v if isinstance(x.get("_runtime"), (int, float))]
                cells.append(f"{st.mean(h) / 3600:.1f} h" if h else "-")
                print(f"| {k} | {len(v)} | {seeds} | " + " | ".join(cells) + " |")
        else:
            print("=" * 78)
            print(f"{p}  env={env}  epochs={ep}")
            for k in sorted(g):
                v = g[k]
                seeds = ",".join(sorted(str(x["config"].get("seed")) for x in v))
                cells = "  ".join(f"{n}={agg([x['summary'].get(key) for x in v], fmt)}"
                                  for n, key, fmt in METRICS)
                print(f"  {k:<46} n={len(v)} seeds={seeds:<10} {cells}")


if __name__ == "__main__":
    main()
