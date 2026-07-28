"""Download eval/AverageTestEpRet learning curves for the synthetic-buffer capacity
ablation into a long-format CSV.

Three arms, all reacher-hard / Ours+curiosity / posterior_param=residual, differing
only in the capacity of the synthetic (diffusion) replay buffer:

  50k, 200k  -> project reacher_buffer_ablation  (this ablation)
  1M         -> project reacher_final            (the recorded runs)

Usage:  python exp/tools/fetch_curves.py [-o exp/data/buffer_ablation_reacher.csv]
"""
import argparse
import csv
import os

import wandb

ENTITY = 'gda-for-orl'
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'data', 'buffer_ablation_reacher.csv')
METRIC = 'eval/AverageTestEpRet'

# (label, project, capacity) -- capacity is matched against config.diffusion_buffer_size
ARMS = [
    ('50k', 'reacher_buffer_ablation', 50_000),
    ('200k', 'reacher_buffer_ablation', 200_000),
    ('1M', 'reacher_final', 1_000_000),
]


def is_ours_residual(run):
    """The Ours arm with the residual posterior parameterization.

    Ours vs the baselines cannot be told apart from the config -- both entry points log
    the same key set, so PGR runs also carry alpha_rtb -- so discriminate on the program
    that produced the run, falling back to the run name.

    The recorded reacher_final runs predate --posterior_param; a missing key means
    residual, which is what the code did back then.
    """
    program = ((run.metadata or {}).get('program') or '')
    is_ours = ('ddpm_ori' in program) if program else ('_Ours+' in run.name)
    if not is_ours:
        return False
    return run.config.get('posterior_param', 'residual') == 'residual'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=DEFAULT_OUT)
    ap.add_argument('--novelty', default='curiosity')
    a = ap.parse_args()

    api = wandb.Api(timeout=180)
    rows = []
    for label, project, capacity in ARMS:
        runs = api.runs(f'{ENTITY}/{project}')
        picked = []
        for r in runs:
            cfg = r.config
            if cfg.get('novelty_measure') != a.novelty:
                continue
            if int(cfg.get('diffusion_buffer_size', -1)) != capacity:
                continue
            if not is_ours_residual(r):
                continue
            picked.append(r)
        print(f'{label:>5}  {project:<26} {len(picked)} run(s): '
              f'seeds {sorted(str(r.config.get("seed")) for r in picked)}')
        for r in picked:
            seed = r.config.get('seed')
            n = 0
            for row in r.scan_history(keys=['_step', METRIC]):
                v = row.get(METRIC)
                if v is None:
                    continue
                rows.append({'capacity': label, 'capacity_value': capacity,
                             'seed': seed, 'epoch': row['_step'], 'return': v,
                             'run_id': r.id, 'run_name': r.name})
                n += 1
            print(f'         seed {seed}: {n} points')

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['capacity', 'capacity_value', 'seed',
                                          'epoch', 'return', 'run_id', 'run_name'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows -> {out}')


if __name__ == '__main__':
    main()
