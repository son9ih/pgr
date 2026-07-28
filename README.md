# Prioritized Generative Replay

[Renhao Wang](https://renwang435.github.io/), [Kevin Frans](https://kvfrans.com/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), [Alexei A. Efros](http://people.eecs.berkeley.edu/~efros/)

[[`arXiv`](https://arxiv.org/abs/2410.18082)] [[`BibTeX`](#Citing)]

## Installation

This repository is based heavily off the release code of [SynthER](https://github.com/conglu1997/SynthER?tab=readme-ov-file#setup). The instructions for setting up their environment are reproduced below: 

To install, clone the repository and run the following:

```bash 
git submodule update --init --recursive
pip install -r requirements.txt
```

Our code is tested on Python 3.8.
If you don't have MuJoCo installed, follow the instructions here: https://github.com/openai/mujoco-py#install-mujoco.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

## Running Instructions

Every per-task hyperparameter (gin config, `cond_top_frac`, `alpha_rtb`,
`accumulation_steps`, `num_posterior_epochs`, `ft_clip_grad`, epoch count) is resolved
from `--env` by [synther/online/env_defaults.py](synther/online/env_defaults.py), so a
run only names the task, the algorithm and the seed.

### With the launcher

```bash
bash scripts/run.sh <algo> <env|all> [seed ...] [-- <extra flags>]
```

| argument | values |
|---|---|
| `algo` | `ours` `ours-rnd` `pgr` `pgr-rnd` `ser` `sac` `redq` |
| `env` | `quad` `cheetah` `reacher` `fingereasy` `fingerhard` `hopper` `walker` `half`, or `all` |
| `seed` | defaults to `0 1 2 3 4` |

```bash
bash scripts/run.sh ours hopper                  # Ours+curiosity, seeds 0-4, GPU 0
bash scripts/run.sh pgr-rnd half 0 1 2           # PGR+rnd, seeds 0,1,2
GPUS="0 1 2 3" bash scripts/run.sh ser all 0     # SER on all 8 tasks, 4 GPUs
DRY=1 bash scripts/run.sh ours-rnd quad          # print the commands only
bash scripts/run.sh ours walker 0 -- --alpha_rtb 4.0   # override one default
```

One job per GPU at a time; logs go to `exp_logs/<algo>_<env>_s<seed>.log`.

### Directly

```bash
# Ours -- PGR with RTB posterior fine-tuning of the DDPM prior
python synther/online/online_cond_ddpm_ori.py --env hopper \
       --novelty_measure curiosity --seed 0 --wandb

# PGR
python synther/online/online_cond_origin_baseline.py --env hopper \
       --novelty_measure curiosity --seed 0 --wandb

# SER / SAC / REDQ
python synther/online/online_cond_origin_baseline.py --env hopper --synther --seed 0 --wandb
python synther/online/online_cond_origin_baseline.py --env hopper --sac     --seed 0 --wandb
python synther/online/online_cond_origin_baseline.py --env hopper --redq    --seed 0 --wandb
```

`--env` takes either the short alias (`half`) or the gym id (`HalfCheetah-v2`).
Anything given on the command line overrides the per-task default, and the resolved
values are printed at startup:

```
[env_defaults] env=Hopper-v2  gin_config_files=['config/online/sac_cond_synther_openai.gin']
  epochs=100  cond_top_frac=0.25  accumulation_steps=4  num_posterior_epochs=100
  ft_clip_grad=1.0  alpha_rtb=2.0
```

See [exp/04_script.md](exp/04_script.md) for the full per-task table, the list of
argparse defaults, and how to change them. Recorded runs and results live in
[exp/01_experiments.md](exp/01_experiments.md).

## <a name="Citing"></a>Citing PGR

```BibTeX
@inproceedings{wang2025prioritized,
  title={Prioritized Generative Replay},
  author={Renhao Wang and Kevin Frans and Pieter Abbeel and Sergey Levine and Alexei A Efros},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=5IkDAfabuo}
}
```

## License and Acknowledgements

This codebase inherits all licenses from the public release of [SynthER](https://github.com/conglu1997/SynthER).
