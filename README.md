# Robust Exploration through Generative Replay

## Installation

Run the following:

```bash
conda create -n regr python=3.8
conda activate regr
pip install -r requirements.txt
```

Our code is tested on **Python 3.8**.

If you do not have MuJoCo installed, follow the instructions here:  
https://github.com/openai/mujoco-py#install-mujoco


---

# Running

This repository supports several algorithms used in our experiments, including **SAC**, **REDQ**, **SER**, **PGR**, and our method **REGR**.

All experiments are executed through the following scripts:

```
synther/online/online_cond.py
synther/online/online_cond_regr.py
```

Configuration files are specified via **gin-config**.

The environments used in our experiments are:

```
quadruped-walk-v0
cheetah-run-v0
reacher-hard-v0
HalfCheetah-v2
Walker2d-v2
Hopper-v2
finger-turn_hard-v0
```

Experiments can optionally log results using **Weights & Biases** by enabling the `--wandb` flag.


---

# Running Baselines

## SAC Baseline

```bash
for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0 HalfCheetah-v2 Walker2d-v2 Hopper-v2 finger-turn_hard-v0; do

    GIN_FILE="config/online/sac.gin"

    CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py \
        --wandb \
        --env $env \
        --gin_config_files $GIN_FILE \
        --seed 0 \
        --sac &

done
```


---

## REDQ Baseline

```bash
for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0 HalfCheetah-v2 Walker2d-v2 Hopper-v2 finger-turn_hard-v0; do

    GIN_FILE="config/online/redq.gin"

    CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py \
        --wandb \
        --env $env \
        --gin_config_files $GIN_FILE \
        --seed 0 \
        --redq &

done
```


---

## SER (SynthER)

```bash
for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0 HalfCheetah-v2 Walker2d-v2 Hopper-v2 finger-turn_hard-v0; do

    case "$env" in
        "quadruped-walk-v0" | "cheetah-run-v0" | "reacher-hard-v0" | "finger-turn_hard-v0")
            GIN_FILE="config/online/sac_cond_synther_dmc.gin"
            ;;
        *)
            GIN_FILE="config/online/sac_cond_synther_openai.gin"
            ;;
    esac

    CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py \
        --wandb \
        --env $env \
        --gin_config_files $GIN_FILE \
        --seed 0 \
        --synther \
        --ddim &

done
```


---

## PGR (Prioritized Generative Replay)

```bash
for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0 HalfCheetah-v2 Walker2d-v2 Hopper-v2 finger-turn_hard-v0; do
    for novelty_measure in curiosity rnd; do
        for cond_top_frac in 0.25; do
            for cfg_scale in 2.0; do

                case "$env" in
                    "quadruped-walk-v0" | "cheetah-run-v0" | "reacher-hard-v0" | "finger-turn_hard-v0")
                        GIN_FILE="config/online/sac_cond_synther_dmc.gin"
                        ;;
                    *)
                        GIN_FILE="config/online/sac_cond_synther_openai.gin"
                        ;;
                esac

                CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py \
                    --wandb \
                    --env $env \
                    --gin_config_files $GIN_FILE \
                    --gin_params "redq_sac.cond_top_frac = $cond_top_frac" "redq_sac.cfg_scale = $cfg_scale" \
                    --seed 0 \
                    --novelty_measure $novelty_measure \
                    --ddim &

            done
        done
    done
done
```


---

# Running REGR (Ours)

```bash
for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0 HalfCheetah-v2 Walker2d-v2 Hopper-v2 finger-turn_hard-v0; do
    for novelty_measure in curiosity rnd; do
        for alpha_rtb in 1.0; do

            case "$env" in
                "quadruped-walk-v0")
                    POST_EPOCH=150
                    GIN_FILE="config/online/sac_cond_synther_dmc.gin"
                    ;;
                "HalfCheetah-v2" | "Hopper-v2" | "Walker2d-v2")
                    POST_EPOCH=100
                    GIN_FILE="config/online/sac_cond_synther_openai.gin"
                    ;;
                "reacher-hard-v0" | "cheetah-run-v0")
                    POST_EPOCH=100
                    GIN_FILE="config/online/sac_cond_synther_dmc.gin"
                    ;;
                *)
                    POST_EPOCH=100
                    GIN_FILE="config/online/sac_cond_synther_dmc.gin"
                    ;;
            esac

            CUDA_VISIBLE_DEVICES=1 python synther/online/online_cond_regr.py \
                --wandb \
                --env $env \
                --gin_config_files $GIN_FILE \
                --seed 0 \
                --alpha_rtb $alpha_rtb \
                --training_posterior both \
                --novelty_measure $novelty_measure \
                --num_posterior_epochs $POST_EPOCH \
                --accumulation_steps 1 \
                --ddim &

        done
    done
done
```


---

## License and Acknowledgements

This codebase inherits all licenses from the public release of [SynthER](https://github.com/conglu1997/SynthER), [PGR](https://github.com/renwang435/pgr), [RTB](https://github.com/GFNOrg/diffusion-finetuning).