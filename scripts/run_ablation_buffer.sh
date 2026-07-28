#!/usr/bin/env bash
# Ablation: capacity of the synthetic (diffusion) replay buffer, on reacher-hard.
#
#   bash scripts/run_ablation_buffer.sh          # launch all 10 runs
#   DRY=1 bash scripts/run_ablation_buffer.sh    # print the plan only
#
# 10 runs = {50k, 200k} capacity x 5 seeds, pinned to GPUs 0-3 as (3,3,2,2). The 50k
# arm goes on the two GPUs that carry three runs, since it is the cheaper arm (one
# 50k sampling batch instead of two 100k ones).
#
# `--posterior_param residual` matches the recorded reacher_final 1M runs, so the
# three curves in the figure differ only in buffer capacity. Everything else for
# reacher-hard comes from synther/online/env_defaults.py (dmc.gin, cond_top_frac
# 0.25, alpha_rtb 10.0, ft_clip_grad 1.0, accumulation_steps 4, 100 epochs).
#
# `num_samples` follows the capacity: the retained samples are i.i.d. from the
# posterior either way, so generating 1M and keeping the last 50k is the same
# distribution as generating 50k -- just 20x more sampling compute.
set -uo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

SCRIPT=synther/online/online_cond_ddpm_ori.py
ENV=reacher
PROJECT=reacher_buffer_ablation
LOGDIR="${LOGDIR:-exp_logs/buffer_ablation}"
DRY="${DRY:-0}"

# "gpu capacity seed" per run
JOBS=(
    # GPU 0 -- 3 runs, 50k
    "0 50000  0"
    "0 50000  1"
    "0 50000  2"
    # GPU 1 -- 3 runs, rest of 50k plus one 200k
    "1 50000  3"
    "1 50000  4"
    "1 200000 0"
    # GPU 2 -- 2 runs, 200k
    "2 200000 1"
    "2 200000 2"
    # GPU 3 -- 2 runs, 200k
    "3 200000 3"
    "3 200000 4"
)

mkdir -p "$LOGDIR"

launch_gpu() {   # $1 = gpu id; runs that GPU's jobs sequentially
    local gpu=$1
    for job in "${JOBS[@]}"; do
        read -r g cap seed <<< "$job"
        [ "$g" = "$gpu" ] || continue
        # sample_batch_size must divide num_samples (asserted in the entry point).
        local batch=100000
        [ "$cap" -lt 100000 ] && batch=$cap
        local log="$LOGDIR/reacher_buf$((cap / 1000))k_s${seed}.log"
        echo "[gpu $gpu] capacity=$cap seed=$seed -> $log"
        if [ "$DRY" = "1" ]; then
            echo "    CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT --env $ENV --seed $seed --wandb" \
                 "--wandb_project $PROJECT --posterior_param residual" \
                 "--num_samples $cap --sample_batch_size $batch" \
                 "--gin_params \"redq_sac.diffusion_buffer_size = $cap\""
        else
            CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" --env "$ENV" --seed "$seed" --wandb \
                --wandb_project "$PROJECT" --posterior_param residual \
                --num_samples "$cap" --sample_batch_size "$batch" \
                --gin_params "redq_sac.diffusion_buffer_size = $cap" \
                > "$log" 2>&1
        fi
    done
}

for gpu in 0 1 2 3; do
    if [ "$DRY" = "1" ]; then
        launch_gpu "$gpu"
    else
        launch_gpu "$gpu" &          # GPUs run in parallel, jobs within a GPU serially
    fi
done
[ "$DRY" != "1" ] && wait
echo "done"
