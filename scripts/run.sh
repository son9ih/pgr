#!/usr/bin/env bash
# Launch one algorithm on one (or all) task(s) for a set of seeds.
#
#   bash scripts/run.sh ours hopper                 # seeds 0..4, GPU 0
#   bash scripts/run.sh pgr-rnd half 0 1 2          # seeds 0,1,2
#   GPUS="0 1 2 3" bash scripts/run.sh ser all      # all 8 tasks, round-robin GPUs
#   DRY=1 bash scripts/run.sh ours walker           # print commands, run nothing
#   bash scripts/run.sh ours walker 0 -- --alpha_rtb 4.0 --no_ddim   # extra flags
#
# Per-task hyperparameters (gin file, cond_top_frac, alpha_rtb, accumulation_steps,
# num_posterior_epochs, ft_clip_grad, epochs) come from
# synther/online/env_defaults.py -- do not repeat them here.
set -uo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

OURS=synther/online/online_cond_ddpm_ori.py
BASE=synther/online/online_cond_origin_baseline.py

ALL_ENVS="quad cheetah reacher fingereasy fingerhard hopper walker half"
GPUS="${GPUS:-0}"
SEEDS_DEFAULT="0 1 2 3 4"
LOGDIR="${LOGDIR:-exp_logs}"
DRY="${DRY:-0}"

usage() {
    cat <<'USAGE'
usage: bash scripts/run.sh <algo> <env|all> [seed ...] [-- <extra flags>]

  algo  ours | ours-rnd | pgr | pgr-rnd | ser | sac | redq
  env   quad cheetah reacher fingereasy fingerhard hopper walker half | all
  seed  defaults to "0 1 2 3 4"

  env vars: GPUS="0 1"  LOGDIR=exp_logs  DRY=1
USAGE
}

case "${1:-}" in
    ours|ours-curiosity) SCRIPT=$OURS; FLAGS=(--novelty_measure curiosity) ;;
    ours-rnd)            SCRIPT=$OURS; FLAGS=(--novelty_measure rnd) ;;
    pgr|pgr-curiosity)   SCRIPT=$BASE; FLAGS=(--novelty_measure curiosity) ;;
    pgr-rnd)             SCRIPT=$BASE; FLAGS=(--novelty_measure rnd) ;;
    ser)                 SCRIPT=$BASE; FLAGS=(--synther) ;;
    sac)                 SCRIPT=$BASE; FLAGS=(--sac) ;;
    redq)                SCRIPT=$BASE; FLAGS=(--redq) ;;
    -h|--help|"")        usage; exit 0 ;;
    *) echo "unknown algo: $1" >&2; usage >&2; exit 1 ;;
esac
ALGO=$1; shift

ENVS="${1:?missing env (use 'all' for every task)}"; shift
[ "$ENVS" = "all" ] && ENVS=$ALL_ENVS

# Remaining args: seeds, then anything after `--` is forwarded to python.
SEEDS=""
EXTRA=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; EXTRA=("$@"); break; fi
    SEEDS="$SEEDS $1"; shift
done
[ -z "${SEEDS// /}" ] && SEEDS=$SEEDS_DEFAULT

read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p "$LOGDIR"

i=0
for env in $ENVS; do
    for seed in $SEEDS; do
        gpu=${GPU_ARR[$((i % NGPU))]}
        log="$LOGDIR/${ALGO}_${env}_s${seed}.log"
        echo "[gpu $gpu] $ALGO $env seed $seed -> $log"
        if [ "$DRY" = "1" ]; then
            echo "  CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT --env $env ${FLAGS[*]} --seed $seed --wandb ${EXTRA[*]}"
        else
            CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
                --env "$env" "${FLAGS[@]}" --seed "$seed" --wandb \
                ${EXTRA[@]+"${EXTRA[@]}"} > "$log" 2>&1 &
        fi
        i=$((i + 1))
        # Keep at most one job per GPU in flight.
        if [ "$DRY" != "1" ] && [ $((i % NGPU)) -eq 0 ]; then wait; fi
    done
done
[ "$DRY" != "1" ] && wait
echo "done: $i run(s)"
