export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/son9ih/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# ================================
# key parameters: 
# ================================
# finetune_lr: learning rate for fine-tuning the diffusion model
# beta: guidance strength for the diffusion model
# backprop_iters: number of iterations for finetuning the diffusion model
# top_reward_exclude_ratio: ratio of top rewards to exclude when computing reward statistics for normalization and threshold for RTB
# uniform: whether to use uniform sampling of off-policy data
# amplify: amplification factor for weighted sampling of off-policy data
# target_rnd_every: frequency of updating the target RND instead of original RND frozen network
# sample_freq: frequency of training Off-policy compared to On-policy when doing RTB
# ================================

# 기본 코드
CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin \
    --seed 0 --algorithm Ours --wandb \
    --finetune_lr 1e-4 --beta 1.0 --backprop_iters 100 --amplify 2.0 --target_rnd_every 0 --sample_freq 1 --top_reward_exclude_ratio 0.2 &



#!/bin/bash

# 사용 가능한 GPU 번호 배열
GPUS=(0 1 2 3 4 5)
NUM_GPUS=${#GPUS[@]}
GPU_IDX=0

for beta in 1.0; do
    for sample_freq in 1 2 4; do
        for top_reward_exclude_ratio in 0.1 0.2; do
            
            # 현재 할당할 GPU 선택
            CURRENT_GPU=${GPUS[$GPU_IDX]}
            
            echo "Running on GPU $CURRENT_GPU: sample_freq=$sample_freq, exclude_ratio=$top_reward_exclude_ratio"
            
            # 프로세스 실행
            CUDA_VISIBLE_DEVICES=$CURRENT_GPU python synther/online/online_cond.py \
                --env reacher-hard-v0 \
                --gin_config_files config/online/sac_cond_synther_dmc.gin \
                --seed 0 --algorithm Ours --wandb \
                --finetune_lr 1e-4 --beta $beta --backprop_iters 200 \
                --amplify 1.0 --target_rnd_every 0 \
                --sample_freq $sample_freq \
                --top_reward_exclude_ratio $top_reward_exclude_ratio &
            
            # GPU 인덱스 업데이트 및 동기화 제어
            ((GPU_IDX++))
            
            # 만약 모든 GPU(0-5)를 다 썼다면, 실행 중인 프로세스가 끝날 때까지 대기
            if [ $GPU_IDX -eq $NUM_GPUS ]; then
                wait
                GPU_IDX=0
                echo "Batch finished, moving to next set of GPUs..."
            fi
            
            sleep 2
        done
    done
done

# 남은 모든 프로세스가 종료될 때까지 대기
wait
echo "All experiments completed."


# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin \
#         --seed 0 --algorithm Ours --wandb \
#         --finetune_lr 1e-4 --beta 5.0 --backprop_iters 100 --amplify 2.0 --target_rnd_every 0 --sample_freq 1 --top_reward_exclude_ratio 0.25 &


# 일반코드
# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 10.0 --ft_batch_size 1024 --backprop_iters 100 --amplify 1.0 --wandb &
# sleep 5
# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 50.0 --ft_batch_size 1024 --backprop_iters 100 --amplify 3.0 --wandb &
# sleep 5
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 5.0 --ft_batch_size 1024 --backprop_iters 100 --amplify 3.0 --wandb &
# # sleep 5
# # CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 10.0 --ft_batch_size 1024 --backprop_iters 30 &
# # sleep 5
# # CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 100.0 --ft_batch_size 1024 --backprop_iters 30 &


# RTB running code for dmc

# 환경마다 최적의 beta/ backprop_iters 값 찾기

# for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0; do
#     for algorithm in Ours; do
        
#         case $algorithm in
#             SAC)
#                 GIN_FILE=config/online/sac.gin
#                 ;;
#             REDQ)
#                 GIN_FILE=config/online/redq.gin
#                 ;;
#             SER | PGRrnd | Ours)
#                 GIN_FILE=config/online/sac_cond_synther_dmc.gin
#                 ;;
#             *)
#                 echo "Unknown algorithm: $algorithm"
#                 exit 1
#                 ;;
#         esac

#         case $env in 
#             quadruped-walk-v0)
#                 COND_TOP_FRAC=0.1
#                 ;;
#             cheetah-run-v0 | reacher-hard-v0)
#                 COND_TOP_FRAC=0.25
#                 ;;
#         esac

#         # beta 값 조절 필요
#         for beta in 1 3 5; do
#             for seed in 0 1 2; do
#                 echo "Running $algorithm on $env with seed $seed, beta $beta using $GIN_FILE"
                
#                 # CUDA_VISIBLE_DEVICES 설정은 상황에 맞게 조절 필요 (현재는 seed 값 사용)
#                 CUDA_VISIBLE_DEVICES=$seed python synther/online/online_cond.py \
#                 --env $env \
#                 --gin_config_files $GIN_FILE \
#                 --gin_params "redq_sac.cond_top_frac = $COND_TOP_FRAC" \
#                 --seed $seed \
#                 --beta $beta \
#                 --wandb \
#                 --algorithm $algorithm &
                
#                 sleep 2
#             done
#             wait # 각 beta 실험 세트가 끝날 때까지 기다림 (optional)
#         done
#     done
# done



# # RTB running code for mujoco

# # 환경마다 최적의 beta 값 찾기
# for env in Hopper-v2 Walker2d-v2 HalfCheetah-v2; do
#     for algorithm in Ours; do
        
#         case $algorithm in
#             SAC)
#                 GIN_FILE=config/online/sac.gin
#                 ;;
#             REDQ)
#                 GIN_FILE=config/online/redq.gin
#                 ;;
#             SER | PGRrnd | Ours)
#                 GIN_FILE=config/online/sac_cond_synther_dmc.gin
#                 ;;
#             *)
#                 echo "Unknown algorithm: $algorithm"
#                 exit 1
#                 ;;
#         esac

#         case $env in 
#             Hopper-v2)
#                 COND_TOP_FRAC=0.25
#                 ;;
#             Walker2d-v2)
#                 COND_TOP_FRAC=0.25
#                 ;;
#             HalfCheetah-v2)
#                 COND_TOP_FRAC=0.25
#                 ;;
#         esac

#         # beta 값 조절 필요
#         for beta in 1 3 5; do
#             for seed in 0 1 2; do
#                 echo "Running $algorithm on $env with seed $seed, beta $beta using $GIN_FILE"
                
#                 # CUDA_VISIBLE_DEVICES 설정은 상황에 맞게 조절 필요 (현재는 seed 값 사용)
#                 CUDA_VISIBLE_DEVICES=$seed python synther/online/online_cond.py \
#                 --env $env \
#                 --gin_config_files $GIN_FILE \
#                 --gin_params "redq_sac.cond_top_frac = $COND_TOP_FRAC" \
#                 --seed $seed \
#                 --beta $beta \
#                 --wandb \
#                 --algorithm $algorithm &
                
#                 sleep 2
#             done
#             wait # 각 beta 실험 세트가 끝날 때까지 기다림 (optional)
#         done
#     done
# done



# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/son9ih/.mujoco/mujoco210/bin 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# for env in reacher-hard-v0; do
#     for algorithm in Ours; do
        
#         case $algorithm in
#             SAC)
#                 GIN_FILE=config/online/sac.gin
#                 ;;
#             REDQ)
#                 GIN_FILE=config/online/redq.gin
#                 ;;
#             SER | PGRrnd | Ours)
#                 GIN_FILE=config/online/sac_cond_synther_dmc.gin
#                 # for debugging, try this in advance!
#                 # GIN_FILE=config/online/sac_cond_synther_dmc2.gin
#                 ;;
#             *)
#                 echo "Unknown algorithm: $algorithm"
#                 exit 1
#                 ;;
#         esac

#         case $env in 
#             quadruped-walk-v0)
#                 COND_TOP_FRAC=0.1
#                 ;;
#             cheetah-run-v0 | reacher-hard-v0)
#                 COND_TOP_FRAC=0.25
#                 ;;
#         esac

#         # beta 값 조절 필요
#         for beta in 1 2 3; do
#             for seed in 0 1 2 3; do
#                 echo "Running $algorithm on $env with seed $seed, beta $beta using $GIN_FILE"
                
#                 # CUDA_VISIBLE_DEVICES 설정은 상황에 맞게 조절 필요 (현재는 seed 값 사용)
#                 CUDA_VISIBLE_DEVICES=$seed python synther/online/online_cond.py \
#                 --env $env \
#                 --gin_config_files $GIN_FILE \
#                 --gin_params "redq_sac.cond_top_frac = $COND_TOP_FRAC" \
#                 --seed $seed \
#                 --beta $beta \
#                 --wandb \
#                 --backprop_iters 100 \
#                 --ft_batch_size 1024 \
#                 --accumulation_steps 4 \
#                 --algorithm $algorithm &
                
#                 sleep 2
#             done
#             wait # 각 beta 실험 세트가 끝날 때까지 기다림 (optional)
#         done
#     done
# done