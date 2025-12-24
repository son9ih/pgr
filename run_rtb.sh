export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/son9ih/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


# key parameters: beta, backprop_iters, ft_batch_size

# 일반코드
CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 1.0 --ft_batch_size 1024 --backprop_iters 30 --wandb &
sleep 5
CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 10.0 --ft_batch_size 1024 --backprop_iters 30 --wandb &
sleep 5
CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 100.0 --ft_batch_size 1024 --backprop_iters 30 --wandb &
# sleep 5
# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 10.0 --ft_batch_size 1024 --backprop_iters 30 &
# sleep 5
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 100.0 --ft_batch_size 1024 --backprop_iters 30 &


# RTB running code for dmc

# # 환경마다 최적의 beta 값 찾기
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