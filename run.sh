export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/son9ih/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

GPUS=(5 6 7)
NUM_GPUS=${#GPUS[@]}
COUNT=0


# # # SER
# for env in reacher-hard-v0 quadruped-walk-v0 cheetah-run-v0; do
#     for algorithm in SER; do
#         case $algorithm in 
#             SAC) GIN_FILE=config/online/sac.gin
#             REDQ) GIN_FILE=config/online/redq.gin ;;
#             # # for debugging, try this in advance!
#             # SER | PGRrnd | Ours) GIN_FILE=config/online/sac_cond_synther_dmc2.gin ;;
#             SER | PGRrnd | Ours | PGR) GIN_FILE=config/online/sac_cond_synther_dmc.gin ;;
#             *) echo "Unknown algorithm: $algorithm"; exit 1 ;;
#         esac

#         case $env in
#             quadruped-walk-v0) COND_TOP_FRAC=0.1 ;;
#             cheetah-run-v0 | reacher-hard-v0) COND_TOP_FRAC=0.25 ;;
#         esac

#             for seed in 0 1 2 3 4; do
#                             # 현재 순서에 맞는 GPU ID 할당 (0, 1, 2, 3, 4, 5 순환)
#                             GPU_ID=${GPUS[$((COUNT % NUM_GPUS))]}
                            
#                             echo "Running $algorithm on $env (GPU $GPU_ID): seed=$seed"

#                             CUDA_VISIBLE_DEVICES=$GPU_ID python synther/online/online_cond_ddpm.py \
#                                 --env $env \
#                                 --gin_config_files $GIN_FILE \
#                                 --gin_params "redq_sac.cond_top_frac = $COND_TOP_FRAC" \
#                                 --cond_top_frac $COND_TOP_FRAC \
#                                 --seed $seed \
#                                 --num_prior_epochs 100000 \
#                                 --diffusion_steps 500 \
#                                 --num_samples 1000000 \
#                                 --wandb \
#                                 --train_batch_size 1024 \
#                                 --ft_batch_size 1024 \
#                                 --algorithm $algorithm &
                            
#                             sleep 2
#                             ((COUNT++))

#                             # 작업이 다 차면 모두 끝날 때까지 대기
#                             # if [ $((COUNT % NUM_GPUS)) -eq 0 ]; then
#                             #     echo "All GPUs are busy. Waiting for this batch to finish..."
#                             #     wait
#                             # fi
#             done
#     done
# done

# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond_ddpm.py \
#                                 --env reacher-hard-v0 \
#                                 --gin_config_files config/online/sac_cond_synther_dmc2.gin \
#                                 --gin_params "redq_sac.cond_top_frac = 0.25" \
#                                 --cond_top_frac 0.25 \
#                                 --seed 0 \
#                                 --num_prior_epochs 5000 \
#                                 --diffusion_steps 200 \
#                                 --num_samples 100000 \
#                                 --train_batch_size 1024 \
#                                 --novelty_measure curiosity \
#                                 --algorithm SER &


# # PGR
# for env in reacher-hard-v0 quadruped-walk-v0 cheetah-run-v0; do
# # for env in reacher-hard-v0; do
#     for algorithm in PGR; do
#         case $algorithm in 
#             SAC) GIN_FILE=config/online/sac.gin ;;
#             REDQ) GIN_FILE=config/online/redq.gin ;;
#             # # for debugging, try this in advance!
#             # SER | PGRrnd | Ours) GIN_FILE=config/online/sac_cond_synther_dmc2.gin ;;
#             SER | PGRrnd | Ours | PGR) GIN_FILE=config/online/sac_cond_synther_dmc.gin ;;
#             *) echo "Unknown algorithm: $algorithm"; exit 1 ;;
#         esac

#         case $env in
#             quadruped-walk-v0) COND_TOP_FRAC=0.1 ;;
#             cheetah-run-v0 | reacher-hard-v0) COND_TOP_FRAC=0.25 ;;
#         esac

#             for seed in 0 1 2 3 4; do
#                 for novelty_measure in curiosity rnd; do
#                             # 현재 순서에 맞는 GPU ID 할당 (0, 1, 2, 3, 4, 5 순환)
#                             GPU_ID=${GPUS[$((COUNT % NUM_GPUS))]}
                            
#                             echo "Running $algorithm on $env (GPU $GPU_ID): seed=$seed"

#                             CUDA_VISIBLE_DEVICES=$GPU_ID python synther/online/online_cond_ddpm.py \
#                                 --env $env \
#                                 --gin_config_files $GIN_FILE \
#                                 --gin_params "redq_sac.cond_top_frac = $COND_TOP_FRAC" \
#                                 --cond_top_frac $COND_TOP_FRAC \
#                                 --seed $seed \
#                                 --num_prior_epochs 100000 \
#                                 --diffusion_steps 500 \
#                                 --num_samples 1000000 \
#                                 --wandb \
#                                 --novelty_measure $novelty_measure \
#                                 --train_batch_size 1024 \
#                                 --ft_batch_size 1024 \
#                                 --algorithm $algorithm &

                            
                            
#                             sleep 2
#                             ((COUNT++))

#                             # 작업이 다 차면 모두 끝날 때까지 대기
#                             # if [ $((COUNT % NUM_GPUS)) -eq 0 ]; then
#                             #     echo "All GPUs are busy. Waiting for this batch to finish..."
#                             #     wait
#                             # fi
#                 done
#             done
#     done
# done


# Ours
for env in reacher-hard-v0; do
# for env in reacher-hard-v0 quadruped-walk-v0; do
# for env in reacher-hard-v0; do
    for algorithm in Ours; do
        case $algorithm in 
            SAC) GIN_FILE=config/online/sac.gin ;;
            REDQ) GIN_FILE=config/online/redq.gin ;;
            SER | PGRrnd | Ours | PGR) GIN_FILE=config/online/sac_cond_synther_dmc.gin ;;
            *) echo "Unknown algorithm: $algorithm"; exit 1 ;;
        esac

        case $env in
            quadruped-walk-v0) COND_TOP_FRAC=0.1 ;;
            cheetah-run-v0 | reacher-hard-v0 | Hopper-v2 | Walker2d-v2 | HalfCheetah-v2) COND_TOP_FRAC=0.25 ;;
        esac

            # for seed in 0 1 2 3 4; do
                    # rnd: 0.1, curiosity: 
                    for alpha_rtb in 0.1; do
                        # for inter_onpolicy in 0.0; do
                        for inter_onpolicy in 0.04; do
                            # for novelty_measure in rnd eco; do
                            for novelty_measure in curiosity; do
                                for diffusion_steps in 1000; do
                                    for num_posterior_epochs in 50; do
                                        for seed in 0 1 2 3 4; do
                                        # 현재 순서에 맞는 GPU ID 할당 (0, 1, 2, 3, 4, 5 순환)
                                            GPU_ID=${GPUS[$((COUNT % NUM_GPUS))]}
                                            
                                            echo "Running $algorithm on $env (GPU $GPU_ID): seed=$seed"

                                            CUDA_VISIBLE_DEVICES=$GPU_ID python synther/online/online_cond_ddpm.py \
                                                --env $env \
                                                --gin_config_files $GIN_FILE \
                                                --gin_params "redq_sac.cond_top_frac = $COND_TOP_FRAC" \
                                                --cond_top_frac $COND_TOP_FRAC \
                                                --seed $seed \
                                                --num_prior_epochs 100000 \
                                                --num_posterior_epochs $num_posterior_epochs \
                                                --diffusion_steps $diffusion_steps \
                                                --num_samples 1000000 \
                                                --train_batch_size 256 \
                                                --ft_batch_size 1024 \
                                                --accumulation_steps 4 \
                                                --alpha_rtb $alpha_rtb \
                                                --inter_onpolicy $inter_onpolicy \
                                                --novelty_measure $novelty_measure \
                                                --wandb \
                                                --clip_reward \
                                                --ddim \
                                                --eta 1.0 \
                                                --algorithm $algorithm &

                                            
                                            
                                            sleep 2
                                            ((COUNT++))

                                            # 작업이 다 차면 모두 끝날 때까지 대기
                                            # if [ $((COUNT % NUM_GPUS)) -eq 0 ]; then
                                            #     echo "All GPUs are busy. Waiting for this batch to finish..."
                                            #     wait
                                            # fi
                                        done
                                    done
                                done
                            done
                        done
                    done
            # done
    done
done

# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond_ddpm.py \
#                                         --env reacher-hard-v0 \
#                                         --gin_config_files config/online/sac_cond_synther_dmc2.gin \
#                                         --gin_params "redq_sac.cond_top_frac = 0.25" \
#                                         --cond_top_frac 0.25 \
#                                         --seed 0 \
#                                         --num_prior_epochs 5000 \
#                                         --num_posterior_epochs 50 \
#                                         --diffusion_steps 1000 \
#                                         --num_samples 1000000 \
#                                         --train_batch_size 1024 \
#                                         --ft_batch_size 512 \
#                                         --alpha_rtb 0.1 \
#                                         --inter_onpolicy 0.04 \
#                                         --novelty_measure rnd \
#                                         --wandb \
#                                         --clip_reward \
#                                         --ddim \
#                                         --eta 1.0 \
#                                         --algorithm Ours &


# # # Ours-debugging
# # for env in reacher-hard-v0 quadruped-walk-v0 cheetah-run-v0; do
# #     for algorithm in Ours; do
# #         case $algorithm in 
# #             SAC) GIN_FILE=config/online/sac.gin ;;
# #             REDQ) GIN_FILE=config/online/redq.gin ;;
# #             # # for debugging, try this in advance!
# #             SER | PGRrnd | Ours) GIN_FILE=config/online/sac_cond_synther_dmc2.gin ;;
# #             # SER | PGRrnd | Ours | PGR) GIN_FILE=config/online/sac_cond_synther_dmc.gin ;;
# #             *) echo "Unknown algorithm: $algorithm"; exit 1 ;;
# #         esac

# #         case $env in
# #             quadruped-walk-v0) COND_TOP_FRAC=0.1 ;;
# #             cheetah-run-v0 | reacher-hard-v0) COND_TOP_FRAC=0.25 ;;
# #         esac

# #             for seed in 0 1 2 3 4; do
# #                     for alpha_rtb in 1.0 1e-1 1e-2 1e-3 1e-4 1e-5; do
# #                         for inter_onpolicy in 0.1 0.3 0.5; do
# #                             for novelty_measure in curiosity rnd; do
# #                                 # 현재 순서에 맞는 GPU ID 할당 (0, 1, 2, 3, 4, 5 순환)
# #                                 GPU_ID=${GPUS[$((COUNT % NUM_GPUS))]}
                                
# #                                 echo "Running $algorithm on $env (GPU $GPU_ID): seed=$seed"

# #                                 CUDA_VISIBLE_DEVICES=$GPU_ID python synther/online/online_cond_ddpm.py \
# #                                     --env $env \
# #                                     --gin_config_files $GIN_FILE \
# #                                     --gin_params "redq_sac.cond_top_frac = $COND_TOP_FRAC" \
# #                                     --cond_top_frac $COND_TOP_FRAC \
# #                                     --seed $seed \
# #                                     --num_prior_epochs 10000 \
# #                                     --num_posterior_epochs 50 \
# #                                     --diffusion_steps 500 \
# #                                     --num_samples 100000 \
# #                                     --alpha_rtb $alpha_rtb \
# #                                     --inter_onpolicy $inter_onpolicy \
# #                                     --novelty_measure $novelty_measure \
# #                                     --wandb \
# #                                     --algorithm $algorithm &

                                
                                
# #                                 sleep 2
# #                                 ((COUNT++))

# #                                 # 작업이 다 차면 모두 끝날 때까지 대기
# #                                 if [ $((COUNT % NUM_GPUS)) -eq 0 ]; then
# #                                     echo "All GPUs are busy. Waiting for this batch to finish..."
# #                                     wait
# #                                 fi
# #                         done
# #                     done
# #             done
# #     done
# # done



# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond_ddpm.py \
#                                     --env reacher-hard-v0 \
#                                     --gin_config_files config/online/sac_cond_synther_dmc2.gin \
#                                     --gin_params "redq_sac.cond_top_frac = 0.25" \
#                                     --cond_top_frac 0.25 \
#                                     --seed 0 \
#                                     --num_prior_epochs 5000 \
#                                     --num_posterior_epochs 80 \
#                                     --diffusion_steps 500 \
#                                     --num_samples 100000 \
#                                     --alpha_rtb 1.0 \
#                                     --inter_onpolicy 0.0 \
#                                     --novelty_measure curiosity \
#                                     --algorithm Ours
                                    




# # CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond_ddpm.py \
# #                                     --env reacher-hard-v0 \
# #                                     --gin_config_files config/online/sac_cond_synther_dmc2.gin \
# #                                     --gin_params "redq_sac.cond_top_frac = 0.1" \
# #                                     --cond_top_frac 0.1 \
# #                                     --seed 0 \
# #                                     --num_prior_epochs 5000 \
# #                                     --num_posterior_epochs 50 \
# #                                     --diffusion_steps 128 \
# #                                     --num_samples 100000 \
# #                                     --alpha_rtb 1.0 \
# #                                     --inter_onpolicy 0.1 \
# #                                     --novelty_measure eco \
# #                                     --algorithm PGR 



# for env in reacher-hard-v0; do
# # for env in reacher-hard-v0 quadruped-walk-v0; do
# # for env in reacher-hard-v0; do
#     for algorithm in Ours; do
#         case $algorithm in 
#             SAC) GIN_FILE=config/online/sac.gin ;;
#             REDQ) GIN_FILE=config/online/redq.gin ;;
#             SER | PGRrnd | Ours | PGR) GIN_FILE=config/online/sac_cond_synther_dmc.gin ;;
#             *) echo "Unknown algorithm: $algorithm"; exit 1 ;;
#         esac

#         case $env in
#             quadruped-walk-v0) COND_TOP_FRAC=0.1 ;;
#             cheetah-run-v0 | reacher-hard-v0 | Hopper-v2 | Walker2d-v2 | HalfCheetah-v2) COND_TOP_FRAC=0.25 ;;
#         esac

#             for seed in 0; do
#                     # rnd: 0.1, curiosity: 
#                     for alpha_rtb in 1.0; do
#                         for inter_onpolicy in 0.0; do
#                         # for inter_onpolicy in 0.0; do
#                             # for novelty_measure in rnd eco; do
#                             for novelty_measure in rnd; do
#                                 for diffusion_steps in 500; do
#                                 # 현재 순서에 맞는 GPU ID 할당 (0, 1, 2, 3, 4, 5 순환)
#                                     GPU_ID=${GPUS[$((COUNT % NUM_GPUS))]}
                                    
#                                     echo "Running $algorithm on $env (GPU $GPU_ID): seed=$seed"

#                                     CUDA_VISIBLE_DEVICES=$GPU_ID python synther/online/online_cond_ddpm.py \
#                                         --env $env \
#                                         --gin_config_files $GIN_FILE \
#                                         --gin_params "redq_sac.cond_top_frac = $COND_TOP_FRAC" \
#                                         --cond_top_frac $COND_TOP_FRAC \
#                                         --seed $seed \
#                                         --num_prior_epochs 100000 \
#                                         --num_posterior_epochs 50 \
#                                         --diffusion_steps $diffusion_steps \
#                                         --num_samples 1000000 \
#                                         --train_batch_size 1024 \
#                                         --ft_batch_size 512 \
#                                         --alpha_rtb $alpha_rtb \
#                                         --inter_onpolicy $inter_onpolicy \
#                                         --novelty_measure $novelty_measure \
#                                         --wandb \
#                                         --algorithm $algorithm &

                                    
                                    
#                                     sleep 2
#                                     ((COUNT++))

#                                 # 작업이 다 차면 모두 끝날 때까지 대기
#                                 # if [ $((COUNT % NUM_GPUS)) -eq 0 ]; then
#                                 #     echo "All GPUs are busy. Waiting for this batch to finish..."
#                                 #     wait
#                                 # fi
#                                 done
#                             done
#                         done
#                     done
#             done
#     done
# done