
#!/bin/bash

# for GPU_ID in 6; do
#     for ft_epochs in 20 50 100; do
#         for ft_kl_weight in 0.1 1.0 0.01; do
#             CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight &
#             sleep 10
#             CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight &
#             sleep 10
#             CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight &
#             sleep 10
#             wait
#         done
#     done
# done

# for GPU_ID in 6; do
#     for ft_epochs in 20 50; do
#         for ft_kl_weight in 0.01 0.05; do
#             CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight &
#             sleep 10
#             CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight &
#             sleep 10
#             CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight &
#             sleep 10
#             wait
#         done
#     done
# done

# for GPU_ID in 6; do
#     for ft_epochs in 20 40; do
#         for ft_kl_weight in 1.0 10.0 100.0; do
#             for ft_lr in 1e-4; do
#                 CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr $ft_lr &
#                 sleep 10
#                 CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 4 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr $ft_lr &
#                 sleep 10
#                 CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 5 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr $ft_lr &
#                 sleep 10
#                 wait
#             done
#         done
#     done
# done

# This for squared loss
# for GPU_ID in 6; do
#     for ft_epochs in 30; do
#         for ft_kl_weight in 100.0; do
#             # for ft_lr in 1e-5; do
#                 CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr 1e-3 &
#                 sleep 10
#                 CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr 1e-4 &
#                 sleep 10
#                 CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr 1e-5 &
#                 sleep 10
#                 # CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 4 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr $ft_lr &
#                 # sleep 10
#                 # CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 5 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr $ft_lr &
#                 # sleep 10
#                 wait
#             # done
#         done
#     done
# done


# This for 0.0*
for GPU_ID in 7; do
    for ft_epochs in 30; do
        for ft_kl_weight in 1.0; do
            # for ft_lr in 1e-4; do
                CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr 1e-3 &
                sleep 10
                CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr 1e-4 &
                sleep 10
                CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr 1e-5 &
                sleep 10
                # CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 4 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr $ft_lr &
                # sleep 10
                # CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 5 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight --ft_lr $ft_lr &
                # sleep 10
                wait
            # done
        done
    done
done