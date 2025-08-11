
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

for GPU_ID in 7; do
    for ft_epochs in 20 50; do
        for ft_kl_weight in 0.1 1.0; do
            CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight &
            sleep 10
            CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight &
            sleep 10
            CUDA_VISIBLE_DEVICES=$GPU_ID python ./synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --enable_finetuning --enable_curio --ft_epochs $ft_epochs --ft_kl_weight $ft_kl_weight &
            sleep 10
            wait
        done
    done
done