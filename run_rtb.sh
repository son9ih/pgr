export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/son9ih/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# CUDA_VISIBLE_DEVICES=1 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --wandb --algorithm Ours --finetune_lr 1e-4
# sleep 5
# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --wandb --algorithm Ours --finetune_lr 1e-5
# sleep 5
CUDA_VISIBLE_DEVICES=1 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --seed 0 --wandb --algorithm Ours --finetune_lr 3e-4

# sleep 5
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --wandb --algorithm Ours --beta 100.0
# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 100.0
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 1000.0
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 10000.0
