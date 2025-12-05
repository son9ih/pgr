export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/son9ih/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 5.0
CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 10.0
CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 100.0
CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 1000.0
CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --algorithm Ours --beta 10000.0
