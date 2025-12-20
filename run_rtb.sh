export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/son9ih/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# beta 1.0 이면 over-optimization, 0.01 이면 non-optimization
# RTB running code for dmc
# key parameters: beta, backprop_epochs, gfn_batch_size, ft_batch_size
CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --wandb --algorithm Ours --backprop_epochs 10 --beta 0.5 --gfn_batch_size 4 --ft_batch_size 128 &
sleep 5
CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --seed 0 --wandb --algorithm Ours --backprop_epochs 10 --beta 0.5 --gfn_batch_size 4 --ft_batch_size 128 &
sleep 5
CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --wandb --algorithm Ours --backprop_epochs 10 --beta 0.5 --gfn_batch_size 4 --ft_batch_size 128 &
# sleep 5



