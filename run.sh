export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/son9ih/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# ===============================================================================
# Baseline
# ==============================================================================


# for algorithm in SER REDQ; do
#     CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --algorithm $algorithm &
#     wait    
# done

# for algorithm in PGR PGRrnd; do
#     CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --algorithm $algorithm &
#     wait    
# done



# for algorithm in SER REDQ; do
#     CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 3 --algorithm $algorithm &
#     wait    
# done

# for algorithm in PGRrnd PGR; do
#     CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 3 --algorithm $algorithm &
#     wait    
# done



for algorithm in SER REDQ; do
    CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm $algorithm &
    sleep 5
    CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --algorithm $algorithm &
    sleep 5
    CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --algorithm $algorithm &
    sleep 5
    CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 3 --algorithm $algorithm &
    wait    
done

for algorithm in PGRrnd PGR; do
    CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm $algorithm &
    sleep 5
    CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --algorithm $algorithm &
    sleep 5
    CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --algorithm $algorithm &
    sleep 5
    CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 3 --algorithm $algorithm &
    wait    
done



# ===============================================================================
# RTB
# ==============================================================================

# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc2.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --algorithm Ours
