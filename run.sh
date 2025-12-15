#!/bin/bash
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



# for algorithm in SER REDQ; do
#     CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 3 --algorithm $algorithm &
#     wait    
# done

# for algorithm in PGRrnd PGR; do
#     CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --algorithm $algorithm &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 3 --algorithm $algorithm &
#     wait    
# done



# ===============================================================================
# RTB
# ==============================================================================

# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --algorithm Ours &
# sleep 5
# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm Ours &
# sleep 5
# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm Ours &


# # ===============================================================================
# # SAC
# # ==============================================================================

# # Different .gin

# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --algorithm SAC



# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 3 --algorithm SAC



# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 3 --algorithm SAC


# ==============================================================================



# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm SAC &
# sleep 5
# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm SAC &



for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0; do
    for algorithm in SAC REDQ SER PGRrnd; do
        
        case $algorithm in
            SAC)
                GIN_FILE=config/online/sac.gin
                ;;
            REDQ)
                GIN_FILE=config/online/redq.gin
                ;;
            SER)
                GIN_FILE=config/online/sac_cond_synther_dmc.gin
                ;;
            PGRrnd)
                GIN_FILE=config/online/sac_cond_synther_dmc.gin
                ;;
            *)
                echo "Unknown algorithm: $algorithm"
                exit 1
                ;;
        esac

        for seed in 0 1 2 3 4; do
            echo "Running $algorithm on $env with seed $seed using $GIN_FILE"
            
            CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py \
            --env $env \
            --gin_config_files $GIN_FILE \
            --gin_params 'redq_sac.cond_top_frac = 0.25' \
            --seed $seed \
            --wandb \
            --algorithm $algorithm &
            
            sleep 2
        done
        wait
    done
done