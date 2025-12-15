export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/son9ih/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# ================================================================================
# SAC
# ================================================================================

# for env in Hopper-v2 Walker2d-v2 HalfCheetah-v2; do
#     CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm SAC &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --algorithm SAC &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --algorithm SAC &
#     sleep 5
#     CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc_sac.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --algorithm SAC &
# done


# ================================================================================
# REDQ, SER, PGR, PGRrnd
# ================================================================================


# for env in Hopper-v2 Walker2d-v2 HalfCheetah-v2; do
#     for algorithm in REDQ ; do
#         CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 1 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 2 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 3 --wandb --algorithm $algorithm &
#         wait
#     done
# done    


# for env in Hopper-v2 Walker2d-v2 HalfCheetah-v2; do
#     for algorithm in SER ; do
#         CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 1 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 2 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 3 --wandb --algorithm $algorithm &
#         wait
#     done
# done    

# for env in Hopper-v2 Walker2d-v2 HalfCheetah-v2; do
#     for algorithm in PGRrnd ; do
#         CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 1 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 2 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 3 --wandb --algorithm $algorithm &
#         wait
#     done
# done    


        # CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env Hopper-v2 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 3 --wandb --algorithm PGRrnd

# for env in Hopper-v2 Walker2d-v2 HalfCheetah-v2; do
#     for algorithm in PGR ; do
#         CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 0 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 1 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 2 --wandb --algorithm $algorithm &
#         sleep 5
#         CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env $env --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --seed 3 --wandb --algorithm $algorithm &
#         wait
#     done
# done    





for env in Hopper-v2 Walker2d-v2 HalfCheetah-v2; do
    for algorithm in SAC REDQ SER PGRrnd; do
        
        case $algorithm in
            SAC)
                GIN_FILE=config/online/sac.gin
                ;;
            REDQ)
                GIN_FILE=config/online/redq.gin
                ;;
            SER)
                GIN_FILE=config/online/sac_cond_synther_openai.gin
                ;;
            PGRrnd)
                GIN_FILE=config/online/sac_cond_synther_openai.gin
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