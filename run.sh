export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/son9ih/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 

# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --seed 0 &
# CUDA_VISIBLE_DEVICES=1 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --seed 1 &
# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --seed 2 &

# for seed in 1 2 3; do 
#     CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed $seed --knn_avg --state_ent --synther &
#     CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed $seed --knn_avg --state_ent &
#     wait
# done
# for seed in 1 2 4; do 
#     CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --synther --state_ent --knn_k 12 --ent_eval_num 10000 --rnd
#     CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd
#     wait
# done


# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb

# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --synther --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=1 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --synther --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=1 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 4 --synther --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 4 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &

# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=1 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 4 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &

# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --synther --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --synther --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 4 --synther --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 4 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &

# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --state_ent --knn_k 9 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 4 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd &

# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 1 --state_ent --knn_k 9 --ent_eval_num 1000 --rnd --finetune --synther 
# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --kl_weight 0.1 &
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --kl_weight 0.1 &
# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --kl_weight 0.01 &
# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --kl_weight 0.01 &
# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --kl_weight 0.001 &
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --kl_weight 0.001 &

# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 1000 --rnd --finetune --synther --reward_coef 1e5 --kl_weight 0.001 --histo &
# sleep 5
# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 1000 --rnd --finetune --synther --reward_coef 1e5 --kl_weight 0.01 --histo &
# sleep 5
# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 1000 --rnd --finetune --synther --reward_coef 1e5 --kl_weight 0.1 --histo &
# sleep 5
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 2 --state_ent --knn_k 12 --ent_eval_num 1000 --rnd --histo &





# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 5 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --reward_coef 1e5 --kl_weight 0.1 --histo &
# sleep 5
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 5 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --histo &
# sleep 5
# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 6 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --reward_coef 1e5 --kl_weight 0.1 --histo &
# sleep 5
# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 6 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --histo &
# sleep 5
# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 7 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --reward_coef 1e5 --kl_weight 0.1 --histo &
# sleep 5
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 7 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --histo &



# CUDA_VISIBLE_DEVICES=1 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 5 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --reward_coef 1e8 --kl_weight 0.01 --histo --finetune_lr 0.5e-4 &
# sleep 5
# CUDA_VISIBLE_DEVICES=2 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 5 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --reward_coef 1e7 --kl_weight 0.01 --histo --finetune_lr 0.5e-4 &
# sleep 5
# CUDA_VISIBLE_DEVICES=3 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 5 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --reward_coef 1e6 --kl_weight 0.01 --histo --finetune_lr 0.5e-4 &
# sleep 5
# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 5 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --reward_coef 1e8 --kl_weight 0.1 --histo --finetune_lr 0.5e-4 &
# sleep 5
# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 5 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --reward_coef 1e7 --kl_weight 0.1 --histo --finetune_lr 0.5e-4 &
# sleep 5
# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 5 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --reward_coef 1e6 --kl_weight 0.1 --histo --finetune_lr 0.5e-4 &
# sleep 5
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 5 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --histo &

# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 6 --state_ent --knn_k 12 --ent_eval_num 2000 --disable_diffusion &
# sleep 5
# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 6 --state_ent --knn_k 12 --ent_eval_num 2000 --synther --histo --tsne &
# sleep 5
# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 6 --state_ent --knn_k 12 --ent_eval_num 10000 --histo --tsne &
# sleep 5
# CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.5' --wandb --seed 6 --state_ent --knn_k 12 --ent_eval_num 2000 --rnd --histo --tsne &
# sleep 5
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 6 --state_ent --knn_k 12 --ent_eval_num 10000 --rnd --finetune --synther --reward_coef 1e5 --kl_weight 0.005 --histo --tsne &




# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond_rtb.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.5' --wandb --seed 6 --state_ent --knn_k 12 --ent_eval_num 2000 --synther --rtb --histo

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

for algorithm in PGR PGRrnd; do
    CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 0 --algorithm $algorithm &
    sleep 5
    CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 1 --algorithm $algorithm &
    sleep 5
    CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 2 --algorithm $algorithm &
    sleep 5
    CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1' --wandb --seed 3 --algorithm $algorithm &
    wait    
done


# CUDA_VISIBLE_DEVICES=4 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm REDQ &
# sleep 5
# CUDA_VISIBLE_DEVICES=5 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm SER &
# sleep 5
# CUDA_VISIBLE_DEVICES=6 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm PGR &
# sleep 5
# CUDA_VISIBLE_DEVICES=7 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --wandb --seed 0 --algorithm PGRrnd 
