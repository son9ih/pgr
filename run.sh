# SAC baseline
for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0 HalfCheetah-v2 Walker2d-v2 Hopper-v2 finger-turn_hard-v0; do
                            
                            GIN_FILE="config/online/sac.gin"
                            
                            CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py \
                                --env $env \
                                --gin_config_files $GIN_FILE \
                                --seed 0 \
                                --wandb \
                                --sac &

done

# REDQ baseline
for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0 HalfCheetah-v2 Walker2d-v2 Hopper-v2 finger-turn_hard-v0; do

                            GIN_FILE="config/online/redq.gin"
                            
                            CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py \
                                --env $env \
                                --gin_config_files $GIN_FILE \
                                --seed 0 \
                                --wandb \
                                --redq &

done

# SER baseline
for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0 HalfCheetah-v2 Walker2d-v2 Hopper-v2 finger-turn_hard-v0; do

                            case "$env" in
                                "quadruped-walk-v0" | "cheetah-run-v0" | "reacher-hard-v0" | "finger-turn_hard-v0")
                                    GIN_FILE="config/online/sac_cond_synther_dmc.gin"
                                    ;;
                                *)
                                    GIN_FILE="config/online/sac_cond_synther_openai.gin"
                                    ;;
                            esac

    
                            CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py \
                                --env $env \
                                --gin_config_files $GIN_FILE \
                                --seed 0 \
                                --synther \
                                --ddim &

done

# PGR baseline
for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0 HalfCheetah-v2 Walker2d-v2 Hopper-v2 finger-turn_hard-v0; do
    for novelty_measure in curiosity rnd; do
        for cond_top_frac in 0.25; do
            for cfg_scale in 2.0; do

                            case "$env" in
                                "quadruped-walk-v0" | "cheetah-run-v0" | "reacher-hard-v0" | "finger-turn_hard-v0")
                                    GIN_FILE="config/online/sac_cond_synther_dmc.gin"
                                    ;;
                                *)
                                    GIN_FILE="config/online/sac_cond_synther_openai.gin"
                                    ;;
                            esac

                            CUDA_VISIBLE_DEVICES=0 python synther/online/online_cond.py \
                                --env $env \
                                --gin_config_files $GIN_FILE \
                                --gin_params "redq_sac.cond_top_frac = $cond_top_frac" "redq_sac.cfg_scale = $cfg_scale" \
                                --seed 0 \
                                --novelty_measure $novelty_measure \
                                --ddim &

            done
        done
    done
done

# REGR (Ours)
for env in quadruped-walk-v0 cheetah-run-v0 reacher-hard-v0 HalfCheetah-v2 Walker2d-v2 Hopper-v2 finger-turn_hard-v0; do
    for novelty_measure in curiosity rnd; do
        for alpha_rtb in 1.0; do

                            case "$env" in
                                "quadruped-walk-v0")
                                    POST_EPOCH=150
                                    GIN_FILE="config/online/sac_cond_synther_dmc.gin"
                                    ;;
                                "HalfCheetah-v2" | "Hopper-v2" | "Walker2d-v2")
                                    POST_EPOCH=100
                                    GIN_FILE="config/online/sac_cond_synther_openai.gin"
                                    ;;
                                "reacher-hard-v0" | "cheetah-run-v0")
                                    POST_EPOCH=100
                                    GIN_FILE="config/online/sac_cond_synther_dmc.gin"
                                    ;;
                                *)
                                    POST_EPOCH=100
                                    GIN_FILE="config/online/sac_cond_synther_dmc.gin"
                                    ;;
                            esac

                            CUDA_VISIBLE_DEVICES=1 python synther/online/online_cond_regr.py \
                                --env $env \
                                --gin_config_files $GIN_FILE \
                                --seed 0 \
                                --alpha_rtb $alpha_rtb \
                                --novelty_measure $novelty_measure \
                                --num_posterior_epochs $POST_EPOCH \
                                --accumulation_steps 1 \
                                --ddim &

        done
    done
done