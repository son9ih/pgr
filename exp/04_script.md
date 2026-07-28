# 04. 실행 방법 (짧아진 커맨드)

`*_final` 실험은 커맨드 하나가 25개 플래그였다. 태스크별 값을
[synther/online/env_defaults.py](../synther/online/env_defaults.py)로 옮겼기 때문에
이제 **어떤 태스크 / 어떤 알고리즘 / 어떤 seed**만 말하면 된다.

```bash
# 전 (exp/01_experiments.md에 남아 있는 원본)
python synther/online/online_cond_ddpm_ori.py --env Hopper-v2 \
  --gin_config_files config/online/sac_cond_synther_openai.gin \
  --gin_params "redq_sac.cond_top_frac = 0.25" --seed 0 --wandb \
  --novelty_measure curiosity --diffusion_steps 1000 --num_prior_epochs 100000 \
  --num_posterior_epochs 100 --training_posterior both --train_batch_size 256 \
  --num_samples 1000000 --sample_batch_size 100000 --prior_lr_scheduler cosine \
  --rtb_lr_scheduler cosine --prior_lr 3e-4 --finetune_lr 1e-4 --ft_clip_grad 1.0 \
  --alpha_rtb 2.0 --cond_top_frac 0.25 --accumulation_steps 4 --ft_batch_size 1024 \
  --inter_onpolicy 0.0 --ddim --eta 1.0 --clip_reward 0.95

# 후 (완전히 동일한 설정으로 돌아간다)
python synther/online/online_cond_ddpm_ori.py --env hopper --seed 0 --wandb
```

---

## 1. 준비

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
conda activate pgr          # Python 3.8
```

`scripts/run.sh`는 이 `LD_LIBRARY_PATH`를 스스로 export하므로, 런처를 쓸 때는
따로 안 해도 된다. 직접 `python ...`을 부를 때만 필요하다.

## 2. 런처 — 대부분 이걸 쓰면 된다

```bash
bash scripts/run.sh <algo> <env|all> [seed ...] [-- <추가 플래그>]
```

| 인자 | 값 |
|---|---|
| `algo` | `ours` `ours-rnd` `pgr` `pgr-rnd` `ser` `sac` `redq` |
| `env` | `quad` `cheetah` `reacher` `fingereasy` `fingerhard` `hopper` `walker` `half`, 또는 `all` |
| `seed` | 생략하면 `0 1 2 3 4` |

환경변수: `GPUS="0 1"` (기본 `0`), `LOGDIR` (기본 `exp_logs`), `DRY=1` (커맨드만 출력)

```bash
# Hopper에서 Ours(curiosity) 5 seed, GPU 0
bash scripts/run.sh ours hopper

# HalfCheetah에서 PGR+rnd, seed 0/1/2만
bash scripts/run.sh pgr-rnd half 0 1 2

# 8개 태스크 전부에서 SER seed 0, GPU 4장에 나눠서
GPUS="0 1 2 3" bash scripts/run.sh ser all 0

# 뭐가 돌아갈지만 확인
DRY=1 bash scripts/run.sh ours-rnd quad

# 기본값을 하나만 덮어쓰고 싶을 때 (-- 뒤는 그대로 python에 전달)
bash scripts/run.sh ours walker 0 -- --alpha_rtb 4.0
```

동작 방식:
- GPU당 **동시에 1개**만 띄운다. `GPUS`에 준 개수만큼 채우고 `wait` → 다음 배치.
  (한 run이 4~16시간이라 GPU를 겹치면 OOM 난다)
- 로그는 `$LOGDIR/<algo>_<env>_s<seed>.log`. 진행은 `tail -f`로 본다.
- `--wandb`는 런처가 항상 붙인다. wandb 없이 돌리려면 런처 대신 직접 `python`을 부른다.

## 3. 직접 부르는 형태

런처가 만드는 커맨드는 이게 전부다.

```bash
# Ours (RTB posterior fine-tuning)
python synther/online/online_cond_ddpm_ori.py --env <env> \
       --novelty_measure {curiosity|rnd} --seed <s> --wandb

# PGR
python synther/online/online_cond_origin_baseline.py --env <env> \
       --novelty_measure {curiosity|rnd} --seed <s> --wandb

# SER
python synther/online/online_cond_origin_baseline.py --env <env> --synther --seed <s> --wandb

# SAC / REDQ
python synther/online/online_cond_origin_baseline.py --env <env> --sac  --seed <s> --wandb
python synther/online/online_cond_origin_baseline.py --env <env> --redq --seed <s> --wandb
```

`--env`는 짧은 별칭(`half`)과 정식 id(`HalfCheetah-v2`) 둘 다 받는다.

## 4. `--env`가 결정하는 값

시작할 때 아래 한 줄이 찍히므로 무엇으로 해석됐는지 항상 확인할 수 있다.

```
[env_defaults] env=Hopper-v2  gin_config_files=['config/online/sac_cond_synther_openai.gin']
  epochs=100  cond_top_frac=0.25  accumulation_steps=4  num_posterior_epochs=100
  ft_clip_grad=1.0  alpha_rtb=2.0
```

| env | gin | epochs | `cond_top_frac` | `acc_steps` | `post_epochs` | `ft_clip_grad` | `alpha_rtb` (curiosity / rnd) |
|---|---|---|---|---|---|---|---|
| `quad` | dmc | 100 | 0.1 | 8 | 150 | 0.0 | 1.0 / 10.0 |
| `cheetah` | dmc | 100 | 0.25 | 4 | 100 | 1.0 | 10.0 / 4.0 |
| `reacher` | dmc | 100 | 0.25 | 4 | 100 | 1.0 | 10.0 / 4.0 |
| `fingereasy` | dmc | 300 | 0.25 | 4 | 100 | 0.0 | 2.0 / 4.0 |
| `fingerhard` | dmc | 300 | 0.25 | 4 | 100 | 0.0 | 10.0 / 10.0 |
| `hopper` | **openai** | 100 | 0.25 | 4 | 100 | 1.0 | 2.0 / 1.0 |
| `walker` | **openai** | 100 | 0.25 | 4 | 100 | 1.0 | 0.5 / 2.0 |
| `half` | **openai** ⭐ | 100 | 0.25 | 4 | 100 | 1.0 | 2.0 / 2.0 |

⭐ `*_final` 실험에서 HalfCheetah만 `dmc.gin`으로 돌아갔는데(→ reward normalization이 꺼짐),
이제 Hopper/Walker2d와 같은 `openai.gin`을 쓴다. **따라서 HalfCheetah 결과는
[01_experiments.md](01_experiments.md)의 값과 직접 비교할 수 없고 재실행이 필요하다.**

표에 없는 env(`humanoid-*` 등)는 DMC 기본값으로 떨어지고 경고가 찍힌다.

## 5. 기본값을 바꾸고 싶을 때

**한 번만 바꿀 때** — CLI가 항상 이긴다.

```bash
python synther/online/online_cond_ddpm_ori.py --env walker --alpha_rtb 4.0 --epochs 50 --seed 0
bash scripts/run.sh ours walker 0 -- --alpha_rtb 4.0 --epochs 50
```

**계속 바꿀 때** — `synther/online/env_defaults.py`의 `ENV_CONFIG`를 고친다.
값을 바꾸면 그 뒤 모든 run이 바뀌므로 [02_code_notes.md](02_code_notes.md) §6 수정 이력에
커밋 해시를 남길 것.

## 6. `.py` 기본값 (전 태스크 공통)

`env_defaults.py`가 관여하지 않는, argparse에 그대로 박힌 값들. 전부 `*_final` 실행값과 같다.

| 인자 | 기본값 | 인자 | 기본값 |
|---|---|---|---|
| `--diffusion_steps` | 1000 | `--prior_lr` | 3e-4 |
| `--num_prior_epochs` | 100000 | `--finetune_lr` | 1e-4 |
| `--training_posterior` | `both` | `--prior_lr_scheduler` | `cosine` |
| `--train_batch_size` | 256 | `--rtb_lr_scheduler` | `cosine` |
| `--ft_batch_size` | 1024 | `--num_samples` | 1000000 |
| `--inter_onpolicy` | 0.0 | `--sample_batch_size` | 100000 |
| `--eta` | 1.0 | `--clip_reward` | 0.95 |
| `--ddim` | **True** (`--no_ddim`으로 끔) | `--anneal` | False |

이전 기본값과 달라진 것: `ft_batch_size` 256→1024, `inter_onpolicy` 0.1→0.0,
`accumulation_steps` 2→env별, `num_posterior_epochs` 50→env별, `ddim` False→True.
`--gin_params`는 이제 필요 없다 (`redq_sac.cond_top_frac` 바인딩은 로깅에만 쓰였고,
동작은 항상 `args.cond_top_frac`을 읽었다). 여전히 다른 gin 값을 덮어쓰는 데는 쓸 수 있다.

## 7. Ablation

[online_cond_ddpm_ori_abl.py](../synther/online/online_cond_ddpm_ori_abl.py)는 아직
`env_defaults.py`를 쓰지 않는다 (진행 중인 ablation을 건드리지 않기 위해 그대로 뒀다).
`epochs=62`가 하드코딩되어 있고 모든 플래그를 직접 넘겨야 한다.
정리는 [03_backlog.md](03_backlog.md) 참고.

## 8. 확인한 것

- `python -m py_compile synther/online/*.py` 통과
- `env_defaults`가 8개 태스크 × {curiosity, rnd} 전부를 `*_final` 값으로 해석 (HalfCheetah gin만 의도적으로 변경)
- CLI 명시값이 표를 덮어쓰는지, 미등록 env가 fallback으로 가는지, `eco`가 curiosity 값으로 떨어지는지 확인
- `--epochs 1` 스모크런: baseline(SER)과 Ours 모두 exit 0.
  Ours는 `--gin_params "redq_sac.retrain_diffusion_every = 500"`으로 prior 학습 →
  RTB fine-tuning → 샘플링 → DynMSE 경로까지 실제로 통과시켜 확인했다.
