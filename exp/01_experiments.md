# 01. 실험 기록 — wandb `gda-for-orl` / `*_final` 8개 프로젝트

스냅샷 기준일: **2026-07-28** · 총 **205 runs** (모두 `finished`)
원본 데이터: [data/wandb_final_runs.json](data/wandb_final_runs.json)

---

## 0. 큰 그림

- entity: `gda-for-orl`, 프로젝트 8개 = 태스크 8개 × (baseline 3 + ours 2) × seed 5.
- **주의**: 코드에는 `wandb.init(project=env_name)` (online_cond_ddpm_ori.py:253) 으로 박혀 있다.
  즉 `*_final` 프로젝트는 **wandb UI에서 수동으로 모아둔 큐레이션 결과**이고,
  지금 그대로 재실행하면 `Walker2d-v2` 같은 env 이름 프로젝트로 들어간다.
  → `--wandb_project` 인자화 필요 ([02_code_notes.md](02_code_notes.md) I-9).

### 알고리즘 ↔ 스크립트

| 라벨 | 스크립트 | 구분 플래그 |
|---|---|---|
| `SER` | `synther/online/online_cond_origin_baseline.py` | `--synther` |
| `PGR+curiosity` | `synther/online/online_cond_origin_baseline.py` | `--novelty_measure curiosity` (기본값) |
| `PGR+rnd` | `synther/online/online_cond_origin_baseline.py` | `--novelty_measure rnd` |
| `Ours+curiosity` | `synther/online/online_cond_ddpm_ori.py` | `--novelty_measure curiosity` |
| `Ours+rnd` | `synther/online/online_cond_ddpm_ori.py` | `--novelty_measure rnd` |

`SER`은 `novelty_measure`를 쓰지 않는다 (아래 §4 seed 사고의 원인).

### 사용된 commit

| commit | 날짜 | 메시지 | 어디에 |
|---|---|---|---|
| `8a1a0d9` | 2026-02-11 | update: visual environments | Ours — mujoco 3종 + reacher/cheetah/quad(clip=1) |
| `ebe8bb9` | 2026-02-15 | fix: ddpm ver. SER, PGR | baseline — mujoco 3종 + reacher/cheetah |
| `d57a476` | 2026-02-16 | update: finger-turn_easy-v0 | finger 2종 전부 + quad(clip=0) + PGR+rnd 재실행 |

---

## 1. 태스크별 하이퍼파라미터 요약

| 프로젝트 | env | gin | `cond_top_frac` | `accumulation_steps` | `num_posterior_epochs` | 총 epoch | Ours+curiosity | Ours+rnd |
|---|---|---|---|---|---|---|---|---|
| `quad_final` | `quadruped-walk-v0` | dmc | 0.1 | 8 | 150 | 100 | α=1, clip=**0**/1 둘 다 | α=10, clip=0 |
| `cheetah_final` | `cheetah-run-v0` | dmc | 0.25 | 4 | 100 | 100 | α=10, clip=1 | α=4, clip=1 |
| `reacher_final` | `reacher-hard-v0` | dmc | 0.25 | 4 | 100 | 100 | α=10, clip=1 | α=4, clip=1 |
| `fingereasy_final` | `finger-turn_easy-v0` | dmc | 0.25 | 4 | 100 | **300** | α=2, clip=0 | α=4, clip=0 |
| `fingerhard_final` | `finger-turn_hard-v0` | dmc | 0.25 | 4 | 100 | **300** | α=10, clip=0 | α=10, clip=0 |
| `hopper_final` | `Hopper-v2` | **openai** | 0.25 | 4 | 100 | 100 | α=2, clip=1 | α=1, clip=1 |
| `walker_final` | `Walker2d-v2` | **openai** | 0.25 | 4 | 100 | 100 | α=0.5, clip=1 | α=2, clip=1 |
| `half_final` | `HalfCheetah-v2` | **dmc** ⚠️ | 0.25 | 4 | 100 | 100 | α=2, clip=1 | α=2, clip=1 |

공통: `--diffusion_steps 1000 --num_prior_epochs 100000 --training_posterior both
--train_batch_size 256 --num_samples 1000000 --sample_batch_size 100000
--prior_lr 3e-4 --finetune_lr 1e-4 --ft_batch_size 1024 --inter_onpolicy 0.0
--ddim --eta 1.0 --clip_reward 0.95`, 두 lr scheduler 모두 `cosine`.
gin 공통: `utd_ratio=20, num_Q=2, num_min=2, cfg_scale=2.0, cfg_dropout=0.25,
diffusion_sample_ratio=0.5, retrain_diffusion_every=10000, start_steps=5000`.

⚠️ `half_final`만 MuJoCo인데 `sac_cond_synther_dmc.gin`을 썼다 →
`skip_reward_norm=True`, `model_terminals=False`. Hopper/Walker2d(openai.gin)는
`skip_reward_norm=False`, `model_terminals=True`. **MuJoCo 3종 표가 서로 다른 설정**이다.
(HalfCheetah는 종료가 없어 `model_terminals`는 무해하지만 reward normalization은 다르다.)

- 총 epoch은 CLI가 아니라 `redq_sac()` 본문에 하드코딩되어 있다
  ([online_cond_ddpm_ori.py:240-243](../synther/online/online_cond_ddpm_ori.py#L240-L243)):
  `finger-turn_*`/`humanoid-*` → 300, 그 외 → 100.

---

## 2. 실행 커맨드 (seed는 0–4)

> ⚠️ **아래는 당시 실제로 쓴 원본 커맨드 기록이다.** 지금은 이 값들이
> [env_defaults.py](../synther/online/env_defaults.py)에 들어가 있어서
> `bash scripts/run.sh ours hopper` 한 줄로 같은 설정이 나온다 → [04_script.md](04_script.md).
> 단 **HalfCheetah는 gin이 `dmc.gin` → `openai.gin`으로 교정**되었으므로(I-5),
> 새로 돌린 HalfCheetah 결과는 §3의 `half_final` 값과 직접 비교할 수 없다.

`$SEED` 자리에 0,1,2,3,4를 넣어 5개씩 돌렸다. 앞에 `CUDA_VISIBLE_DEVICES=<gpu>`를 붙인다.

### 2.1 Baselines (공통 형태)

```bash
# SER
python synther/online/online_cond_origin_baseline.py --env $ENV \
  --gin_config_files $GIN --gin_params "redq_sac.cond_top_frac = $TOPFRAC" \
  --wandb --seed $SEED --synther --cond_top_frac $TOPFRAC --ddim

# PGR + curiosity
python synther/online/online_cond_origin_baseline.py --env $ENV \
  --gin_config_files $GIN --gin_params "redq_sac.cond_top_frac = $TOPFRAC" \
  --wandb --seed $SEED --novelty_measure curiosity --cond_top_frac $TOPFRAC --ddim

# PGR + rnd
python synther/online/online_cond_origin_baseline.py --env $ENV \
  --gin_config_files $GIN --gin_params "redq_sac.cond_top_frac = $TOPFRAC" \
  --wandb --seed $SEED --novelty_measure rnd --cond_top_frac $TOPFRAC --ddim
```

`($ENV, $GIN, $TOPFRAC)` 조합은 §1 표와 동일.
finger 2종·quad의 `PGR+curiosity`는 `--novelty_measure`를 아예 생략했는데 기본값이
`curiosity`라 동일하다.

### 2.2 Ours (RTB posterior fine-tuning)

```bash
python synther/online/online_cond_ddpm_ori.py --env $ENV \
  --gin_config_files $GIN --gin_params "redq_sac.cond_top_frac = $TOPFRAC" \
  --seed $SEED --wandb \
  --novelty_measure $NOV \
  --diffusion_steps 1000 --num_prior_epochs 100000 --num_posterior_epochs $POST \
  --training_posterior both --train_batch_size 256 \
  --num_samples 1000000 --sample_batch_size 100000 \
  --prior_lr_scheduler cosine --rtb_lr_scheduler cosine \
  --prior_lr 3e-4 --finetune_lr 1e-4 \
  --ft_clip_grad $CLIP --alpha_rtb $ALPHA --cond_top_frac $TOPFRAC \
  --accumulation_steps $ACC --ft_batch_size 1024 --inter_onpolicy 0.0 \
  --ddim --eta 1.0 --clip_reward 0.95
```

| 태스크 | `$NOV` | `$ALPHA` | `$CLIP` | `$ACC` | `$POST` |
|---|---|---|---|---|---|
| quadruped-walk-v0 | curiosity | 1.0 | **0.0** (best) / 1.0 | 8 | 150 |
| quadruped-walk-v0 | rnd | 10.0 | 0.0 | 8 | 150 |
| cheetah-run-v0 | curiosity / rnd | 10.0 / 4.0 | 1.0 | 4 | 100 |
| reacher-hard-v0 | curiosity / rnd | 10.0 / 4.0 | 1.0 | 4 | 100 |
| finger-turn_easy-v0 | curiosity / rnd | 2.0 / 4.0 | 0.0 | 4 | 100 |
| finger-turn_hard-v0 | curiosity / rnd | 10.0 / 10.0 | 0.0 | 4 | 100 |
| Hopper-v2 | curiosity / rnd | 2.0 / 1.0 | 1.0 | 4 | 100 |
| Walker2d-v2 | curiosity / rnd | 0.5 / 2.0 | 1.0 | 4 | 100 |
| HalfCheetah-v2 | curiosity / rnd | 2.0 / 2.0 | 1.0 | 4 | 100 |

전체 원본 커맨드는 `python exp/tools/summarize.py --cmds`로 뽑을 수 있다.

---

## 3. 결과 (최종 epoch, seed 평균 ± 표준편차)

지표: `eval/AverageTestEpRet`(Return), `eval/DynMSE_mean`, `eval/StateEnt`, `eval/AverageNormQBias`.
DynMSE는 MuJoCo에서만 계산된다 (DMC는 `set_state` 경로가 없어 `-`).

### quad_final — `quadruped-walk-v0` (100 epochs)

| method | n | seeds | Return | StateEnt | NormQBias | wall-clock |
|---|---|---|---|---|---|---|
| **Ours+curiosity (A=1, clip=0)** | 5 | 0-4 | **910.3 ± 57.7** | 2.97 ± 0.03 | 0.33 ± 0.32 | 6.2 h |
| Ours+curiosity (A=1, clip=1) | 5 | 0-4 | 736.6 ± 314.2 | 2.95 ± 0.05 | 0.85 ± 1.22 | 6.2 h |
| Ours+rnd (A=10, clip=0) | 5 | 0-4 | 688.8 ± 350.1 | 2.93 ± 0.08 | 0.82 ± 0.70 | 6.2 h |
| PGR+curiosity | 5 | 0-4 | 739.5 ± 309.4 | 3.00 ± 0.05 | 3.10 ± 1.35 | 5.0 h |
| PGR+rnd | 5 | 0-4 | 767.6 ± 238.9 | 2.93 ± 0.07 | 1.87 ± 1.20 | 5.4 h |
| SER | 5 | 0-4 | 658.3 ± 344.3 | 2.95 ± 0.05 | 1.65 ± 1.56 | 4.3 h |

### cheetah_final — `cheetah-run-v0` (100 epochs)

| method | n | seeds | Return | StateEnt | NormQBias | wall-clock |
|---|---|---|---|---|---|---|
| Ours+curiosity (A=10, clip=1) | 5 | 0-4 | 514.6 ± 88.8 | 1.83 ± 0.08 | 1.66 ± 0.23 | 5.3 h |
| **Ours+rnd (A=4, clip=1)** | 5 | 0-4 | **521.7 ± 84.6** | 1.85 ± 0.06 | 1.71 ± 0.23 | 5.3 h |
| PGR+curiosity | 5 | 0-4 | 410.6 ± 57.3 | 1.64 ± 0.11 | 1.38 ± 0.12 | 4.8 h |
| PGR+rnd | 5 | 0-4 | 497.8 ± 275.6 | 1.80 ± 0.04 | 1.54 ± 0.24 | 4.9 h |
| SER | 5 | 0-4 | 345.5 ± 116.6 | 1.81 ± 0.06 | 1.47 ± 0.19 | 4.1 h |

### reacher_final — `reacher-hard-v0` (100 epochs)

| method | n | seeds | Return | StateEnt | NormQBias | wall-clock |
|---|---|---|---|---|---|---|
| Ours+curiosity (A=10, clip=1) | 5 | 0-4 | 840.8 ± 106.1 | 0.54 ± 0.03 | 0.51 ± 0.81 | 5.2 h |
| **Ours+rnd (A=4, clip=1)** | 5 | 0-4 | **847.0 ± 161.0** | 0.52 ± 0.03 | 1.07 ± 1.83 | 5.3 h |
| PGR+curiosity | 5 | 0-4 | 575.2 ± 170.5 | 0.58 ± 0.05 | 1.93 ± 0.93 | 4.8 h |
| PGR+rnd | 5 | 0-4 | 567.8 ± 140.1 | 0.57 ± 0.05 | 1.10 ± 0.76 | 4.8 h |
| SER | 5 | 0-4 | 783.7 ± 125.3 | 0.48 ± 0.02 | 0.55 ± 0.74 | 4.1 h |

### fingereasy_final — `finger-turn_easy-v0` (300 epochs)

| method | n | seeds | Return | StateEnt | NormQBias | wall-clock |
|---|---|---|---|---|---|---|
| **Ours+curiosity (A=2, clip=0)** | 5 | 0-4 | **835.8 ± 158.7** | 0.79 ± 0.02 | 0.81 ± 1.23 | 16.1 h |
| Ours+rnd (A=4, clip=0) | 5 | 0-4 | 744.4 ± 211.0 | 0.78 ± 0.02 | 0.11 ± 0.49 | 16.3 h |
| PGR+curiosity | 5 | 0-4 | 456.6 ± 151.9 | 0.71 ± 0.03 | 0.41 ± 1.77 | 14.8 h |
| PGR+rnd | 5 | 0-4 | 530.5 ± 233.0 | 0.79 ± 0.04 | -0.35 ± 0.89 | 15.2 h |
| SER | 5 | 0-4 | 742.5 ± 52.8 | 0.79 ± 0.01 | 0.36 ± 0.82 | 12.6 h |

### fingerhard_final — `finger-turn_hard-v0` (300 epochs)

| method | n | seeds | Return | StateEnt | NormQBias | wall-clock |
|---|---|---|---|---|---|---|
| **Ours+curiosity (A=10, clip=0)** | 5 | 0-4 | **583.3 ± 145.8** | 0.81 ± 0.02 | 0.14 ± 1.14 | 16.0 h |
| Ours+rnd (A=10, clip=0) | 5 | 0-4 | 533.0 ± 63.3 | 0.79 ± 0.02 | 0.07 ± 0.97 | 16.0 h |
| PGR+curiosity | 5 | 0,1,**1**,3,4 ⚠️ | 237.5 ± 150.2 | 0.74 ± 0.02 | -0.33 ± 0.83 | 15.0 h |
| PGR+rnd | 5 | 0-4 | 140.1 ± 113.2 | 0.77 ± 0.02 | 266.6 ± 597.1 ⚠️ | 15.1 h |
| SER | 5 | 0-4 | 312.6 ± 162.4 | 0.77 ± 0.02 | -0.35 ± 1.05 | 12.6 h |

### hopper_final — `Hopper-v2` (100 epochs)

| method | n | seeds | Return | DynMSE | StateEnt | NormQBias | wall-clock |
|---|---|---|---|---|---|---|---|
| **Ours+curiosity (A=2, clip=1)** | 5 | 0-4 | **2895.0 ± 749.8** | 0.02 ± 0.00 | 0.91 ± 0.03 | 0.00 ± 0.12 | 5.1 h |
| Ours+rnd (A=1, clip=1) | 5 | 0-4 | 2256.3 ± 1056.0 | 0.02 ± 0.00 | 0.92 ± 0.02 | -0.02 ± 0.03 | 5.1 h |
| PGR+curiosity | 5 | 0-4 | 1036.0 ± 733.7 | 0.14 ± 0.05 | 0.66 ± 0.11 | 0.02 ± 0.05 | 4.7 h |
| PGR+rnd | 5 | 0-4 | 2335.7 ± 1261.4 | 0.08 ± 0.01 | 0.73 ± 0.06 | 0.00 ± 0.14 | 4.7 h |
| SER | 5 | 0,1,2,3,**3** ⚠️ | 1795.8 ± 484.5 | 0.02 ± 0.00 | 0.87 ± 0.02 | -0.03 ± 0.03 | 4.0 h |

### walker_final — `Walker2d-v2` (100 epochs)

| method | n | seeds | Return | DynMSE | StateEnt | NormQBias | wall-clock |
|---|---|---|---|---|---|---|---|
| Ours+curiosity (A=0.5, clip=1) | 5 | 0-4 | 2920.0 ± 1178.9 | 0.96 ± 0.04 | 1.88 ± 0.03 | 1.05 ± 1.62 | 5.2 h |
| Ours+rnd (A=2, clip=1) | 5 | 0-4 | 1862.3 ± 1570.9 | 0.72 ± 0.16 | 1.76 ± 0.17 | 1.56 ± 1.22 | 5.2 h |
| PGR+curiosity | 5 | 0-4 | 2439.1 ± 995.0 | 2.57 ± 0.32 | 1.62 ± 0.15 | 0.15 ± 0.12 | 4.8 h |
| **PGR+rnd** | 5 | 0-4 | **3152.5 ± 1491.2** | 2.05 ± 0.14 | 1.73 ± 0.12 | 0.07 ± 0.07 | 4.8 h |
| SER | 5 | **0**,0,2,3,4 ⚠️ | 1369.2 ± 483.7 | 1.22 ± 0.12 | 1.64 ± 0.08 | 1.20 ± 1.32 | 4.0 h |

### half_final — `HalfCheetah-v2` (100 epochs, dmc.gin ⚠️)

| method | n | seeds | Return | DynMSE | StateEnt | NormQBias | wall-clock |
|---|---|---|---|---|---|---|---|
| Ours+curiosity (A=2, clip=1) | 5 | 0-4 | 6558.9 ± 824.8 | 0.29 ± 0.06 | 1.93 ± 0.11 | 0.05 ± 0.09 | 5.2 h |
| **Ours+rnd (A=2, clip=1)** | 5 | 0-4 | **6960.5 ± 637.9** | 0.28 ± 0.08 | 1.82 ± 0.08 | 0.00 ± 0.07 | 5.2 h |
| PGR+curiosity | 5 | 0-4 | 6286.2 ± 1097.2 | 2.59 ± 0.59 | 1.75 ± 0.19 | -0.00 ± 0.06 | 4.8 h |
| PGR+rnd | 5 | 0-4 | 6707.5 ± 562.4 | 1.65 ± 0.45 | 1.86 ± 0.13 | 0.11 ± 0.12 | 5.0 h |
| SER | 5 | **0**,0,1,2,4 ⚠️ | 6199.6 ± 643.7 | 0.49 ± 0.09 | 1.90 ± 0.13 | 0.13 ± 0.15 | 4.1 h |

---

## 4. 읽어낸 것

**좋은 신호**

1. **DynMSE가 확실히 개선된다.** MuJoCo 3종 모두에서 Ours가 PGR 대비 생성 transition의
   one-step dynamics 오차를 크게 낮춘다 — Walker 2.05→0.72~0.96, HalfCheetah 1.65~2.59→0.28,
   Hopper 0.08~0.14→0.02. **이게 지금 이 방법의 가장 깨끗한 주장거리다.**
   (RTB fine-tuning이 novelty를 좇으면서도 prior의 dynamics를 안 깬다는 근거)
2. **StateEnt도 대체로 같이 올라간다** (Hopper 0.66~0.73 → 0.91). novelty 보상이
   실제로 커버리지를 늘리고 있다는 뜻.
3. Return 기준 8개 중 **7개에서 Ours가 최고** (예외: Walker2d — PGR+rnd 3152 vs Ours 2920,
   다만 ±1200~1500이라 유의하지 않음).
4. **각 태스크의 최고 baseline 대비 Ours 최고값**:
   finger-hard +86.6% (583.3 vs SER 312.6) · hopper +23.9% (2895 vs PGR+rnd 2336) ·
   quad +18.6% (910 vs PGR+rnd 768) · finger-easy +12.6% (836 vs SER 743) ·
   reacher +8.1% (847 vs SER 784) · cheetah +4.8% (522 vs PGR+rnd 498) ·
   half +3.8% (6961 vs PGR+rnd 6707) · walker **−7.4%** (2920 vs PGR+rnd 3152).
   → **마진이 큰 쪽(finger 2종, hopper, quad)과 오차범위 안인 쪽(cheetah/half/reacher)이 갈린다.**
   PGR+curiosity만 baseline으로 잡으면 마진이 훨씬 커 보이지만, `PGR+rnd`가 여러 태스크에서
   더 센 baseline이므로 **표에는 두 PGR을 모두 남겨야 한다.**

**주의해야 할 것**

5. **`alpha_rtb`가 태스크마다 0.5~10으로 제각각**이고 baseline엔 대응 하이퍼파라미터가 없다.
   지금 표는 "Ours만 태스크별 튜닝됨" 비판을 그대로 받는다. → sensitivity 곡선 필요.
6. **`ft_clip_grad`도 통일 안 됨** (finger/quad는 0.0, 나머지는 1.0). quad에서 둘 다 돌아간
   결과로는 clip=0이 clip=1보다 +174 (910 vs 737) — 전 태스크에 clip=0을 다시 볼 값어치가 있다.
7. **quad만 `accumulation_steps=8`, `num_posterior_epochs=150`** — Ours 쪽에만 추가 연산을 준 셈.
8. **`half_final`의 gin 불일치** (§1 ⚠️) — MuJoCo 표의 reward normalization이 서로 다르다.
9. Walker2d의 분산이 ±1200~1570으로 지나치게 크다. seed 5개로는 결론이 안 난다.
10. `fingerhard` `PGR+rnd`의 `NormQBias`가 266 ± 597 — Q가 발산한 seed가 섞여 있다.
    로그를 열어 해당 seed를 확인해야 한다.

**seed 사고 (재실행 필요)** ⚠️

`SER`은 `--novelty_measure`를 무시하는데, 런처가 `curiosity`/`rnd` 두 루프를 돌면서
SER을 두 번 제출했다. 그 결과 **같은 seed가 중복되고 다른 seed가 빠졌다**:

| 프로젝트 | method | 실제 seeds | 빠진 seed |
|---|---|---|---|
| `walker_final` | SER | 0, 0, 2, 3, 4 | **1** |
| `hopper_final` | SER | 0, 1, 2, 3, 3 | **4** |
| `half_final` | SER | 0, 0, 1, 2, 4 | **3** |
| `fingerhard_final` | PGR+curiosity | 0, 1, 1, 3, 4 | **2** |

`reacher_final` / `cheetah_final`의 SER도 curiosity/rnd가 섞여 있지만 seed는 0–4가
정확히 한 번씩이라 값 자체는 유효하다 (라벨만 혼동).
