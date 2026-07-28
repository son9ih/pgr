# 02. 코드 맵 & 수정 방향

기준 commit: `404186a` (fix: time logging consistency), branch `explore`.

---

## 1. 파일 맵

| 파일 | 역할 |
|---|---|
| [synther/online/online_cond_ddpm_ori.py](../synther/online/online_cond_ddpm_ori.py) | **Ours.** SAC/REDQ 루프 + DDPM prior 학습 + RTB posterior fine-tuning (1356 lines) |
| [synther/online/online_cond_origin_baseline.py](../synther/online/online_cond_origin_baseline.py) | **SER / PGR / PGR-rnd** baseline (DDPM 버전, 1141 lines) |
| [synther/online/online_cond_ddpm_ori_abl.py](../synther/online/online_cond_ddpm_ori_abl.py) | ablation 변형 (untracked). ↓ §4 |
| [synther/diffusion/diffusion_cond.py](../synther/diffusion/diffusion_cond.py) | `DiffusionModel`, `QFlow`(RTB), `posterior_log_reward`, `compute_loss*` |
| [synther/diffusion/denoiser_network_cond.py](../synther/diffusion/denoiser_network_cond.py) | `ResidualMLPDenoiser` |
| [config/online/sac_cond_synther_dmc.gin](../config/online/sac_cond_synther_dmc.gin) | DMC 기본 설정 (UTD 20, cfg_scale 2.0, `skip_reward_norm=True`, terminal 없음) |
| [config/online/sac_cond_synther_openai.gin](../config/online/sac_cond_synther_openai.gin) | dmc.gin include + `skip_reward_norm=False` + `modelled_terminals=True` |
| `run_ori.sh` / `run_ori_ours.sh` / `run_abl.sh` | 런처 (⚠️ 현재 내용이 `*_final` 실행값과 어긋남 — I-8) |

## 2. 알고리즘 요약 (Ours)

`retrain_diffusion_every=10000` step(=10 epoch)마다:

1. **Prior**: replay buffer로 DDPM 학습 (`num_prior_epochs=100000` step, AdamW,
   cosine, grad-clip 1.0). EMA(β=0.995, every 10) 유지.
   `cond_top_frac` 상위 novelty transition을 conditioning signal로 쓰는 CFG
   (`cfg_dropout=0.25`, `cfg_scale=2.0`).
2. **Posterior (RTB)**: `QFlow`가 prior EMA 사본을 복제해 fine-tuning
   ([online_cond_ddpm_ori.py:761](../synther/online/online_cond_ddpm_ori.py#L761)).
   loss ([diffusion_cond.py:1294](../synther/diffusion/diffusion_cond.py#L1294)):

   ```
   0.5 * ( ( logZ + α·logpf_prior − logr.detach() − α·logpf_posterior ) / x_dim )²
   ```

   - `α = --alpha_rtb`, `logZ`는 학습 파라미터 (`diffusion_cond.py:559`)
   - `logr = log(posterior_log_reward(x))`, reward는 novelty measure
     ([diffusion_cond.py:1085](../synther/diffusion/diffusion_cond.py#L1085)):
     `curiosity` = forward-dynamics ensemble error `q_net(obs, next_obs, act)`,
     `rnd` = `agent.compute_intrinsic_reward(next_obs)`, `eco` = `compute_eco_reward(obs)`
   - reward는 `cond_normalizer` → `clamp(-1,1)` → `(x+1)/2` 로 [0,1]에 밀어넣는다
     (`--clip_reward 0.95` 분위수로 스케일 결정)
   - `--training_posterior both`: epoch마다 **on-policy**(모델에서 샘플)와
     **off-policy**(prior 샘플을 reward-weighted로 재사용)를 번갈아
   - DDIM 100 step, `eta=1.0`, gradient accumulation `--accumulation_steps`
3. **샘플링**: `--num_samples 1_000_000`을 `--sample_batch_size 100_000`씩 생성해
   diffusion buffer를 채우고, agent는 `diffusion_sample_ratio=0.5`로 섞어 학습.

## 3. 로깅

- `eval/AverageTestEpRet` — 성능
- `eval/DynMSE_mean|median` — 생성 transition의 one-step dynamics 오차.
  `compute_dynamic_mse_from_diffusion_buffer()`가 `set_state(s)` → `step(a)`로
  ground-truth `(s', r)`를 뽑아 비교
  ([online_cond_ddpm_ori.py:91](../synther/online/online_cond_ddpm_ori.py#L91)).
  **MuJoCo 전용** — `_mujoco_set_state_from_obs()`가 `env.unwrapped.set_state`를 요구해서
  DMC에서는 조용히 `False`를 반환하고 지표가 안 찍힌다.
- `eval/StateEnt` — k-NN state entropy
- `eval/AverageNormQBias` — Q bias
- `diffusion/*_time_*` — 최근 커밋(`d478081`~`404186a`)이 추가한 prior/RTB/샘플링 소요시간

---

## 4. 현재 진행 중인 갈래 (untracked)

`online_cond_ddpm_ori_abl.py`는 `online_cond_ddpm_ori.py`와 **딱 3곳** 다르다:

1. `epochs = 62` 하드코딩 (원본 100)
2. wandb group 접미사 `_ablation`
3. 샘플 생성 직후 각 샘플의 novelty reward를 계산해
   `ablation_data/{env}_epoch{E}_seed{S}_alpha{A}_{ts}/reward_vector_epoch{E}.npy`로 저장

→ `ablation.py` / `ablation_arrange.py` / `plot_hist_re.py`가 이걸 읽어
`ablation_figures/`에 **생성 샘플의 reward 분포 히스토그램**을 그린다.
즉 지금 하려는 건 *"RTB fine-tuning이 실제로 novelty 높은 쪽으로 분포를 옮겼는가"*의 직접 증거.
**이건 §5의 주장 1(DynMSE)과 짝이 되는 그림이니 논문 figure 후보다.**

---

## 5. 이슈 & 수정 방향

우선순위: **P0 = 결과 신뢰성에 직결 / P1 = 실험 설계 / P2 = 위생**

### I-1 (P0) `log(0)` → `-inf` 위험

[diffusion_cond.py:1085-1128](../synther/diffusion/diffusion_cond.py#L1085)에서
`q_r = clamp(normalize(q_r), -1, 1); q_r = (q_r + 1) / 2` → **정확히 0이 될 수 있다**.
그 뒤 `logr = self.posterior_log_reward(x).log()` (L949, L1174, L1309) 이므로
`-inf` → loss `inf`/`nan`. `--clip_reward 0.95`를 쓰면 하위 샘플이 clamp 하한에 붙기 때문에
드문 일이 아니다.

```python
# 수정안
q_r = ((q_r + 1) / 2).clamp_min(1e-6)
```

→ `ft_clip_grad=1.0`이 finger/quad에서 오히려 나빴던 것도(§01 관찰 6) 이 경로에서
튄 gradient를 clip이 방향까지 뭉갠 결과일 수 있다. **먼저 이걸 고치고 clip 실험을 다시 해석해야 한다.**

### I-2 (P0) off-policy prioritization 분포가 고정됨

[online_cond_ddpm_ori.py:801-806](../synther/online/online_cond_ddpm_ori.py#L801-L806)에서
`y_weights = softmax(ys)`를 한 번 만들고, 갱신 라인(L921-924)은 주석 처리되어 있다.
posterior 100~150 epoch 내내 **초기 prior 샘플의 reward 순위**만 재사용한다.
fine-tuning으로 분포가 옮겨가도 prioritization은 따라가지 않음 → `both` 모드의 off-policy
절반이 사실상 stale replay. 의도한 설계인지 확인하고, 아니라면 N epoch마다 재샘플·재계산.

### I-3 (P1) `alpha_rtb`가 태스크별로 0.5~10

baseline엔 대응 하이퍼파라미터가 없어서 "Ours만 튜닝됨" 지적을 받는다.
한 태스크(hopper 또는 walker)에서 **α ∈ {0.5, 1, 2, 4, 10} sensitivity 곡선**을 그려
"넓은 구간에서 baseline 이상"임을 보이는 게 방어에 필요하다.

### I-4 (P1) `ft_clip_grad`가 태스크별로 0/1

quad에서만 둘 다 돌았고 **clip=0이 +174 (910 vs 737)**. I-1 수정 후 전 태스크
clip=0으로 재실행할 값어치가 있다. 최소한 논문 표는 한쪽으로 통일해야 한다.

### I-5 (P1) `half_final`의 gin 불일치

HalfCheetah만 `sac_cond_synther_dmc.gin`(`skip_reward_norm=True`)로 돌았다.
Hopper/Walker2d는 `openai.gin`(`skip_reward_norm=False`). MuJoCo 표 3종을 같은 설정으로
맞추려면 HalfCheetah를 `openai.gin`으로 재실행하거나, 표에 각주를 달아야 한다.

### I-6 (P1) quad만 연산량이 다름

`accumulation_steps=8`, `num_posterior_epochs=150` (다른 태스크는 4 / 100).
Ours 쪽에만 추가 연산을 준 셈이라 quad의 +18.6%는 할인해서 봐야 한다.

### I-7 (P1) seed 중복/누락

`walker/hopper/half`의 SER, `fingerhard`의 PGR+curiosity에서 seed가 중복되고 하나씩 빠졌다
(원인·목록은 [01_experiments.md §4](01_experiments.md)).
런처가 SER을 novelty loop 안에서 제출한 게 원인 → **SER은 novelty loop 밖으로 빼야 한다.**

### I-8 (P2) 런처 스크립트가 실제 실행값과 어긋남

`run_ori.sh`는 지금 `ACC_STEPS=6`, `for env in HalfCheetah-v2`, `for seed in 6 7`로
남아 있어 `*_final` 실행(ACC 4/8, seed 0-4)을 재현하지 못한다.
`python exp/tools/summarize.py --cmds` 출력으로 **런처를 다시 생성**해 드리프트를 없앨 것.

### I-9 (P2) wandb project가 코드에 하드코딩

`wandb.init(project=env_name)` ([L253](../synther/online/online_cond_ddpm_ori.py#L253)).
`*_final`은 UI에서 수동으로 모은 것 → `--wandb_project` / `--wandb_group` 인자 추가.

### I-10 (P2) 총 epoch이 함수 본문에 하드코딩

[L240-243](../synther/online/online_cond_ddpm_ori.py#L240-L243)에서 env 이름으로 300/100을
정하고 gin의 `epochs`를 덮어쓴다. ablation 스크립트가 이 줄만 62로 바꾼 사본인 이유.
→ `--epochs`(기본 None이면 현재 규칙) 인자로 빼면 사본 유지가 필요 없다.

### I-11 (P2) reward 함수 안의 `print`

`posterior_log_reward()` 안에 `print(f'Check if q_r is bounded ...')`가 있어
**micro-batch마다** 찍힌다 (`accumulation_steps` × posterior epoch × 10회 재학습).
로그가 GB 단위로 붇고 I/O로 느려진다 → `if self.debug:` 게이트.

### I-12 (P2) DMC에 DynMSE 없음

가장 설득력 있는 지표(§01 관찰 1)가 MuJoCo 3종에서만 나온다.
DMC는 `env.unwrapped.physics.set_state()` / `get_state()`로 같은 걸 할 수 있으므로
`_mujoco_set_state_from_obs`에 dm_control 분기를 추가하면 8개 태스크 전부에서
같은 그림을 그릴 수 있다. **여기가 지금 가장 가성비 높은 코드 작업이다.**

---

## 6. 수정 이력

| 날짜 | 이슈 | commit | 비고 |
|---|---|---|---|
| | | | |
