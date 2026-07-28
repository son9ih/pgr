# 03. Backlog — 다음에 할 것

`[ ]` 대기 · `[~]` 진행중 · `[x]` 완료(→ 01/02로 이동)

## 코드 (실험 전에 먼저)

- [ ] **I-1** `posterior_log_reward` 반환값 `clamp_min(1e-6)` — `log(0)` 방지
      → 이거 고치기 전에 돌린 clip/α 결과는 재해석 대상
- [ ] **I-12** `_mujoco_set_state_from_obs`에 dm_control 분기 추가 → DMC 5종에도 DynMSE
- [ ] **I-11** reward 함수 안 `print` 디버그 플래그로 게이트
- [ ] **I-2** off-policy `y_weights` 갱신 여부 결정 (의도된 고정인지 확인 후)
- [ ] **I-9** `--wandb_project` / `--wandb_group` 인자화
- [ ] **I-10** `--epochs` 인자화 → `*_abl.py` 사본 제거
- [ ] **I-8** `exp/tools/summarize.py --cmds` 출력으로 `run_ori*.sh` 재생성

## 재실행 (결과 신뢰성)

- [ ] **I-7** 빠진 seed 채우기 — walker SER seed 1 / hopper SER seed 4 /
      half SER seed 3 / fingerhard PGR+curiosity seed 2
      (런처에서 SER을 novelty loop 밖으로 뺄 것)
- [ ] **I-5** HalfCheetah를 `sac_cond_synther_openai.gin`으로 재실행 (또는 표에 각주)
- [ ] **I-6** quad를 다른 태스크와 같은 `accumulation_steps=4`, `num_posterior_epochs=100`으로
      한 번 더 돌려 연산량 confound 제거

## 추가 실험 (논문 방어용)

- [ ] **I-3** α sensitivity: hopper 또는 walker에서 `alpha_rtb ∈ {0.5, 1, 2, 4, 10}` × seed 3
- [ ] **I-4** I-1 수정 후 `ft_clip_grad ∈ {0, 1}` 전 태스크 통일 실험
- [ ] Walker2d seed 5→10 (분산 ±1200~1570이라 지금은 PGR+rnd와 구분 불가)
- [ ] `fingerhard` `PGR+rnd`의 `NormQBias=266±597` — 발산 seed 특정하고 원인 확인
- [~] ablation: 생성 샘플 reward 분포 히스토그램 (`*_abl.py` + `plot_hist_re.py`)
      → RTB가 분포를 novelty 쪽으로 옮겼다는 직접 증거, figure 후보

## 논문 스토리 메모

주장 순서는 **"성능"이 아니라 "품질"**로 잡는 게 데이터에 맞는다:

1. RTB fine-tuning은 생성 transition의 dynamics 오차를 크게 줄인다 (DynMSE, MuJoCo 3/3)
2. 동시에 state entropy를 유지·향상시킨다 (novelty를 좇되 prior를 안 깬다)
3. 그 결과 8개 중 7개에서 최고 return, 특히 탐험이 병목인 태스크(finger-hard +87%,
   hopper +24%, quad +19%)에서 마진이 크다
4. 나머지(cheetah/half/reacher +4~8%, walker −7%)는 오차범위 — **과장하지 말 것**
