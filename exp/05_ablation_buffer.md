# 05. Ablation — synthetic replay buffer capacity (reacher-hard)

실행: 2026-07-28 ~ 07-29 · 10 run 전부 `finished` · 데이터 [data/buffer_ablation_reacher.csv](data/buffer_ablation_reacher.csv)
그림: `exp/figures/buffer_ablation_reacher.{png,pdf}` (생성물이라 untracked — CSV로 재생성)

## 1. 무엇을 바꿨나

diffusion으로 생성한 transition이 들어가는 **synthetic replay buffer의 capacity** 하나만.

| arm | capacity | `num_samples` | `sample_batch_size` | project |
|---|---|---|---|---|
| 50k | 50,000 | 50,000 | 50,000 | `reacher_buffer_ablation` |
| 200k | 200,000 | 200,000 | 100,000 | `reacher_buffer_ablation` |
| 1M (기준) | 1,000,000 | 1,000,000 | 100,000 | `reacher_final` (기록된 run) |

- 세 arm 모두 **`--posterior_param residual`** — 1M이 기록된 데이터라서 거기에 맞췄다.
  reacher는 dmc.gin이라 I-5 무관, I-16은 로깅값 불변, I-13은 no-op이고 `env_defaults`의
  reacher 값(α=10.0, clip=1.0, acc=4, post_epochs=100, top_frac=0.25)이 기록된 run과 동일하다.
  → **세 곡선은 capacity만 다르다.**
- `num_samples`를 capacity에 맞춘 이유: 남는 샘플은 링버퍼에서 밀려날 뿐이고 유지되는
  샘플은 어느 쪽이든 posterior에서 i.i.d.라 분포가 같다. sampling만 20배 싸진다.
- ⚠️ **200k seed 4만** `--sample_batch_size 50000 --accumulation_steps 8`로 돌았다
  (GPU 메모리 때문. 10개 동시 실행 시 표준 설정으로는 어떤 배치에서도 여유가 92 MiB였다).
  총 샘플 수 200,000과 effective batch 1024는 동일하고 RNG 소비 순서만 다르다 —
  통계적으로 같은 arm이지만 bit-identical한 경로는 아니다.

재현:
```bash
bash scripts/run_ablation_buffer.sh              # DRY=1로 계획 확인
python exp/tools/fetch_curves.py                 # -> exp/data/buffer_ablation_reacher.csv
python exp/tools/plot_buffer_ablation.py --smooth 5
```

## 2. 결과

**최종 성능** — seed별로 epoch 80–99를 평균한 뒤 seed 간 집계 (n=5):

| capacity | mean ± std | seed별 (정렬) | min | median |
|---|---|---|---|---|
| 50,000 | 533.1 ± 206.2 | 244, 426, 553, 678, 765 | 244 | 553 |
| 200,000 | 639.5 ± 181.7 | 343, 638, 642, 777, 798 | 343 | 642 |
| **1,000,000** | **727.5 ± 68.0** | 656, 666, 743, 751, 821 | **656** | 743 |

**학습 전체 (100 epoch 평균)**: 50k 344.8 ± 156.0 · 200k 365.1 ± 126.0 · 1M 442.0 ± 32.7

**유의성** (n=5):

| 비교 | 차이 | Welch t | Mann-Whitney |
|---|---|---|---|
| 50k vs 200k | +106.4 | p = 0.412 | p = 0.210 |
| 50k vs 1M | +194.4 | p = 0.103 | p = 0.111 |
| 200k vs 1M | +88.0 | p = 0.356 | p = 0.210 |

## 3. 읽어낸 것

1. **capacity가 클수록 좋다는 순서가 세 arm에서 단조롭게 나온다** — 최종 성능,
   100 epoch 평균, median, 최솟값 전부 50k < 200k < 1M.
2. **그런데 n=5로는 어느 쌍도 유의하지 않다** (최소 p = 0.10). 평균 차이만으로
   "capacity를 키우면 성능이 오른다"고 주장하면 근거가 약하다.
3. **가장 견고한 신호는 평균이 아니라 분산이다.**

   | capacity | std | seed 간 range |
   |---|---|---|
   | 50,000 | 206.2 | 520.5 |
   | 200,000 | 181.7 | 455.1 |
   | 1,000,000 | **68.0** | **165.0** |

   1M의 **최악 seed(656)가 50k·200k의 median(553, 642)보다 높다.**
   50k는 244까지 떨어지는 seed가 있고 200k도 343이 있는데, 1M은 656 아래로 내려간 seed가 없다.
   → capacity의 효과는 *"더 잘 배운다"*보다 **"seed에 따라 실패하지 않는다"**로 서술하는 게
   데이터에 맞는다. 100 epoch 평균의 std(±156 → ±126 → ±33)도 같은 방향이다.
4. 그림([figures/buffer_ablation_reacher.png](figures/))에서도 1M의 ±1σ 밴드가 눈에 띄게 좁다.
   50k와 200k의 밴드는 거의 겹친다 — 이 둘은 사실상 구분되지 않는다.

## 4. 이 결과를 쓸 때 주의할 것

- **wall-clock은 비교값으로 쓸 수 없다.** 50k는 GPU당 3개, 200k는 2개씩 얹혀 돌았다
  (50k 9.0 ep/h vs 200k 10.6~11.0 ep/h). 이건 capacity 효과가 아니라 GPU 경합이다.
  시간을 주장하려면 GPU 독점 상태에서 따로 재야 한다.
- 유의성을 원하면 **seed를 10개로 늘리는 게 가장 확실하다** — 지금 효과 크기(194)와
  분산(±206)이면 n=5로는 검정력이 부족하다.
- 세 arm 모두 `residual` parameterization이다. `absolute`(I-15)로 넘어가면
  이 ablation도 다시 돌려야 한다.
