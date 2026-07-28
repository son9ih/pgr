# exp/ — 실험 기록 & 코드 수정 방향

PGR(Prioritized Generative Replay) 기반 **RTB posterior fine-tuning** 연구의 실험 로그와
코드 수정 방향을 한 곳에 모아 두는 폴더.

## 파일

| 파일 | 내용 |
|---|---|
| [01_experiments.md](01_experiments.md) | wandb `*_final` 8개 프로젝트에서 복원한 **실제 실행 커맨드 + 최종 결과표** |
| [02_code_notes.md](02_code_notes.md) | 코드 맵, 알고리즘 요약, **코드를 읽으며 찾은 이슈와 수정 방향** |
| [03_backlog.md](03_backlog.md) | 다음에 돌릴 실험 / 고칠 코드 체크리스트 |
| [tools/fetch_wandb.py](tools/fetch_wandb.py) | wandb에서 run 메타데이터를 `data/wandb_final_runs.json`으로 덤프 |
| [tools/summarize.py](tools/summarize.py) | 덤프를 집계해 표(`--md`) / 커맨드(`--cmds`) 출력 |
| [data/wandb_final_runs.json](data/wandb_final_runs.json) | 2026-07-28 기준 205개 run 스냅샷 (config·args·summary) |

## 갱신 방법

```bash
python exp/tools/fetch_wandb.py            # wandb → data/wandb_final_runs.json
python exp/tools/summarize.py --md         # 결과표 재생성 → 01_experiments.md에 붙여넣기
python exp/tools/summarize.py --cmds       # 커맨드 재생성
```

## 기록 규칙

- **실험을 돌리기 전**에 `03_backlog.md`에 "무엇을/왜" 한 줄 추가한다.
- **끝난 뒤**에 `01_experiments.md`에 커맨드·commit·결과를 옮기고 backlog 항목을 지운다.
- 코드를 고치면 `02_code_notes.md`의 해당 이슈에 `→ fixed in <commit>`을 남긴다.
  (결과 비교가 깨질 수 있으므로 **수정 커밋 해시를 반드시 기록**할 것)

## 환경 (README.md 기준)

```bash
git submodule update --init --recursive
pip install -r requirements.txt          # Python 3.8, MuJoCo 210 필요
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

실행은 반드시 `python <script>.py ...` 형태로 한다. 스크립트에 실행 비트도 shebang도
없어서 `./synther/online/...py`로 직접 부르면 `Permission denied`가 난다.
