# TSP 근사 해법 비교

평면 위 n개 점을 모두 한 번씩 방문하고 돌아오는 외판원 순회 경로(TSP)를 세 근사 해법으로 구하고, 같은 조건에서 **품질**과 **속도**를 비교하는 프로젝트입니다.

| 방식 | 원리 | 시간 | 품질 보장 |
|---|---|---|---|
| 최근접 이웃 | 지금 점에서 가장 가까운 미방문 점으로 이동 | O(n²) | 없음 (최악 약 ½·log₂n배) |
| MST 2-근사 | 최소 신장 트리를 전위 순회 | O(n²) | 최적의 2배 이하 |
| 2-opt | 최근접 이웃 경로에서 교차 간선을 교체 | O(n²) × 바퀴 수 | 지역 최적 |
| Held-Karp (기준값) | 비트마스크 동적 계획법 | O(n²·2ⁿ) | 최적 (n ≤ 16) |

- **품질**: 최적값을 알면 초과율 = (경로 길이 − 최적) ÷ 최적 × 100%, 모르면 MST 대비 비율로 잽니다.
- **속도**: 주 지표는 **거리 계산 횟수**입니다. 하드웨어와 무관하게 다시 돌려도 같은 값이 나옵니다. 실행 시간은 보조 지표입니다.
- 자세한 규칙은 [명세 docs/SPEC.md](docs/SPEC.md)에 있습니다.

## 환경

측정에 쓴 환경입니다. 실험을 돌리면 `results/environment.json`에 다시 기록됩니다.

| 항목 | 값 |
|---|---|
| Python | **3.12.10** (CPython) |
| 의존 패키지 | matplotlib 3.11.2 (그래프에만 사용) |
| OS | Windows 11 (10.0.26200) |
| CPU | Intel64 Family 6 Model 198, 논리 코어 24개 |
| 테스트 | 표준 라이브러리 `unittest` |

> **Python 버전 주의**
>
> 무작위·군집형·격자 입력은 `random` 모듈에 고정 seed를 넣어 만듭니다. Python 공식 문서([random — Notes on Reproducibility](https://docs.python.org/3.12/library/random.html#notes-on-reproducibility))는 버전이 바뀌어도 seed 방식과 `random()`의 출력만 같게 유지된다고 보장합니다. 이 프로젝트가 쓰는 `uniform`·`gauss`·`randrange`·`shuffle`은 버전에 따라 달라질 수 있습니다.
>
> 따라서 **Python 3.12.10이 아니면 입력 점이 달라져 결과 숫자가 이 저장소의 `results/`와 다를 수 있습니다.** 경향은 같게 나올 것으로 예상하지만 숫자를 그대로 재현하려면 같은 버전을 쓰세요. TSPLIB 실험은 파일에서 입력을 읽으므로 이 영향을 받지 않습니다.

> **그래프 글꼴**
>
> `plot.py`는 [Pretendard](https://github.com/orioncactus/pretendard) 글꼴을 씁니다. 설치되어 있지 않으면 맑은 고딕 → AppleGothic → NanumGothic 순서로 대신하므로, 그래프의 글자 모양과 너비가 조금 달라질 수 있습니다. 숫자와 그래프 내용은 같습니다.

## 설치

```bash
git clone https://github.com/Moonwon0701/tsp-compare.git
cd tsp-compare
python -m venv .venv
```

가상 환경을 켜고 의존 패키지를 설치합니다.

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Windows에서 `python`이 Microsoft Store 안내로 연결되면 `python` 대신 `py`를 쓰세요(예: `py -m venv .venv`).

## 실행

모든 명령은 저장소 최상위 폴더에서 실행합니다.

### 1. 테스트

```bash
python -m unittest
```

알고리즘 정확성(전수 탐색·독립 구현과 비교, 2배 보장, 2-opt 지역 최적 등), 입력 재현성, 측정 규칙, 그래프 생성까지 137개 테스트가 있습니다. 몇 초 안에 끝납니다.

### 2. 실험

| 명령 | 실험 | 내용 | 걸린 시간 (위 환경) |
|---|---|---|---|
| `python bench.py small` | A | n = 8·10·12·14, 무작위·군집형·격자 × 10개, Held-Karp 최적값 대비 초과율 | 약 20초 |
| `python bench.py large` | B | n = 100~2000, 무작위·군집형(MST 대비 비율)·격자(초과율) × 10개, 2-opt는 n ≤ 1000 | 약 6~7분 |
| `python bench.py tsplib` | C | TSPLIB 5개, 알려진 최적값 대비 초과율 | 약 5초 |
| `python bench.py start` | D | 최근접 이웃·2-opt의 시작점 민감도 (여러 프로세스로 병렬 실행) | 약 2분 |
| `python bench.py smoke` | – | 러너가 도는지만 빠르게 확인 (결과는 저장소에 넣지 않음) | 몇 초 |

- 실험 A·B·C는 시간을 5번 재서 중앙값을 씁니다. `--reps 3`처럼 바꿀 수 있습니다.
- 같은 입력의 알고리즘들은 시간을 번갈아 잽니다. 실행 시간은 실행할 때마다 10~30% 흔들릴 수 있으니 절대값보다 알고리즘 사이 비율과 증가 추세로 읽으세요. 측정 중에는 다른 무거운 프로그램을 끄는 것이 좋습니다.
- 품질(길이·초과율·비율)과 거리 계산 횟수는 같은 Python 버전이면 몇 번을 돌려도 똑같습니다.

### 3. 그래프

```bash
python plot.py            # 그래프 6개 전부
python plot.py small      # 하나만 (small, large, tsplib, start)
```

`results/`의 CSV를 읽어 PNG를 만듭니다. 실험을 다시 돌린 뒤 실행하면 새 결과로 그려집니다.

## 결과 파일

모든 결과는 `results/`에 있습니다. 실험마다 원자료(`<이름>.csv`)와 요약(`<이름>_summary.csv`)이 있습니다.

| 파일 | 내용 |
|---|---|
| `small.csv`, `small_summary.csv` | 실험 A: 인스턴스·알고리즘별 길이, 초과율, 거리 계산 횟수, 시간 |
| `large.csv`, `large_summary.csv`, `large_slopes.csv` | 실험 B, 그리고 log-log 기울기 (O(n²)이면 2) |
| `tsplib.csv`, `tsplib_summary.csv` | 실험 C |
| `start_sensitivity.csv`, `start_sensitivity_summary.csv` | 실험 D: 시작점별 원자료, 인스턴스별 최소·중앙값·최대·점 0의 순위·상관계수 |
| `environment.json` | 측정 환경 |
| `small_excess.png` | 실험 A 유형별 평균 초과율 |
| `small_held_karp_time.png` | Held-Karp 실행 시간과 n²·2ⁿ 비례 곡선 |
| `large_quality.png` | 실험 B 품질 |
| `large_cost_loglog.png` | 거리 계산 횟수(주)·실행 시간(보조)의 log-log 그래프 |
| `tsplib_excess.png` | 실험 C 값과 실험 D의 시작점별 범위 |
| `start_spread.png` | 시작점에 따른 퍼짐이 n과 함께 줄어드는 모습 |

### 주요 결과 요약

| 입력 | 최근접 이웃 | MST 2-근사 | 2-opt |
|---|---|---|---|
| 작은 n 무작위 (초과율, 10개 평균) | 7.6 ~ 12.8% | 20.8 ~ 24.3% | 0.4 ~ 1.5% |
| 큰 n 무작위 (MST 대비 비율) | 1.39 ~ 1.43 | 1.56 ~ 1.58 | 1.20 ~ 1.22 |
| 큰 n 격자 (초과율) | 23.8 ~ 25.3% | 39.8 ~ 42.8% | 5.5 ~ 5.9% |
| TSPLIB (초과율, 점 0 시작) | 19.1 ~ 30.7% | 35.4 ~ 43.4% | 1.8 ~ 9.2% |
| 거리 계산 횟수 log-log 기울기 | 2.00 | 2.00 | 2.16 ~ 2.18 |

TSPLIB 값은 시작점 하나의 결과라 시작점에 따라 크게 달라집니다(실험 D, `tsplib_excess.png`). 해석은 보고서를 참고하세요.

## 알고리즘 시각화

알고리즘이 한 단계씩 움직이는 모습을 보여 주는 HTML 파일입니다. 외부 라이브러리 없이 파일 하나로 동작합니다. GitHub에서는 소스만 보이므로 **저장소를 받은 뒤 브라우저로 여세요**.

- [docs/viz/held_karp.html](docs/viz/held_karp.html): 점 6개에서 DP 표가 부분집합 크기별로 채워지는 과정
- [docs/viz/nearest_neighbor.html](docs/viz/nearest_neighbor.html): 가장 가까운 점을 고르는 단계와 끝에 몰리는 긴 간선
- [docs/viz/mst_approx.html](docs/viz/mst_approx.html): MST 만들기 → 트리 두 번 돌기 → 건너뛰기
- [docs/viz/two_opt.html](docs/viz/two_opt.html): 간선 두 개를 바꾸며 교차를 푸는 과정과 지역 최적

## 저장소 구성

```
tsp-compare/
├─ tsp/              알고리즘과 공통 함수
│  ├─ common.py          거리, 경로 길이, 경로 검증, 거리 계산 카운터
│  ├─ instances.py       무작위·군집형·격자 입력 생성 (고정 seed)
│  ├─ tsplib.py          TSPLIB 로더, 최적값 표, 최적 경로 읽기
│  ├─ held_karp.py       Held-Karp DP
│  ├─ nearest_neighbor.py
│  ├─ mst_approx.py      Prim + 전위 순회
│  ├─ two_opt.py
│  ├─ bench.py           측정 러너 (시간·거리 계산 횟수 분리, 번갈아 재기)
│  └─ start_sensitivity.py  실험 D
├─ bench.py          실험 실행 스크립트
├─ plot.py           그래프 생성 스크립트
├─ tests/            unittest
├─ data/tsplib/      TSPLIB 인스턴스와 최적 경로 파일
├─ results/          실험 결과 CSV·그래프
├─ docs/             명세(SPEC.md), 알고리즘 시각화(viz/)
└─ report/           보고서, AI 사용 기록
```

## 데이터 출처

- TSPLIB95 인스턴스 eil51, berlin52, kroA100, ch150, a280과 각각의 최적 경로 파일(`*.opt.tour`): [TSPLIB95](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) (Universität Heidelberg)
- 최적값 426, 7542, 21282, 6528, 2579: 같은 사이트의 `tsp/TSP-BEST.html`. 최적 경로를 TSPLIB 반올림 거리로 계산하면 이 값과 정확히 같다는 것을 테스트로 확인합니다.
- 무작위·군집형·격자 입력은 `tsp/instances.py`가 고정 seed로 만듭니다(파일로 저장하지 않음).

## AI 사용

이 프로젝트는 Claude Code(Claude Opus 5.5)의 도움을 받아 진행했습니다. 요청 전문과 응답 요약은 [report/AI_USAGE_LOG.md](report/AI_USAGE_LOG.md)에 있습니다.
