"""실험 D: 최근접 이웃과 2-opt가 시작점(과 검사 순서)에 따라 얼마나 달라지는지 잰다 (명세 5장).

같은 입력, 같은 시작점 집합 S에 대해 세 가지를 잰다.
    nn             점 s에서 시작한 최근접 이웃
    two_opt_start  위 경로를 초기 경로로 한 2-opt (초기 경로와 검사 순서가 함께 바뀜)
    two_opt_rot    점 0에서 시작한 최근접 이웃 경로를 s칸 회전해 초기 경로로 준 2-opt
                   (경로 모양은 같고 검사 시작 위치만 바뀜)

품질 지표: 최적값을 알면 초과율(%), 모르면 MST 대비 비율. 실행 시간은 재지 않는다.
시간을 재지 않으므로 인스턴스 단위로 여러 프로세스에서 나눠 돌려도 결과가 같다.
"""

import random
import statistics

from .common import euclidean, tour_length
from .nearest_neighbor import nearest_neighbor
from .two_opt import two_opt

VARIANTS = ("nn", "two_opt_start", "two_opt_rot")


def sample_starts(n: int, count: int, seed: str) -> list[int]:
    """점 0을 포함한 시작점 count개. n ≤ count면 모든 점. 점 0은 본 실험(A·B·C)의 시작점이라 항상 넣는다."""
    if n <= count:
        return list(range(n))
    return [0] + sorted(random.Random(seed).sample(range(1, n), count - 1))


def run_instance(points, starts, optimum=None, mst=None, dist=euclidean,
                 with_two_opt: bool = True) -> list[dict]:
    """시작점마다 세 가지를 돌려 행 목록을 돌려준다. 행: variant, start, length, quality."""
    if optimum is None and mst is None:
        raise ValueError("optimum이나 mst 중 하나는 있어야 품질을 낼 수 있습니다")
    base = nearest_neighbor(points, dist, 0)
    rows = []

    def add(variant, s, tour):
        length = tour_length(points, tour, dist)
        quality = (length / optimum - 1) * 100 if optimum else length / mst
        rows.append({"variant": variant, "start": s, "length": length, "quality": quality})

    for s in starts:
        nn_tour = nearest_neighbor(points, dist, s)
        add("nn", s, nn_tour)
        if with_two_opt:
            add("two_opt_start", s, two_opt(points, dist, initial=nn_tour))
            add("two_opt_rot", s, two_opt(points, dist, initial=base[s:] + base[:s]))
    return rows


def summarize_instance(rows: list[dict]) -> list[dict]:
    """변형별 최소·중앙값·최대, 시작점 0의 값과 순위(1 = 가장 좋음), nn과 2-opt(시작점)의 상관계수."""
    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], {})[r["start"]] = r["quality"]
    corr = None
    if "nn" in by_variant and "two_opt_start" in by_variant:
        starts = sorted(by_variant["nn"])
        corr = pearson([by_variant["nn"][s] for s in starts],
                       [by_variant["two_opt_start"][s] for s in starts])
    out = []
    for variant, values in by_variant.items():
        q = list(values.values())
        q0 = values.get(0)
        out.append({
            "variant": variant, "starts": len(q),
            "min": min(q), "median": statistics.median(q), "max": max(q), "range": max(q) - min(q),
            "start0": q0,
            "start0_rank": None if q0 is None else 1 + sum(1 for v in q if v < q0 - 1e-9),
            "corr_nn_two_opt": corr if variant == "two_opt_start" else None,
        })
    return out


def pearson(xs, ys):
    """피어슨 상관계수. 한쪽이 모두 같은 값이면 None."""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
