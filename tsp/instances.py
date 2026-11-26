"""실험 입력 생성기: 무작위(균일), 군집형(가우시안), 격자.

모든 입력은 (유형, n, 인스턴스 번호)로 정해지는 고정 seed로 만든다.
seed는 "uniform-100-3" 같은 문자열이다. random.Random은 문자열 seed를
SHA-512로 정수로 바꾸므로, 실행할 때마다(PYTHONHASHSEED와 무관하게) 같은 입력이 나온다.
"""

import math
import random

from .common import Point

AREA = 1000.0  # 무작위·군집형 점을 뽑는 정사각형 [0, AREA]²의 한 변
CLUSTER_SIGMA = 0.05 * AREA  # 군집 안 점들의 표준편차
KINDS = ("uniform", "clustered", "grid")


def seed_for(kind: str, n: int, index: int) -> str:
    """(유형, n, 인스턴스 번호)에 대응하는 고정 seed."""
    return f"{kind}-{n}-{index}"


def uniform_points(n: int, rng: random.Random) -> list[Point]:
    """정사각형 [0, AREA]² 안에서 균일 분포로 n개 점을 뽑는다."""
    return [(rng.uniform(0, AREA), rng.uniform(0, AREA)) for _ in range(n)]


def default_clusters(n: int) -> int:
    """군집 수: 점 4개당 1개, 최소 2개, 최대 5개 (n=8 → 2, n=12 → 3, n≥20 → 5)."""
    return max(2, min(5, n // 4))


def clustered_points(n: int, rng: random.Random, k: int | None = None,
                     sigma: float = CLUSTER_SIGMA) -> list[Point]:
    """군집 중심 k개를 정사각형 안쪽(10%~90%)에 뽑고, 각 점은 임의의 중심 주위에
    표준편차 sigma인 가우시안으로 뽑는다. 영역 밖으로 나간 점도 그대로 둔다."""
    if k is None:
        k = default_clusters(n)
    centers = [(rng.uniform(0.1 * AREA, 0.9 * AREA), rng.uniform(0.1 * AREA, 0.9 * AREA))
               for _ in range(k)]
    points = []
    for _ in range(n):
        cx, cy = centers[rng.randrange(k)]
        points.append((rng.gauss(cx, sigma), rng.gauss(cy, sigma)))
    return points


def grid_shape(n: int) -> tuple[int, int]:
    """n = m×k 이고 m, k ≥ 2, 한 변이 짝수인 격자 중 정사각형에 가장 가까운 (m, k), m ≤ k.

    이런 격자는 길이 1인 간선 n개로 한 바퀴를 돌 수 있어 최적 순회 길이가 n이다.
    명세의 격자 표(8 → 2×4, …, 2000 → 40×50)가 이 규칙으로 나온다.
    """
    best = None
    for m in range(2, math.isqrt(n) + 1):
        if n % m == 0:
            k = n // m
            if m % 2 == 0 or k % 2 == 0:
                best = (m, k)  # m이 커질수록 정사각형에 가까우므로 마지막 것이 최선
    if best is None:
        raise ValueError(f"n={n}은 한 변이 짝수인 2×2 이상 격자로 만들 수 없습니다")
    return best


def grid_points(n: int, rng: random.Random) -> list[Point]:
    """간격 1인 m×k 격자 점을 만들고 순서를 섞는다. 최적 순회 길이는 n이다."""
    m, k = grid_shape(n)
    points = [(float(x), float(y)) for y in range(m) for x in range(k)]
    rng.shuffle(points)
    return points


def grid_optimum(n: int) -> float:
    """격자 입력의 최적 순회 길이."""
    grid_shape(n)  # 만들 수 없는 n이면 여기서 ValueError
    return float(n)


_GENERATORS = {
    "uniform": uniform_points,
    "clustered": clustered_points,
    "grid": grid_points,
}


def make_instance(kind: str, n: int, index: int = 0) -> list[Point]:
    """(유형, n, 인스턴스 번호)로 정해지는 입력을 만든다. 같은 인자면 항상 같은 결과."""
    if kind not in _GENERATORS:
        raise ValueError(f"알 수 없는 입력 유형: {kind} (가능: {', '.join(KINDS)})")
    if n < 0:
        raise ValueError(f"n은 0 이상이어야 합니다: {n}")
    rng = random.Random(seed_for(kind, n, index))
    return _GENERATORS[kind](n, rng)
