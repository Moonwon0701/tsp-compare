"""모든 알고리즘이 함께 쓰는 거리 함수, 경로 길이, 경로 검증."""

import math

Point = tuple[float, float]


def euclidean(p: Point, q: Point) -> float:
    """두 점 사이의 유클리드 거리."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def tsplib_distance(p: Point, q: Point) -> int:
    """TSPLIB EUC_2D 거리: 유클리드 거리를 가장 가까운 정수로 반올림한 값.

    TSPLIB 문서의 정의 nint(sqrt(xd*xd + yd*yd)), 즉 (int)(x + 0.5)를 따른다.
    공식 최적값이 이 거리로 계산되어 있으므로 TSPLIB 초과율을 낼 때 쓴다.
    """
    return int(euclidean(p, q) + 0.5)


class CountingDistance:
    """호출 횟수를 세는 거리 함수.

    알고리즘에 dist 인자로 넘기면 거리 계산 횟수를 잴 수 있다.
    시간 측정 때는 카운터 오버헤드가 섞이지 않도록 euclidean을 그대로 넘긴다.
    """

    def __init__(self, base=euclidean):
        self.base = base
        self.count = 0

    def __call__(self, p: Point, q: Point) -> float:
        self.count += 1
        return self.base(p, q)

    def reset(self) -> None:
        self.count = 0


def tour_length(points: list[Point], tour: list[int], dist=euclidean) -> float:
    """방문 순서 tour대로 돌고 시작점으로 돌아오는 순회 길이."""
    n = len(tour)
    if n < 2:
        return 0
    total = 0
    for i in range(n):
        total += dist(points[tour[i]], points[tour[(i + 1) % n]])
    return total


def tour_problems(tour, n: int) -> list[str]:
    """tour가 0..n-1의 순열이 아니면 그 이유 목록을, 올바르면 빈 목록을 돌려준다."""
    problems = []
    if len(tour) != n:
        problems.append(f"길이 {len(tour)} != 점 개수 {n}")
    bad = [v for v in tour if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < n]
    if bad:
        problems.append(f"범위 밖이거나 정수가 아닌 값: {bad[:5]}")
    seen = set()
    dup = []
    for v in tour:
        if v in seen:
            dup.append(v)
        seen.add(v)
    if dup:
        problems.append(f"중복 방문: {dup[:5]}")
    missing = [v for v in range(n) if v not in seen]
    if missing:
        problems.append(f"방문하지 않은 점: {missing[:5]}")
    return problems


def is_valid_tour(tour, n: int) -> bool:
    """모든 점을 정확히 한 번씩 방문하는 순열이면 True."""
    return not tour_problems(tour, n)


def validate_tour(tour, n: int) -> None:
    """올바른 순회가 아니면 ValueError를 낸다."""
    problems = tour_problems(tour, n)
    if problems:
        raise ValueError("잘못된 경로: " + "; ".join(problems))
