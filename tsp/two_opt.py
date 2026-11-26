"""2-opt 지역 탐색: 경로에서 두 간선을 골라 바꿨을 때 짧아지면 바꾸기를, 더 이상 짧아지지 않을 때까지 반복한다.

경로 … a → b … c → d … 에서 간선 (a, b), (c, d)를 빼고 (a, c), (b, d)를 넣으면
b부터 c까지 구간의 방향이 뒤집힌 경로가 된다. 길이 변화는

    delta = d(a, c) + d(b, d) − d(a, b) − d(c, d)

이고 delta < 0이면 바꾼다. 교차하는 두 간선은 이 교체로 항상 풀린다(교차를 풀면 짧아진다).

탐색 방식 (first improvement, 끝까지 훑기):
    한 바퀴(pass) 동안 모든 간선 쌍 (i, j)를 차례로 보며, 짧아지는 쌍을 만나면 바로 바꾸고 이어서 훑는다.
    한 바퀴 동안 한 번도 바꾸지 않았으면 멈춘다. 이때 경로는 "2-opt 지역 최적"이다.

거리 계산:
    - 초기 경로를 주지 않으면 점 0에서 시작한 최근접 이웃 경로를 쓰며, 그 비용(n(n−1)/2)도 포함한다.
    - 현재 경로의 간선 길이를 배열 elen에 들고 있어 d(a, b), d(c, d)는 다시 계산하지 않는다.
      처음 elen을 만들 때 n번, 쌍 하나를 검사할 때마다 2번(d(a, c), d(b, d)) 계산한다.
    - 한 바퀴 검사는 약 n²/2쌍이므로 O(n²), 전체는 O(n²) × 바퀴 수이다.
      바퀴 수는 최악에는 지수적이지만 실제로는 훨씬 적다고 알려져 있다.

품질 보장: 지역 최적에서 멈출 뿐 전체 최적은 보장하지 않는다.
"""

from .common import Point, euclidean
from .nearest_neighbor import nearest_neighbor

EPS = 1e-9  # 부동소수 오차로 0에 가까운 개선을 반복하지 않도록 하는 문턱


def two_opt_with_stats(points: list[Point], dist=euclidean,
                       initial: list[int] | None = None) -> tuple[list[int], dict]:
    """2-opt 지역 최적 경로와 통계를 돌려준다.

    통계: passes(바퀴 수, 마지막 확인 바퀴 포함), improvements(교체 횟수), checks(검사한 쌍 수)
    """
    n = len(points)
    tour = list(initial) if initial is not None else nearest_neighbor(points, dist)
    stats = {"passes": 0, "improvements": 0, "checks": 0}
    if n < 4:  # 서로 겹치지 않는 간선 쌍이 없다
        return tour, stats

    elen = [dist(points[tour[k]], points[tour[(k + 1) % n]]) for k in range(n)]
    improved = True
    while improved:
        improved = False
        stats["passes"] += 1
        for i in range(n - 2):
            a, b = points[tour[i]], points[tour[i + 1]]
            # i = 0이면 마지막 간선 (n−1, 0)은 점 tour[0]을 공유하므로 제외
            j_end = n if i > 0 else n - 1
            for j in range(i + 2, j_end):
                c, d = points[tour[j]], points[tour[(j + 1) % n]]
                stats["checks"] += 1
                d_ac = dist(a, c)
                d_bd = dist(b, d)
                delta = d_ac + d_bd - elen[i] - elen[j]
                if delta < -EPS:
                    # tour[i+1 .. j] 뒤집기, 안쪽 간선 길이도 거꾸로
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    elen[i + 1:j] = elen[i + 1:j][::-1]
                    elen[i] = d_ac
                    elen[j] = d_bd
                    stats["improvements"] += 1
                    improved = True
                    b = points[tour[i + 1]]
    return tour, stats


def two_opt(points: list[Point], dist=euclidean, initial: list[int] | None = None) -> list[int]:
    """2-opt 지역 최적 경로의 방문 순서. 초기 경로를 주지 않으면 최근접 이웃(점 0 시작)에서 출발한다."""
    return two_opt_with_stats(points, dist, initial)[0]
