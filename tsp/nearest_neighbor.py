"""최근접 이웃(Nearest Neighbor): 지금 위치에서 가장 가까운 미방문 점으로 계속 이동한다.

    1. 시작점에서 출발한다.
    2. 아직 방문하지 않은 점 중 지금 점에서 가장 가까운 점으로 이동한다.
    3. 모든 점을 방문하면 시작점으로 돌아온다.

k번째 이동에서 남은 n−k개 점을 모두 보므로 거리 계산은 n(n−1)/2 + 1번(마지막 +1은
시작점으로 돌아오는 간선), 시간은 O(n²)이다.

품질 보장은 상수배가 없다. 알려진 상한은 최적의 ½(⌈log₂ n⌉ + 1)배이다
(Rosenkrantz, Stearns, Lewis, 1977).
"""

from .common import Point, euclidean


def nearest_neighbor(points: list[Point], dist=euclidean, start: int = 0) -> tuple[list[int], float]:
    """최근접 이웃 순회 (방문 순서, 길이)를 돌려준다. 방문 순서는 start에서 시작한다.

    거리가 같은 후보가 여러 개면 번호가 작은 점을 고른다.
    """
    n = len(points)
    if n == 0:
        return [], 0.0
    if not 0 <= start < n:
        raise ValueError(f"start는 0 이상 {n} 미만이어야 합니다: {start}")

    tour = [start]
    unvisited = [i for i in range(n) if i != start]  # 번호 오름차순 유지
    current = start
    length = 0.0
    cur_point = points[current]
    while unvisited:
        best_pos, best_d = 0, dist(cur_point, points[unvisited[0]])
        for pos in range(1, len(unvisited)):
            d = dist(cur_point, points[unvisited[pos]])
            if d < best_d:  # 같으면 앞(번호가 작은 점)을 유지
                best_pos, best_d = pos, d
        current = unvisited.pop(best_pos)
        cur_point = points[current]
        tour.append(current)
        length += best_d
    if n > 1:
        length += dist(cur_point, points[start])
    return tour, length
