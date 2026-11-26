"""Held-Karp 동적 계획법: 작은 n에서 정확한 최적 순회를 구한다 (기준값 용도).

상태 dp[S][j] = 점 0에서 출발해 집합 S의 점을 모두 한 번씩 거쳐 점 j에서 끝나는
경로 중 가장 짧은 길이. (S는 점 1..n-1의 부분집합, j ∈ S)

    dp[{j}][j]   = d(0, j)
    dp[S][j]     = min over i ∈ S−{j} of  dp[S−{j}][i] + d(i, j)
    최적 길이    = min over j of  dp[{1..n-1}][j] + d(j, 0)

부분집합 S는 비트마스크(점 k ↔ 비트 k−1)로 나타낸다.
상태 수는 (n−1)·2^(n−1), 상태마다 다음 점 후보를 최대 n−1개 보므로 시간 O(n²·2ⁿ), 메모리 O(n·2ⁿ).
"""

from .common import Point, euclidean

MAX_N = 16  # 이보다 크면 Python에서 너무 오래 걸리고 메모리도 많이 쓴다


def held_karp(points: list[Point], dist=euclidean) -> tuple[list[int], float]:
    """최적 순회 (방문 순서, 길이)를 돌려준다. 방문 순서는 점 0에서 시작한다.

    거리는 점 쌍마다 한 번씩, 총 n(n−1)/2번 계산해 표에 저장한 뒤 재사용한다.
    """
    n = len(points)
    if n > MAX_N:
        raise ValueError(f"Held-Karp는 n ≤ {MAX_N}에서만 씁니다 (n={n})")
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0

    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d[i][j] = d[j][i] = dist(points[i], points[j])

    m = n - 1  # 점 0을 뺀 나머지 점 수
    full = (1 << m) - 1
    inf = float("inf")
    dp = [[inf] * m for _ in range(1 << m)]
    parent = [[-1] * m for _ in range(1 << m)]

    for j in range(m):
        dp[1 << j][j] = d[0][j + 1]

    # mask는 커지는 순서로만 갱신되므로, mask를 볼 때 그 값은 이미 확정되어 있다
    for mask in range(1, 1 << m):
        row = dp[mask]
        for j in range(m):
            cost = row[j]
            if cost == inf:
                continue
            dj = d[j + 1]
            for k in range(m):
                bit = 1 << k
                if mask & bit:
                    continue
                new_cost = cost + dj[k + 1]
                nxt = mask | bit
                if new_cost < dp[nxt][k]:
                    dp[nxt][k] = new_cost
                    parent[nxt][k] = j

    best_len, best_end = inf, -1
    for j in range(m):
        total = dp[full][j] + d[j + 1][0]
        if total < best_len:
            best_len, best_end = total, j

    # parent를 거꾸로 따라가며 경로 복원
    order = []
    mask, j = full, best_end
    while j != -1:
        order.append(j + 1)
        prev = parent[mask][j]
        mask ^= 1 << j
        j = prev
    order.reverse()
    return [0] + order, best_len
