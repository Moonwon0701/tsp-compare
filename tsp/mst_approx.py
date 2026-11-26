"""MST 2-근사 (double-tree): 최소 신장 트리를 전위 순회한 순서로 방문한다.

    1. Prim 알고리즘으로 최소 신장 트리(MST)를 만든다.
    2. 루트에서 트리를 깊이 우선으로 돌며 처음 만나는 순서(전위 순회)대로 점을 나열한다.
    3. 그 순서대로 방문하고 루트로 돌아온다.

트리의 모든 간선을 두 번씩 지나는 산책(오일러 투어)의 길이는 2·MST이다. 전위 순회는
이 산책에서 이미 방문한 점을 건너뛴(shortcut) 것이고, 삼각 부등식 때문에 건너뛰어도
길이가 늘지 않는다. 또 최적 순회에서 간선 하나를 빼면 신장 트리이므로 MST ≤ OPT이다.
따라서 경로 길이 ≤ 2·MST ≤ 2·OPT.

Prim은 배열 기반(완전 그래프에 맞는 방식)이라 O(n²)이고, 거리 계산은 n(n−1)/2번이다.
경로 길이는 측정용이라 여기서 계산하지 않는다.
"""

from .common import Point, euclidean


def prim_mst(points: list[Point], dist=euclidean, root: int = 0) -> tuple[list[int], float]:
    """배열 기반 Prim. (parent, MST 길이)를 돌려준다. parent[root] = -1.

    트리 밖의 각 점에 대해 '트리까지의 가장 가까운 거리(key)'를 들고 있다가,
    key가 가장 작은 점을 트리에 넣고 그 점 기준으로 나머지 key를 갱신한다.
    key가 같으면 번호가 작은 점을 먼저 넣는다.
    """
    n = len(points)
    if n == 0:
        return [], 0.0
    inf = float("inf")
    key = [inf] * n
    parent = [-1] * n
    in_tree = [False] * n
    key[root] = 0.0
    total = 0.0
    for _ in range(n):
        u, best = -1, inf
        for v in range(n):
            if not in_tree[v] and key[v] < best:
                u, best = v, key[v]
        in_tree[u] = True
        total += best
        pu = points[u]
        for v in range(n):
            if not in_tree[v]:
                d = dist(pu, points[v])
                if d < key[v]:
                    key[v] = d
                    parent[v] = u
    return parent, total


def preorder(parent: list[int], root: int = 0) -> list[int]:
    """parent 배열로 주어진 트리의 전위 순회 순서. 자식은 번호가 작은 것부터 방문한다."""
    children = [[] for _ in parent]
    for v, p in enumerate(parent):
        if p != -1:
            children[p].append(v)
    order = []
    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(reversed(children[u]))  # 번호가 작은 자식이 먼저 나오도록
    return order


def mst_approx(points: list[Point], dist=euclidean, root: int = 0) -> list[int]:
    """MST 2-근사 순회의 방문 순서를 돌려준다. 방문 순서는 root에서 시작한다."""
    n = len(points)
    if n == 0:
        return []
    if not 0 <= root < n:
        raise ValueError(f"root는 0 이상 {n} 미만이어야 합니다: {root}")
    parent, _ = prim_mst(points, dist, root)
    return preorder(parent, root)


def mst_length(points: list[Point], dist=euclidean) -> float:
    """MST 길이. 최적 순회 길이의 하한으로 쓴다 (MST ≤ OPT)."""
    return prim_mst(points, dist)[1]
