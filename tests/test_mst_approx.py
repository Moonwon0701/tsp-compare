import itertools
import math
import unittest

from tsp.common import CountingDistance, is_valid_tour, tour_length, tsplib_distance
from tsp.held_karp import held_karp_solve
from tsp.instances import make_instance
from tsp.mst_approx import mst_approx, mst_length, preorder, prim_mst
from tsp.tsplib import INSTANCES, load_tsplib


def kruskal_length(points):
    """검증용 Kruskal: 모든 간선을 정렬해 사이클이 안 생기는 것만 고른다 (Prim과 독립 구현)."""
    n = len(points)
    root = list(range(n))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    edges = sorted((math.dist(points[i], points[j]), i, j)
                   for i, j in itertools.combinations(range(n), 2))
    total, used = 0.0, 0
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            root[ri] = rj
            total += d
            used += 1
            if used == n - 1:
                break
    return total


class PrimTest(unittest.TestCase):
    def test_matches_kruskal(self):
        for kind in ("uniform", "clustered", "grid"):
            for n in (10, 50, 200):
                with self.subTest(kind=kind, n=n):
                    pts = make_instance(kind, n)
                    self.assertAlmostEqual(mst_length(pts), kruskal_length(pts), places=6)

    def test_parent_forms_spanning_tree(self):
        pts = make_instance("uniform", 100)
        parent, total = prim_mst(pts)
        self.assertEqual(parent[0], -1)
        self.assertEqual(sum(1 for p in parent if p == -1), 1)
        # 모든 점에서 parent를 따라가면 루트에 도착해야 한다 (사이클 없음)
        for v in range(len(pts)):
            seen = set()
            while v != 0:
                self.assertNotIn(v, seen)
                seen.add(v)
                v = parent[v]
        edge_sum = sum(math.dist(pts[v], pts[p]) for v, p in enumerate(parent) if p != -1)
        self.assertAlmostEqual(edge_sum, total, places=6)

    def test_mst_not_longer_than_optimum(self):
        for kind in ("uniform", "clustered", "grid"):
            for n in (8, 10, 12):
                with self.subTest(kind=kind, n=n):
                    pts = make_instance(kind, n)
                    self.assertLessEqual(mst_length(pts), held_karp_solve(pts)[1] + 1e-9)

    def test_grid_mst_is_n_minus_one(self):
        # 간격 1인 격자의 MST는 길이 1인 간선 n−1개
        for n in (12, 100, 500):
            with self.subTest(n=n):
                self.assertAlmostEqual(mst_length(make_instance("grid", n)), n - 1, places=9)


class PreorderTest(unittest.TestCase):
    def test_small_tree(self):
        #      0
        #    /   \
        #   2     1
        #  / \
        # 3   4
        parent = [-1, 0, 0, 2, 2]
        self.assertEqual(preorder(parent), [0, 1, 2, 3, 4])

    def test_parent_comes_before_child(self):
        pts = make_instance("clustered", 200)
        parent, _ = prim_mst(pts)
        order = preorder(parent)
        pos = {v: i for i, v in enumerate(order)}
        for v, p in enumerate(parent):
            if p != -1:
                self.assertLess(pos[p], pos[v])


class MstApproxTest(unittest.TestCase):
    def test_valid_tour_on_all_kinds(self):
        for kind in ("uniform", "clustered", "grid"):
            for n in (8, 14, 100, 500):
                with self.subTest(kind=kind, n=n):
                    pts = make_instance(kind, n)
                    tour = mst_approx(pts)
                    self.assertTrue(is_valid_tour(tour, n))
                    self.assertEqual(tour[0], 0)

    def test_within_two_times_optimum(self):
        for kind in ("uniform", "clustered", "grid"):
            for n in (8, 10, 12):
                for index in range(3):
                    with self.subTest(kind=kind, n=n, index=index):
                        pts = make_instance(kind, n, index)
                        length = tour_length(pts, mst_approx(pts))
                        opt = held_karp_solve(pts)[1]
                        self.assertGreaterEqual(length, opt - 1e-9)
                        self.assertLessEqual(length, 2 * opt + 1e-9)

    def test_within_two_times_mst_at_large_n(self):
        # 최적을 모르는 큰 n에서도 2·MST 이하는 항상 성립해야 한다
        for kind in ("uniform", "clustered", "grid"):
            for n in (200, 1000):
                with self.subTest(kind=kind, n=n):
                    pts = make_instance(kind, n)
                    length = tour_length(pts, mst_approx(pts))
                    self.assertLessEqual(length, 2 * mst_length(pts) + 1e-6)

    def test_tsplib_within_two_times_optimum(self):
        for name in INSTANCES:
            with self.subTest(name=name):
                inst = load_tsplib(name)
                tour = mst_approx(inst.points)
                self.assertTrue(is_valid_tour(tour, len(inst.points)))
                self.assertLessEqual(tour_length(inst.points, tour, tsplib_distance), 2 * inst.optimum)

    def test_points_on_a_line(self):
        pts = [(0, 0), (3, 0), (1, 0), (2, 0)]
        tour = mst_approx(pts)
        self.assertEqual(tour, [0, 2, 3, 1])
        self.assertAlmostEqual(tour_length(pts, tour), 6)

    def test_root_parameter(self):
        pts = make_instance("uniform", 30)
        tour = mst_approx(pts, root=5)
        self.assertEqual(tour[0], 5)
        self.assertTrue(is_valid_tour(tour, 30))
        with self.assertRaises(ValueError):
            mst_approx(pts, root=30)

    def test_counts_only_search_distances(self):
        # Prim의 n(n−1)/2번뿐. 경로 길이 계산은 세지 않는다
        for n in (1, 2, 5, 50):
            with self.subTest(n=n):
                d = CountingDistance()
                mst_approx(make_instance("uniform", n), d)
                self.assertEqual(d.count, n * (n - 1) // 2)

    def test_tiny_inputs(self):
        self.assertEqual(mst_approx([]), [])
        self.assertEqual(mst_approx([(1, 1)]), [0])
        self.assertEqual(mst_approx([(0, 0), (3, 4)]), [0, 1])

    def test_duplicate_points(self):
        pts = [(0, 0), (2, 0), (0, 0), (2, 0)]
        tour = mst_approx(pts)
        self.assertTrue(is_valid_tour(tour, 4))
        self.assertAlmostEqual(tour_length(pts, tour), 4)


if __name__ == "__main__":
    unittest.main()
