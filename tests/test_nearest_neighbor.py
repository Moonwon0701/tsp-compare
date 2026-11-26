import math
import unittest

from tsp.common import CountingDistance, is_valid_tour, tour_length, tsplib_distance
from tsp.held_karp import held_karp
from tsp.instances import make_instance
from tsp.nearest_neighbor import nearest_neighbor
from tsp.tsplib import load_tsplib


class NearestNeighborTest(unittest.TestCase):
    def test_valid_tour_on_all_kinds(self):
        for kind in ("uniform", "clustered", "grid"):
            for n in (8, 14, 100, 500):
                with self.subTest(kind=kind, n=n):
                    pts = make_instance(kind, n)
                    tour, length = nearest_neighbor(pts)
                    self.assertTrue(is_valid_tour(tour, n))
                    self.assertEqual(tour[0], 0)
                    self.assertAlmostEqual(length, tour_length(pts, tour), places=6)

    def test_valid_tour_on_tsplib(self):
        inst = load_tsplib("a280")
        tour, _ = nearest_neighbor(inst.points)
        self.assertTrue(is_valid_tour(tour, len(inst.points)))

    def test_always_moves_to_nearest_unvisited(self):
        pts = make_instance("uniform", 30, 2)
        tour, _ = nearest_neighbor(pts)
        for step in range(len(tour) - 1):
            cur, nxt = tour[step], tour[step + 1]
            rest = tour[step + 1:]
            nearest = min(math.dist(pts[cur], pts[r]) for r in rest)
            self.assertAlmostEqual(math.dist(pts[cur], pts[nxt]), nearest)

    def test_points_on_a_line(self):
        pts = [(0, 0), (6, 0), (1, 0), (3, 0)]
        tour, length = nearest_neighbor(pts)
        self.assertEqual(tour, [0, 2, 3, 1])
        self.assertAlmostEqual(length, 12)

    def test_tie_prefers_smaller_index(self):
        pts = [(0, 0), (1, 0), (-1, 0), (5, 0)]
        tour, _ = nearest_neighbor(pts)
        self.assertEqual(tour[1], 1)

    def test_start_parameter(self):
        pts = make_instance("uniform", 20)
        tour, length = nearest_neighbor(pts, start=7)
        self.assertEqual(tour[0], 7)
        self.assertTrue(is_valid_tour(tour, 20))
        self.assertAlmostEqual(length, tour_length(pts, tour), places=6)
        with self.assertRaises(ValueError):
            nearest_neighbor(pts, start=20)

    def test_not_shorter_than_optimum_and_within_log_bound(self):
        # 최적 이상이고, 알려진 상한 ½(⌈log₂ n⌉ + 1)·OPT 이하여야 한다
        for kind in ("uniform", "clustered", "grid"):
            for n in (8, 10, 12):
                for index in range(3):
                    with self.subTest(kind=kind, n=n, index=index):
                        pts = make_instance(kind, n, index)
                        _, nn_len = nearest_neighbor(pts)
                        _, opt = held_karp(pts)
                        bound = 0.5 * (math.ceil(math.log2(n)) + 1)
                        self.assertGreaterEqual(nn_len, opt - 1e-9)
                        self.assertLessEqual(nn_len, bound * opt + 1e-9)

    def test_sometimes_not_optimal(self):
        # 품질 보장이 없다는 것을 확인: 작은 무작위 입력 중 최적보다 긴 경우가 있어야 한다
        worse = 0
        for index in range(10):
            pts = make_instance("uniform", 10, index)
            if nearest_neighbor(pts)[1] > held_karp(pts)[1] + 1e-9:
                worse += 1
        self.assertGreater(worse, 0)

    def test_distance_count(self):
        for n in (2, 5, 50):
            with self.subTest(n=n):
                d = CountingDistance()
                nearest_neighbor(make_instance("uniform", n), d)
                self.assertEqual(d.count, n * (n - 1) // 2 + 1)

    def test_tiny_inputs(self):
        self.assertEqual(nearest_neighbor([]), ([], 0.0))
        self.assertEqual(nearest_neighbor([(4, 4)]), ([0], 0.0))
        tour, length = nearest_neighbor([(0, 0), (3, 4)])
        self.assertEqual(tour, [0, 1])
        self.assertAlmostEqual(length, 10)

    def test_duplicate_points(self):
        pts = [(0, 0), (2, 0), (0, 0), (2, 0)]
        tour, length = nearest_neighbor(pts)
        self.assertTrue(is_valid_tour(tour, 4))
        self.assertAlmostEqual(length, 4)

    def test_works_with_tsplib_distance(self):
        pts = load_tsplib("eil51").points
        tour, length = nearest_neighbor(pts, tsplib_distance)
        self.assertEqual(length, tour_length(pts, tour, tsplib_distance))


if __name__ == "__main__":
    unittest.main()
