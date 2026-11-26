import itertools
import unittest

from tsp.common import CountingDistance, is_valid_tour, tour_length, tsplib_distance
from tsp.held_karp import MAX_N, held_karp
from tsp.instances import make_instance


def brute_force(points):
    """점 0을 고정하고 나머지 순서를 모두 시도하는 전수 탐색 ((n−1)!가지)."""
    n = len(points)
    best = float("inf")
    for perm in itertools.permutations(range(1, n)):
        best = min(best, tour_length(points, [0, *perm]))
    return best


class HeldKarpTest(unittest.TestCase):
    def test_matches_brute_force(self):
        for kind in ("uniform", "clustered"):
            for n in (4, 5, 6, 7, 8):
                for index in range(3):
                    with self.subTest(kind=kind, n=n, index=index):
                        pts = make_instance(kind, n, index)
                        tour, length = held_karp(pts)
                        self.assertAlmostEqual(length, brute_force(pts), places=9)

    def test_tour_is_valid_and_length_consistent(self):
        for n in (4, 8, 12):
            with self.subTest(n=n):
                pts = make_instance("uniform", n)
                tour, length = held_karp(pts)
                self.assertTrue(is_valid_tour(tour, n))
                self.assertEqual(tour[0], 0)
                self.assertAlmostEqual(tour_length(pts, tour), length, places=9)

    def test_grid_optimum_is_n(self):
        for n in (8, 10, 12):
            with self.subTest(n=n):
                tour, length = held_karp(make_instance("grid", n))
                self.assertAlmostEqual(length, n, places=9)

    def test_tiny_inputs(self):
        self.assertEqual(held_karp([]), ([], 0.0))
        self.assertEqual(held_karp([(3, 3)]), ([0], 0.0))
        tour, length = held_karp([(0, 0), (3, 4)])
        self.assertEqual(tour, [0, 1])
        self.assertAlmostEqual(length, 10)
        tour, length = held_karp([(0, 0), (3, 0), (3, 4)])
        self.assertTrue(is_valid_tour(tour, 3))
        self.assertAlmostEqual(length, 12)

    def test_duplicate_points(self):
        pts = [(0, 0), (0, 0), (1, 0), (1, 0)]
        tour, length = held_karp(pts)
        self.assertTrue(is_valid_tour(tour, 4))
        self.assertAlmostEqual(length, 2)

    def test_counts_each_pair_once(self):
        d = CountingDistance()
        held_karp(make_instance("uniform", 10), d)
        self.assertEqual(d.count, 10 * 9 // 2)

    def test_works_with_tsplib_distance(self):
        pts = make_instance("uniform", 8)
        tour, length = held_karp(pts, tsplib_distance)
        self.assertEqual(length, tour_length(pts, tour, tsplib_distance))

    def test_rejects_large_n(self):
        with self.assertRaises(ValueError):
            held_karp(make_instance("uniform", MAX_N + 1))


if __name__ == "__main__":
    unittest.main()
