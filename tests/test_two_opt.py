import itertools
import unittest

from tsp.common import CountingDistance, is_valid_tour, tour_length, tsplib_distance
from tsp.held_karp import held_karp_solve
from tsp.instances import make_instance
from tsp.nearest_neighbor import nearest_neighbor
from tsp.tsplib import load_tsplib
from tsp.two_opt import two_opt, two_opt_with_stats


def improving_move_exists(points, tour):
    """검증용: 짧아지는 2-opt 교체가 하나라도 남아 있는지 모든 쌍을 직접 확인한다."""
    n = len(tour)
    base = tour_length(points, tour)
    for i, j in itertools.combinations(range(n), 2):
        if j - i < 2 or (i == 0 and j == n - 1):
            continue
        cand = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
        if tour_length(points, cand) < base - 1e-7:
            return True
    return False


class TwoOptTest(unittest.TestCase):
    def test_valid_tour_on_all_kinds(self):
        for kind in ("uniform", "clustered", "grid"):
            for n in (8, 14, 100, 200):
                with self.subTest(kind=kind, n=n):
                    tour = two_opt(make_instance(kind, n))
                    self.assertTrue(is_valid_tour(tour, n))

    def test_valid_tour_on_tsplib(self):
        inst = load_tsplib("eil51")
        self.assertTrue(is_valid_tour(two_opt(inst.points), len(inst.points)))

    def test_result_is_2opt_local_optimum(self):
        for kind in ("uniform", "clustered", "grid"):
            for n in (10, 40):
                with self.subTest(kind=kind, n=n):
                    pts = make_instance(kind, n)
                    self.assertFalse(improving_move_exists(pts, two_opt(pts)))

    def test_not_longer_than_initial_nn(self):
        for kind in ("uniform", "clustered", "grid"):
            for n in (14, 100, 300):
                with self.subTest(kind=kind, n=n):
                    pts = make_instance(kind, n)
                    nn = tour_length(pts, nearest_neighbor(pts))
                    self.assertLessEqual(tour_length(pts, two_opt(pts)), nn + 1e-9)

    def test_not_shorter_than_optimum(self):
        for kind in ("uniform", "clustered", "grid"):
            for n in (8, 10, 12):
                for index in range(3):
                    with self.subTest(kind=kind, n=n, index=index):
                        pts = make_instance(kind, n, index)
                        self.assertGreaterEqual(tour_length(pts, two_opt(pts)),
                                                held_karp_solve(pts)[1] - 1e-9)

    def test_uncrosses_square(self):
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        tour, stats = two_opt_with_stats(pts, initial=[0, 2, 1, 3])  # 대각선 두 개가 교차
        self.assertAlmostEqual(tour_length(pts, tour), 4)
        self.assertEqual(stats["improvements"], 1)
        self.assertEqual(stats["passes"], 2)  # 바꾼 바퀴 + 더 없음을 확인한 바퀴

    def test_initial_tour_is_not_modified(self):
        pts = make_instance("uniform", 30)
        initial = list(range(30))
        two_opt(pts, initial=initial)
        self.assertEqual(initial, list(range(30)))

    def test_already_optimal_initial_stays(self):
        pts = make_instance("uniform", 9)
        opt_tour, _ = held_karp_solve(pts)
        tour, stats = two_opt_with_stats(pts, initial=opt_tour)
        self.assertEqual(tour, opt_tour)
        self.assertEqual(stats["improvements"], 0)
        self.assertEqual(stats["passes"], 1)

    def test_deterministic(self):
        pts = make_instance("clustered", 150)
        self.assertEqual(two_opt(pts), two_opt(pts))

    def test_distance_count(self):
        # 최근접 이웃 n(n−1)/2 + 간선 길이 배열 n + 검사한 쌍마다 2
        for n in (5, 50, 120):
            with self.subTest(n=n):
                pts = make_instance("uniform", n)
                d = CountingDistance()
                _, stats = two_opt_with_stats(pts, d)
                self.assertEqual(d.count, n * (n - 1) // 2 + n + 2 * stats["checks"])
                d2 = CountingDistance()
                _, stats2 = two_opt_with_stats(pts, d2, initial=list(range(n)))
                self.assertEqual(d2.count, n + 2 * stats2["checks"])

    def test_checks_per_pass(self):
        # 한 바퀴에 검사하는 쌍은 n(n−3)/2개 (서로 이웃하지 않는 간선 쌍의 수)
        n = 60
        _, stats = two_opt_with_stats(make_instance("uniform", n))
        self.assertEqual(stats["checks"], stats["passes"] * n * (n - 3) // 2)

    def test_tiny_inputs(self):
        self.assertEqual(two_opt([]), [])
        self.assertEqual(two_opt([(1, 1)]), [0])
        self.assertEqual(two_opt([(0, 0), (1, 0)]), [0, 1])
        self.assertTrue(is_valid_tour(two_opt([(0, 0), (1, 0), (0, 1)]), 3))

    def test_duplicate_points(self):
        pts = [(0, 0), (2, 0), (0, 0), (2, 0), (1, 5)]
        tour = two_opt(pts)
        self.assertTrue(is_valid_tour(tour, 5))

    def test_works_with_tsplib_distance(self):
        pts = load_tsplib("eil51").points
        nn = nearest_neighbor(pts, tsplib_distance)
        tour = two_opt(pts, tsplib_distance)
        self.assertTrue(is_valid_tour(tour, len(pts)))
        self.assertLessEqual(tour_length(pts, tour, tsplib_distance), tour_length(pts, nn, tsplib_distance))


if __name__ == "__main__":
    unittest.main()
