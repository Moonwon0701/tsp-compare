import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

from tsp.common import is_valid_tour, tour_length
from tsp.instances import (
    AREA,
    KINDS,
    clustered_points,
    default_clusters,
    grid_optimum,
    grid_shape,
    make_instance,
    seed_for,
)

ROOT = Path(__file__).resolve().parent.parent

# 명세 3장의 격자 크기 표
SPEC_GRID_SHAPES = {
    8: (2, 4), 10: (2, 5), 12: (3, 4), 14: (2, 7),
    100: (10, 10), 200: (10, 20), 500: (20, 25), 1000: (25, 40), 2000: (40, 50),
}
EXPERIMENT_SIZES = [8, 10, 12, 14, 100, 200, 500, 1000, 2000]


def snake_cycle(rows, cols):
    """rows가 짝수인 rows×cols 격자를 길이 1 간선만으로 도는 순회 (좌표 (r, c))."""
    cycle = [(r, 0) for r in range(rows)]
    for i, r in enumerate(range(rows - 1, -1, -1)):
        cs = range(1, cols) if i % 2 == 0 else range(cols - 1, 0, -1)
        cycle += [(r, c) for c in cs]
    return cycle


class ReproducibilityTest(unittest.TestCase):
    def test_same_arguments_give_same_points(self):
        for kind in KINDS:
            for n in (8, 100):
                with self.subTest(kind=kind, n=n):
                    self.assertEqual(make_instance(kind, n, 3), make_instance(kind, n, 3))

    def test_different_index_gives_different_points(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.assertNotEqual(make_instance(kind, 100, 0), make_instance(kind, 100, 1))

    def test_seed_is_readable_string(self):
        self.assertEqual(seed_for("uniform", 100, 3), "uniform-100-3")

    def test_same_points_across_processes(self):
        # 문자열 seed가 PYTHONHASHSEED에 영향을 받지 않는지 별도 프로세스로 확인
        code = "from tsp.instances import make_instance; print(make_instance('clustered', 20, 4)[:3])"
        outputs = []
        for hash_seed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                    capture_output=True, text=True, check=True)
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0].strip(), str(make_instance("clustered", 20, 4)[:3]))


class ShapeTest(unittest.TestCase):
    def test_point_count_and_type(self):
        for kind in KINDS:
            for n in EXPERIMENT_SIZES:
                with self.subTest(kind=kind, n=n):
                    pts = make_instance(kind, n)
                    self.assertEqual(len(pts), n)
                    self.assertTrue(all(isinstance(p, tuple) and len(p) == 2 for p in pts))
                    self.assertTrue(all(isinstance(v, float) for p in pts for v in p))

    def test_unknown_kind_and_negative_n(self):
        with self.assertRaises(ValueError):
            make_instance("ring", 10)
        with self.assertRaises(ValueError):
            make_instance("uniform", -1)

    def test_empty_instance(self):
        self.assertEqual(make_instance("uniform", 0), [])
        self.assertEqual(make_instance("clustered", 0), [])


class UniformTest(unittest.TestCase):
    def test_inside_area(self):
        pts = make_instance("uniform", 2000)
        self.assertTrue(all(0 <= x <= AREA and 0 <= y <= AREA for x, y in pts))

    def test_spreads_over_area(self):
        # 2000개면 네 사분면에 모두 점이 꽤 들어가야 한다
        pts = make_instance("uniform", 2000)
        half = AREA / 2
        quadrants = [sum(1 for x, y in pts if (x < half) == qx and (y < half) == qy)
                     for qx in (True, False) for qy in (True, False)]
        self.assertTrue(all(q > 400 for q in quadrants), quadrants)


class ClusteredTest(unittest.TestCase):
    def test_default_cluster_count(self):
        self.assertEqual([default_clusters(n) for n in (8, 10, 12, 14, 20, 2000)],
                         [2, 2, 3, 3, 5, 5])

    def test_points_stay_near_single_center(self):
        rng = random.Random("test")
        pts = clustered_points(500, rng, k=1, sigma=10)
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        self.assertTrue(all(abs(x - cx) < 60 and abs(y - cy) < 60 for x, y in pts))

    def test_more_clumped_than_uniform(self):
        # 군집형은 가장 가까운 점까지의 평균 거리가 균일 분포보다 짧아야 한다
        def mean_nearest(pts):
            total = 0
            for i, (x, y) in enumerate(pts):
                total += min((x - a) ** 2 + (y - b) ** 2
                             for j, (a, b) in enumerate(pts) if j != i) ** 0.5
            return total / len(pts)

        self.assertLess(mean_nearest(make_instance("clustered", 300)),
                        mean_nearest(make_instance("uniform", 300)))


class GridTest(unittest.TestCase):
    def test_shapes_match_spec_table(self):
        for n, shape in SPEC_GRID_SHAPES.items():
            with self.subTest(n=n):
                self.assertEqual(grid_shape(n), shape)

    def test_impossible_sizes(self):
        for n in (1, 2, 3, 7, 9, 15, 25):  # 소수, 1×k, 홀수×홀수
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    grid_shape(n)
                with self.assertRaises(ValueError):
                    make_instance("grid", n)

    def test_points_are_full_lattice(self):
        for n in EXPERIMENT_SIZES:
            with self.subTest(n=n):
                m, k = grid_shape(n)
                expected = {(float(x), float(y)) for y in range(m) for x in range(k)}
                self.assertEqual(set(make_instance("grid", n)), expected)

    def test_order_is_shuffled(self):
        m, k = grid_shape(100)
        row_major = [(float(x), float(y)) for y in range(m) for x in range(k)]
        self.assertNotEqual(make_instance("grid", 100), row_major)

    def test_optimal_length_is_n(self):
        # 한 변이 짝수인 격자에는 길이 n인 순회가 실제로 존재한다
        for n in EXPERIMENT_SIZES:
            with self.subTest(n=n):
                m, k = grid_shape(n)
                pts = make_instance("grid", n)
                index = {p: i for i, p in enumerate(pts)}
                if m % 2 == 0:
                    cycle = [(float(c), float(r)) for r, c in snake_cycle(m, k)]
                else:
                    cycle = [(float(r), float(c)) for r, c in snake_cycle(k, m)]
                tour = [index[p] for p in cycle]
                self.assertTrue(is_valid_tour(tour, n))
                self.assertAlmostEqual(tour_length(pts, tour), n)
                self.assertEqual(grid_optimum(n), n)


if __name__ == "__main__":
    unittest.main()
