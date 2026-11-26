import math
import unittest

from tsp.common import (
    CountingDistance,
    euclidean,
    is_valid_tour,
    tour_length,
    tour_problems,
    tsplib_distance,
    validate_tour,
)

SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


class EuclideanTest(unittest.TestCase):
    def test_345_triangle(self):
        self.assertEqual(euclidean((0, 0), (3, 4)), 5)

    def test_symmetric_and_zero(self):
        p, q = (1.5, -2.0), (4.0, 7.25)
        self.assertEqual(euclidean(p, q), euclidean(q, p))
        self.assertEqual(euclidean(p, p), 0)


class TsplibDistanceTest(unittest.TestCase):
    def test_rounds_to_nearest_integer(self):
        self.assertEqual(tsplib_distance((0, 0), (1, 1)), 1)  # 1.414 → 1
        self.assertEqual(tsplib_distance((0, 0), (1, 2)), 2)  # 2.236 → 2
        self.assertEqual(tsplib_distance((0, 0), (2, 2)), 3)  # 2.828 → 3

    def test_half_rounds_up(self):
        self.assertEqual(tsplib_distance((0, 0), (0, 2.5)), 3)

    def test_returns_int(self):
        self.assertIsInstance(tsplib_distance((0, 0), (3, 4)), int)


class CountingDistanceTest(unittest.TestCase):
    def test_counts_calls_and_keeps_value(self):
        d = CountingDistance()
        self.assertEqual(d((0, 0), (3, 4)), 5)
        d((0, 0), (1, 0))
        self.assertEqual(d.count, 2)
        d.reset()
        self.assertEqual(d.count, 0)

    def test_wraps_other_distance(self):
        d = CountingDistance(tsplib_distance)
        self.assertEqual(d((0, 0), (1, 1)), 1)
        self.assertEqual(d.count, 1)

    def test_tour_length_uses_n_distance_calls(self):
        d = CountingDistance()
        tour_length(SQUARE, [0, 1, 2, 3], d)
        self.assertEqual(d.count, 4)


class TourLengthTest(unittest.TestCase):
    def test_unit_square_perimeter(self):
        self.assertAlmostEqual(tour_length(SQUARE, [0, 1, 2, 3]), 4)

    def test_crossing_tour_is_longer(self):
        # 0→2→1→3: 대각선 두 번(√2)과 변 두 번(1)
        self.assertAlmostEqual(tour_length(SQUARE, [0, 2, 1, 3]), 2 + 2 * math.sqrt(2))

    def test_rotation_and_reversal_do_not_change_length(self):
        base = tour_length(SQUARE, [0, 1, 2, 3])
        self.assertAlmostEqual(tour_length(SQUARE, [2, 3, 0, 1]), base)
        self.assertAlmostEqual(tour_length(SQUARE, [3, 2, 1, 0]), base)

    def test_includes_return_edge(self):
        pts = [(0, 0), (3, 0), (3, 4)]
        self.assertAlmostEqual(tour_length(pts, [0, 1, 2]), 3 + 4 + 5)

    def test_tiny_inputs(self):
        self.assertEqual(tour_length([], []), 0)
        self.assertEqual(tour_length([(5, 5)], [0]), 0)
        self.assertAlmostEqual(tour_length([(0, 0), (3, 4)], [0, 1]), 10)

    def test_duplicate_points(self):
        pts = [(0, 0), (0, 0), (1, 0)]
        self.assertAlmostEqual(tour_length(pts, [0, 1, 2]), 2)

    def test_with_tsplib_distance_is_integer(self):
        pts = [(0, 0), (1, 1), (2, 0)]  # 각 간선 1.414 → 1, 1.414 → 1, 2 → 2
        length = tour_length(pts, [0, 1, 2], tsplib_distance)
        self.assertEqual(length, 4)
        self.assertIsInstance(length, int)


class TourValidationTest(unittest.TestCase):
    def test_valid_permutations(self):
        self.assertTrue(is_valid_tour([0, 1, 2, 3], 4))
        self.assertTrue(is_valid_tour((3, 1, 0, 2), 4))
        self.assertTrue(is_valid_tour([0], 1))
        self.assertTrue(is_valid_tour([], 0))
        validate_tour([2, 0, 1], 3)  # 예외가 나지 않아야 한다

    def test_missing_node(self):
        self.assertFalse(is_valid_tour([0, 1, 2], 4))
        self.assertTrue(any("방문하지 않은" in p for p in tour_problems([0, 1, 2], 4)))

    def test_duplicate_node(self):
        self.assertFalse(is_valid_tour([0, 1, 1, 3], 4))
        self.assertTrue(any("중복" in p for p in tour_problems([0, 1, 1, 3], 4)))

    def test_out_of_range(self):
        self.assertFalse(is_valid_tour([0, 1, 2, 4], 4))
        self.assertFalse(is_valid_tour([-1, 1, 2, 3], 4))

    def test_wrong_length(self):
        self.assertFalse(is_valid_tour([0, 1, 2, 3, 0], 4))

    def test_non_integer(self):
        self.assertFalse(is_valid_tour([0, 1.0, 2, 3], 4))
        self.assertFalse(is_valid_tour([0, True, 2, 3], 4))

    def test_validate_raises_with_reason(self):
        with self.assertRaises(ValueError) as ctx:
            validate_tour([0, 0, 1], 3)
        self.assertIn("중복", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
