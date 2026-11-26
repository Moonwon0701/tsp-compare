import tempfile
import unittest
from pathlib import Path

from tsp.common import is_valid_tour, tour_length, tsplib_distance
from tsp.tsplib import (
    INSTANCES,
    OPTIMA,
    load_optimal_tour,
    load_tsplib,
    parse_tour,
    parse_tsplib,
)

# 명세 3장의 표: 이름 → (n, 최적값)
SPEC_TABLE = {
    "eil51": (51, 426),
    "berlin52": (52, 7542),
    "kroA100": (100, 21282),
    "ch150": (150, 6528),
    "a280": (280, 2579),
}

SMALL = """NAME : tiny
TYPE : TSP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0 0
2 3 0
3 3 4
EOF
"""


class BundledInstancesTest(unittest.TestCase):
    def test_table_matches_spec(self):
        self.assertEqual(INSTANCES, tuple(SPEC_TABLE))
        self.assertEqual(OPTIMA, {name: opt for name, (_, opt) in SPEC_TABLE.items()})

    def test_point_count_matches_dimension(self):
        for name, (n, opt) in SPEC_TABLE.items():
            with self.subTest(name=name):
                inst = load_tsplib(name)
                self.assertEqual(len(inst.points), n)
                self.assertEqual(inst.optimum, opt)
                self.assertTrue(all(isinstance(v, float) for p in inst.points for v in p))

    def test_known_coordinates(self):
        eil51 = load_tsplib("eil51").points
        self.assertEqual(eil51[0], (37.0, 52.0))
        self.assertEqual(eil51[-1], (30.0, 40.0))
        self.assertEqual(load_tsplib("berlin52").points[0], (565.0, 575.0))
        self.assertEqual(load_tsplib("ch150").points[0], (37.4393516691, 541.2090699418))

    def test_any_tour_is_not_shorter_than_optimum(self):
        # 파일 순서대로 도는 경로도 순회이므로 최적값보다 짧을 수 없다
        for name in INSTANCES:
            with self.subTest(name=name):
                inst = load_tsplib(name)
                identity = list(range(len(inst.points)))
                self.assertGreaterEqual(tour_length(inst.points, identity, tsplib_distance),
                                        inst.optimum)

    def test_unknown_instance(self):
        with self.assertRaises(FileNotFoundError):
            load_tsplib("eil76")


class ParseTest(unittest.TestCase):
    def test_parses_header_and_points(self):
        header, points = parse_tsplib(SMALL)
        self.assertEqual(header["NAME"], "tiny")
        self.assertEqual(header["DIMENSION"], "3")
        self.assertEqual(points, [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)])

    def test_spacing_variants_crlf_and_missing_eof(self):
        text = SMALL.replace("DIMENSION : 3", "DIMENSION:3").replace("EOF\n", "")
        text = text.replace("1 0 0", "  1   0   0").replace("\n", "\r\n")
        _, points = parse_tsplib(text)
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0], (0.0, 0.0))

    def test_rejects_other_weight_types(self):
        with self.assertRaises(ValueError):
            parse_tsplib(SMALL.replace("EUC_2D", "GEO"))

    def test_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            parse_tsplib(SMALL.replace("DIMENSION : 3", "DIMENSION : 4"))

    def test_rejects_out_of_order_nodes(self):
        with self.assertRaises(ValueError):
            parse_tsplib(SMALL.replace("2 3 0", "5 3 0"))

    def test_rejects_malformed_coordinate_line(self):
        with self.assertRaises(ValueError):
            parse_tsplib(SMALL.replace("2 3 0", "2 3"))

    def test_rejects_name_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "other.tsp").write_text(SMALL, encoding="ascii")
            with self.assertRaises(ValueError):
                load_tsplib("other", data_dir=Path(tmp))
            Path(tmp, "tiny.tsp").write_text(SMALL, encoding="ascii")
            inst = load_tsplib("tiny", data_dir=Path(tmp))
            self.assertIsNone(inst.optimum)


class OptimalTourTest(unittest.TestCase):
    """TSPLIB가 공개한 최적 경로로 반올림 규칙과 최적값 표를 함께 검증한다."""

    def test_optimal_tour_is_valid(self):
        for name in INSTANCES:
            with self.subTest(name=name):
                n = len(load_tsplib(name).points)
                self.assertTrue(is_valid_tour(load_optimal_tour(name), n))

    def test_optimal_tour_length_equals_known_optimum(self):
        for name in INSTANCES:
            with self.subTest(name=name):
                inst = load_tsplib(name)
                length = tour_length(inst.points, load_optimal_tour(name), tsplib_distance)
                self.assertEqual(length, inst.optimum)

    def test_parse_tour_variants(self):
        text = "NAME : t\nTYPE : TOUR\nTOUR_SECTION\n1 3\n2\n-1\nEOF\n"
        self.assertEqual(parse_tour(text), [0, 2, 1])
        self.assertEqual(parse_tour(text.replace("-1\nEOF\n", "")), [0, 2, 1])  # 끝 표시 없음
        self.assertEqual(parse_tour(text.replace("\n", "\r\n")), [0, 2, 1])

    def test_parse_tour_requires_section(self):
        with self.assertRaises(ValueError):
            parse_tour("NAME : t\nTYPE : TOUR\n1\n2\n-1\n")

    def test_missing_tour_file(self):
        with self.assertRaises(FileNotFoundError):
            load_optimal_tour("eil76")


if __name__ == "__main__":
    unittest.main()
