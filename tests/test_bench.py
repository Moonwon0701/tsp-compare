import csv
import json
import tempfile
import unittest
from pathlib import Path

from tsp.bench import ALGORITHMS, environment, measure, write_csv, write_environment
from tsp.common import tour_length, tsplib_distance
from tsp.instances import make_instance
from tsp.nearest_neighbor import nearest_neighbor
from tsp.tsplib import load_tsplib


class MeasureTest(unittest.TestCase):
    def setUp(self):
        self.points = make_instance("uniform", 40)

    def test_fields_and_valid_values(self):
        for name in ("nn", "mst", "two_opt"):
            with self.subTest(name=name):
                r = measure(name, self.points, reps=2)
                self.assertEqual(sorted(r["tour"]), list(range(40)))
                self.assertAlmostEqual(r["length"], tour_length(self.points, r["tour"]))
                self.assertGreater(r["time_ms"], 0)
                self.assertLessEqual(r["time_min_ms"], r["time_ms"])
                self.assertEqual(r["reps"], 2)

    def test_dist_calls_exclude_length_computation(self):
        n = len(self.points)
        self.assertEqual(measure("nn", self.points, reps=1)["dist_calls"], n * (n - 1) // 2)
        self.assertEqual(measure("mst", self.points, reps=1)["dist_calls"], n * (n - 1) // 2)

    def test_two_opt_stats_and_calls(self):
        n = len(self.points)
        r = measure("two_opt", self.points, reps=1)
        for field in ("passes", "improvements", "checks"):
            self.assertIn(field, r)
        self.assertEqual(r["dist_calls"], n * (n - 1) // 2 + n + 2 * r["checks"])

    def test_options_are_forwarded(self):
        r = measure("nn", self.points, reps=1, start=7)
        self.assertEqual(r["tour"][0], 7)
        initial = nearest_neighbor(self.points, start=3)
        r = measure("two_opt", self.points, reps=1, initial=initial)
        self.assertEqual(r["dist_calls"], len(self.points) + 2 * r["checks"])  # 초기 경로 비용 없음

    def test_length_dist_for_tsplib(self):
        inst = load_tsplib("eil51")
        r = measure("nn", inst.points, reps=1, length_dist=tsplib_distance)
        self.assertIsInstance(r["length"], int)
        self.assertEqual(r["length"], tour_length(inst.points, r["tour"], tsplib_distance))

    def test_held_karp_available(self):
        pts = make_instance("uniform", 8)
        self.assertEqual(measure("held_karp", pts, reps=1)["dist_calls"], 8 * 7 // 2)

    def test_rejects_bad_reps_and_name(self):
        with self.assertRaises(ValueError):
            measure("nn", self.points, reps=0)
        with self.assertRaises(KeyError):
            measure("christofides", self.points)

    def test_registry(self):
        self.assertEqual(set(ALGORITHMS), {"nn", "mst", "two_opt", "held_karp"})


class OutputTest(unittest.TestCase):
    def test_write_csv_roundtrip(self):
        rows = [{"a": 1, "b": 0.5, "extra": "x"}, {"a": 2}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "out.csv"
            write_csv(rows, path, ["a", "b"])
            with open(path, encoding="utf-8") as f:
                read = list(csv.DictReader(f))
        self.assertEqual(read, [{"a": "1", "b": "0.500000"}, {"a": "2", "b": ""}])

    def test_environment(self):
        env = environment()
        for key in ("python", "os", "processor", "cpu_count"):
            self.assertIn(key, env)
        self.assertNotIn("executable", env)  # 사용자 이름이 담긴 경로는 넣지 않는다
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "env.json"
            write_environment(path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["python"], env["python"])


class SmokeRunTest(unittest.TestCase):
    def test_run_cases_rows(self):
        import bench
        rows = bench.run_cases("t", ("uniform", "grid"), (8,), instances=1, reps=1, verbose=False)
        self.assertEqual(len(rows), 2 * 3)
        for row in rows:
            self.assertGreaterEqual(row["excess_pct"], -1e-9)  # 작은 n·격자는 최적값이 있다
            self.assertGreaterEqual(row["mst_ratio"], 1 - 1e-9)


if __name__ == "__main__":
    unittest.main()
