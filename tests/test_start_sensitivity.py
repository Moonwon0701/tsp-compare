import unittest

from tsp.common import tour_length
from tsp.held_karp import held_karp_solve
from tsp.instances import make_instance
from tsp.nearest_neighbor import nearest_neighbor
from tsp.start_sensitivity import pearson, run_instance, sample_starts, summarize_instance
from tsp.two_opt import two_opt


class SampleStartsTest(unittest.TestCase):
    def test_all_points_when_small(self):
        self.assertEqual(sample_starts(12, 20, "s"), list(range(12)))

    def test_sample_includes_zero_and_is_deterministic(self):
        a = sample_starts(500, 20, "starts-uniform-500-0")
        self.assertEqual(len(a), 20)
        self.assertEqual(len(set(a)), 20)
        self.assertEqual(a[0], 0)
        self.assertTrue(all(0 <= s < 500 for s in a))
        self.assertEqual(a, sample_starts(500, 20, "starts-uniform-500-0"))
        self.assertNotEqual(a, sample_starts(500, 20, "starts-uniform-500-1"))


class RunInstanceTest(unittest.TestCase):
    def setUp(self):
        self.points = make_instance("uniform", 10, 3)
        _, self.opt = held_karp_solve(self.points)
        self.rows = run_instance(self.points, range(10), optimum=self.opt)

    def get(self, variant, start):
        return next(r for r in self.rows if r["variant"] == variant and r["start"] == start)

    def test_row_count_and_quality(self):
        self.assertEqual(len(self.rows), 3 * 10)
        for r in self.rows:
            self.assertGreaterEqual(r["quality"], -1e-7)
            self.assertAlmostEqual(r["quality"], (r["length"] / self.opt - 1) * 100)

    def test_variants_match_direct_calls(self):
        pts = self.points
        for s in (0, 4):
            nn = nearest_neighbor(pts, start=s)
            self.assertAlmostEqual(self.get("nn", s)["length"], tour_length(pts, nn))
            self.assertAlmostEqual(self.get("two_opt_start", s)["length"],
                                   tour_length(pts, two_opt(pts, initial=nn)))
            base = nearest_neighbor(pts)
            self.assertAlmostEqual(self.get("two_opt_rot", s)["length"],
                                   tour_length(pts, two_opt(pts, initial=base[s:] + base[:s])))

    def test_start_zero_matches_main_experiment(self):
        # 시작점 0이면 세 변형 모두 본 실험(A·B·C)의 설정과 같다
        main_two_opt = tour_length(self.points, two_opt(self.points))
        self.assertAlmostEqual(self.get("two_opt_start", 0)["length"], main_two_opt)
        self.assertAlmostEqual(self.get("two_opt_rot", 0)["length"], main_two_opt)

    def test_two_opt_not_worse_than_its_initial(self):
        for s in range(10):
            self.assertLessEqual(self.get("two_opt_start", s)["length"],
                                 self.get("nn", s)["length"] + 1e-9)

    def test_mst_ratio_and_no_two_opt(self):
        rows = run_instance(self.points, [0, 1], mst=1000.0, with_two_opt=False)
        self.assertEqual({r["variant"] for r in rows}, {"nn"})
        self.assertAlmostEqual(rows[0]["quality"], rows[0]["length"] / 1000.0)

    def test_needs_reference(self):
        with self.assertRaises(ValueError):
            run_instance(self.points, [0])


class SummaryTest(unittest.TestCase):
    def test_summary_values(self):
        rows = [{"variant": "nn", "start": s, "quality": q} for s, q in ((0, 5.0), (1, 2.0), (2, 9.0), (3, 5.0))]
        # two_opt_start = 0.5 × nn 이라 상관계수가 정확히 1
        rows += [{"variant": "two_opt_start", "start": s, "quality": q} for s, q in ((0, 2.5), (1, 1.0), (2, 4.5), (3, 2.5))]
        s = {x["variant"]: x for x in summarize_instance(rows)}
        self.assertEqual((s["nn"]["min"], s["nn"]["median"], s["nn"]["max"], s["nn"]["range"]), (2.0, 5.0, 9.0, 7.0))
        self.assertEqual(s["nn"]["start0_rank"], 2)  # 2.0 하나만 더 좋다 (동점은 같은 순위)
        self.assertAlmostEqual(s["two_opt_start"]["corr_nn_two_opt"], 1.0)
        self.assertIsNone(s["nn"]["corr_nn_two_opt"])

    def test_pearson_constant_is_none(self):
        self.assertIsNone(pearson([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(pearson([1, 2, 3], [3, 2, 1]), -1.0)


class BenchTaskTest(unittest.TestCase):
    def test_small_tsplib_large_tasks(self):
        import bench
        raw, summary = bench.start_rows([("small", "grid", 8, 0), ("tsplib", "tsplib", None, "eil51"),
                                         ("large", "uniform", 100, 0)], workers=1, verbose=False)
        by_group = {}
        for r in raw:
            by_group.setdefault(r["group"], []).append(r)
        self.assertEqual(len(by_group["small"]), 3 * 8)
        self.assertEqual(len(by_group["tsplib"]), 3 * 51)
        self.assertEqual(len(by_group["large"]), 3 * 20)
        self.assertTrue(all(r["metric"] == "excess_pct" for r in by_group["small"] + by_group["tsplib"]))
        self.assertTrue(all(r["metric"] == "mst_ratio" for r in by_group["large"]))
        self.assertTrue(all(isinstance(r["length"], int) for r in by_group["tsplib"]))  # 반올림 거리
        self.assertEqual(len(summary), 3 * 3)
        self.assertTrue(all(s["start0_rank"] >= 1 for s in summary))


if __name__ == "__main__":
    unittest.main()
