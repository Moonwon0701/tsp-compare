import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
NEEDED = ["small_summary.csv", "large_summary.csv", "large_slopes.csv", "tsplib.csv",
          "start_sensitivity_summary.csv"]
EXPECTED = ["small_excess.png", "small_held_karp_time.png", "large_quality.png",
            "large_cost_loglog.png", "tsplib_excess.png", "start_spread.png"]


@unittest.skipUnless(all((ROOT / "results" / f).exists() for f in NEEDED), "실험 결과 CSV가 없다")
class PlotTest(unittest.TestCase):
    def test_all_plots_from_results(self):
        import plot
        with tempfile.TemporaryDirectory() as tmp:
            for f in NEEDED:
                shutil.copy(ROOT / "results" / f, Path(tmp) / f)
            with mock.patch.object(plot, "RESULTS", Path(tmp)), mock.patch("builtins.print"):
                self.assertEqual(plot.main([]), 0)
            for name in EXPECTED:
                with self.subTest(name=name):
                    path = Path(tmp) / name
                    self.assertTrue(path.exists())
                    self.assertGreater(path.stat().st_size, 10_000)

    def test_unknown_plot_name(self):
        import plot
        with mock.patch("builtins.print"):
            self.assertEqual(plot.main(["nope"]), 2)


if __name__ == "__main__":
    unittest.main()
