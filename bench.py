"""실험 실행 스크립트.

    py bench.py smoke      작은 설정으로 러너가 동작하는지 확인 → results/smoke.csv
    py bench.py small      실험 A: n = 8~14, Held-Karp 최적값 대비 → results/small.csv
    py bench.py large      실험 B: n = 100~2000, MST 대비 비율·격자 초과율 → results/large.csv (약 10분)

실험마다 원자료(<name>.csv)와 요약(<name>_summary.csv)을 저장한다.
n이 3개 이상인 실험은 log-log 기울기(<name>_slopes.csv)도 저장한다.
실험 C·D는 이후 이슈에서 이 스크립트에 추가한다.
모든 실험은 results/environment.json에 측정 환경을 함께 남긴다.
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

from tsp.bench import measure_many, write_csv, write_environment
from tsp.common import tour_length
from tsp.held_karp import held_karp
from tsp.instances import grid_optimum, make_instance
from tsp.mst_approx import mst_length

RESULTS = Path(__file__).resolve().parent / "results"

FIELDS = [
    "experiment", "kind", "n", "instance", "algorithm",
    "length", "optimum", "excess_pct", "mst_length", "mst_ratio",
    "time_ms", "time_min_ms", "dist_calls", "passes", "improvements", "checks", "reps",
]

APPROX = ("nn", "mst", "two_opt")


def run_cases(experiment: str, kinds, sizes, instances: int, reps: int, algorithms=APPROX,
              verbose: bool = True) -> list[dict]:
    """(유형, n, 인스턴스)마다 algorithms를 측정해 CSV 행 목록을 만든다.

    품질 기준값:
        - 최적 길이: algorithms에 held_karp가 있으면 그 결과, 없으면 n ≤ 14일 때 Held-Karp를 따로 계산.
          격자는 항상 n이고, Held-Karp 결과가 n과 다르면 오류로 멈춘다.
        - MST 길이: 항상 계산 (MST 대비 비율용)
    """
    rows = []
    for kind in kinds:
        for n in sizes:
            for index in range(instances):
                points = make_instance(kind, n, index)
                results = measure_many(algorithms, points, reps=reps)
                optimum = results["held_karp"]["length"] if "held_karp" in results else None
                if kind == "grid":
                    if optimum is not None and abs(optimum - n) > 1e-6:
                        raise AssertionError(f"격자 n={n}의 Held-Karp 결과 {optimum}가 n과 다릅니다")
                    optimum = grid_optimum(n)
                elif optimum is None and n <= 14:
                    optimum = tour_length(points, held_karp(points))
                mst = mst_length(points)
                for algo, r in results.items():
                    row = {"experiment": experiment, "kind": kind, "n": n, "instance": index,
                           "algorithm": algo, "optimum": optimum, "mst_length": mst,
                           **{k: v for k, v in r.items() if k != "tour"}}
                    if optimum:
                        row["excess_pct"] = (r["length"] / optimum - 1) * 100
                    row["mst_ratio"] = r["length"] / mst
                    rows.append(row)
                    if verbose:
                        excess = f"{row['excess_pct']:6.2f}%" if optimum else "      -"
                        print(f"{experiment} {kind:9s} n={n:5d} #{index} {algo:9s} "
                              f"time={r['time_ms']:9.2f} ms calls={r['dist_calls']:>10,} "
                              f"excess={excess} ratio={row['mst_ratio']:.3f}", flush=True)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """(유형, n, 알고리즘)별 평균·최대 초과율, 최적을 찾은 횟수, 평균 MST 비율, 시간·거리 계산 중앙값."""
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["n"], r["algorithm"]), []).append(r)
    out = []
    for (kind, n, algo), rs in groups.items():
        s = {"kind": kind, "n": n, "algorithm": algo, "instances": len(rs),
             "mean_mst_ratio": statistics.mean(r["mst_ratio"] for r in rs),
             "median_time_ms": statistics.median(r["time_ms"] for r in rs),
             "median_dist_calls": statistics.median(r["dist_calls"] for r in rs)}
        ex = [r["excess_pct"] for r in rs if r.get("excess_pct") is not None]
        if ex:
            s["mean_excess_pct"] = statistics.mean(ex)
            s["max_excess_pct"] = max(ex)
            s["optimal_hits"] = sum(1 for e in ex if e < 1e-7)
        out.append(s)
    return out


SUMMARY_FIELDS = ["kind", "n", "algorithm", "instances", "mean_excess_pct", "max_excess_pct",
                  "optimal_hits", "mean_mst_ratio", "median_time_ms", "median_dist_calls"]


def print_summary(summary: list[dict]) -> None:
    print(f"{'kind':9s} {'n':>5s} {'algorithm':9s} {'mean ex%':>9s} {'max ex%':>8s} {'opt':>4s} "
          f"{'mst ratio':>9s} {'time ms':>10s} {'calls':>12s}")
    for s in summary:
        mean_ex = f"{s['mean_excess_pct']:9.2f}" if "mean_excess_pct" in s else f"{'-':>9s}"
        max_ex = f"{s['max_excess_pct']:8.2f}" if "max_excess_pct" in s else f"{'-':>8s}"
        hits = f"{s['optimal_hits']:>2d}/{s['instances']:<1d}" if "optimal_hits" in s else f"{'-':>4s}"
        print(f"{s['kind']:9s} {s['n']:5d} {s['algorithm']:9s} {mean_ex} {max_ex} {hits:>4s} "
              f"{s['mean_mst_ratio']:9.3f} {s['median_time_ms']:10.2f} {s['median_dist_calls']:12,.0f}")


def loglog_slopes(summary: list[dict]) -> list[dict]:
    """(유형, 알고리즘)별로 log(시간)·log(거리 계산)을 log(n)에 최소제곱으로 맞춘 기울기.

    n이 3개 이상인 조합만 계산한다. 이론 차수(O(n²)이면 2)와 비교하는 용도이다.
    """
    groups = {}
    for s in summary:
        groups.setdefault((s["kind"], s["algorithm"]), []).append(s)
    out = []
    for (kind, algo), ss in groups.items():
        if len(ss) < 3:
            continue
        xs = [math.log(s["n"]) for s in ss]
        row = {"kind": kind, "algorithm": algo, "n_min": min(s["n"] for s in ss),
               "n_max": max(s["n"] for s in ss), "points": len(ss)}
        for key, col in (("median_time_ms", "time_slope"), ("median_dist_calls", "calls_slope")):
            row[col] = _slope(xs, [math.log(s[key]) for s in ss])
        out.append(row)
    return out


def _slope(xs, ys) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


SLOPE_FIELDS = ["kind", "algorithm", "n_min", "n_max", "points", "time_slope", "calls_slope"]


def save(rows: list[dict], name: str, verbose: bool = True) -> Path:
    """results/<name>.csv(원자료)와 results/<name>_summary.csv(요약)를 저장한다.

    n이 3개 이상이면 log-log 기울기도 results/<name>_slopes.csv로 저장한다.
    """
    path = RESULTS / f"{name}.csv"
    write_csv(rows, path, FIELDS)
    summary = summarize(rows)
    write_csv(summary, RESULTS / f"{name}_summary.csv", SUMMARY_FIELDS)
    slopes = loglog_slopes(summary)
    if slopes:
        write_csv(slopes, RESULTS / f"{name}_slopes.csv", SLOPE_FIELDS)
    if verbose:
        print_summary(summary)
        for s in slopes:
            print(f"slope {s['kind']:9s} {s['algorithm']:9s} n={s['n_min']}~{s['n_max']}: "
                  f"time {s['time_slope']:.2f}, calls {s['calls_slope']:.2f}")
    return path


def smoke(reps: int) -> Path:
    rows = run_cases("smoke", ("uniform", "clustered", "grid"), (12, 100), instances=2, reps=reps)
    return save(rows, "smoke")


def small(reps: int, sizes=(8, 10, 12, 14), instances: int = 10, verbose: bool = True) -> Path:
    """실험 A: 작은 n에서 세 근사 해법을 Held-Karp 최적값과 비교한다 (Held-Karp 시간도 함께 잰다)."""
    rows = run_cases("small", ("uniform", "clustered", "grid"), sizes, instances, reps,
                     algorithms=("held_karp",) + APPROX, verbose=verbose)
    return save(rows, "small", verbose)


LARGE_SIZES = (100, 200, 500, 1000, 2000)
TWO_OPT_MAX_N = 1000  # 명세 5장: 2-opt는 우선 n ≤ 1000


def large_rows(reps: int, sizes=LARGE_SIZES, instances: int = 10, two_opt_max: int = TWO_OPT_MAX_N,
               verbose: bool = True) -> list[dict]:
    """실험 B의 원자료 행. 무작위·군집형은 MST 대비 비율, 격자는 최적값(n) 대비 초과율로 본다."""
    rows = []
    for n in sizes:
        algorithms = APPROX if n <= two_opt_max else tuple(a for a in APPROX if a != "two_opt")
        rows += run_cases("large", ("uniform", "clustered", "grid"), (n,), instances, reps,
                          algorithms=algorithms, verbose=verbose)
    return rows


def large(reps: int) -> Path:
    """실험 B: n = 100~2000에서 세 방식 비교 (2-opt는 n ≤ 1000)."""
    return save(large_rows(reps), "large")


# 이름 → (함수, 기본 반복 횟수)
EXPERIMENTS = {"smoke": (smoke, 3), "small": (small, 5), "large": (large, 5)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TSP 근사 해법 비교 실험")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--reps", type=int, default=None,
                        help="시간 측정 반복 횟수 (중앙값 사용, 기본값은 실험마다 다름)")
    args = parser.parse_args(argv)
    func, default_reps = EXPERIMENTS[args.experiment]
    write_environment(RESULTS / "environment.json")
    path = func(args.reps or default_reps)
    print(f"저장: {path.relative_to(RESULTS.parent).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
