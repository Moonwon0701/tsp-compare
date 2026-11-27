"""실험 실행 스크립트.

    py bench.py smoke      작은 설정으로 러너가 동작하는지 확인 → results/smoke.csv
    py bench.py small      실험 A: n = 8~14, Held-Karp 최적값 대비 → results/small.csv
    py bench.py large      실험 B: n = 100~2000, MST 대비 비율·격자 초과율 → results/large.csv (약 6~10분)
    py bench.py tsplib     실험 C: TSPLIB 5개, 알려진 최적값 대비 → results/tsplib.csv
    py bench.py start      실험 D: 최근접 이웃·2-opt 시작점 민감도 → results/start_sensitivity.csv (병렬 실행)

실험마다 원자료(<name>.csv)와 요약(<name>_summary.csv)을 저장한다.
n이 3개 이상인 실험은 log-log 기울기(<name>_slopes.csv)도 저장한다.
모든 실험은 results/environment.json에 측정 환경을 함께 남긴다.
"""

import argparse
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tsp.bench import measure_many, write_csv, write_environment
from tsp.common import tour_length, tsplib_distance
from tsp.held_karp import held_karp
from tsp.instances import grid_optimum, make_instance
from tsp.mst_approx import mst_length
from tsp.start_sensitivity import run_instance, sample_starts, summarize_instance
from tsp.tsplib import INSTANCES as TSPLIB_INSTANCES, load_tsplib

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


def save(rows: list[dict], name: str, verbose: bool = True, slopes: bool = True) -> Path:
    """results/<name>.csv(원자료)와 results/<name>_summary.csv(요약)를 저장한다.

    slopes가 참이고 n이 3개 이상이면 log-log 기울기도 results/<name>_slopes.csv로 저장한다.
    """
    path = RESULTS / f"{name}.csv"
    write_csv(rows, path, FIELDS)
    summary = summarize(rows)
    write_csv(summary, RESULTS / f"{name}_summary.csv", SUMMARY_FIELDS)
    slopes = loglog_slopes(summary) if slopes else []
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
    # n = 8~14 시간은 0.01~0.1 ms라 잡음이 커서 기울기는 내지 않는다 (Held-Karp 증가는 그래프로 본다)
    return save(rows, "small", verbose, slopes=False)


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


def tsplib_rows(reps: int, names=TSPLIB_INSTANCES, verbose: bool = True) -> list[dict]:
    """실험 C의 원자료 행. TSPLIB 인스턴스마다 세 방식을 재고 알려진 최적값 대비 초과율을 낸다.

    TSPLIB 문제는 EUC_2D 거리(간선마다 반올림한 정수)로 정의되고 공식 최적값도 이 거리로 계산된 값이다.
    그래서 알고리즘이 쓰는 거리와 경로 길이를 재는 거리 모두 tsplib_distance로 한다 (명세 3장).
    instance 열에는 인스턴스 이름을 넣는다.
    """
    rows = []
    for name in names:
        inst = load_tsplib(name)
        points, n = inst.points, len(inst.points)
        results = measure_many(APPROX, points, reps=reps, dist=tsplib_distance)
        mst = mst_length(points, tsplib_distance)
        for algo, r in results.items():
            row = {"experiment": "tsplib", "kind": "tsplib", "n": n, "instance": name,
                   "algorithm": algo, "optimum": inst.optimum, "mst_length": mst,
                   **{k: v for k, v in r.items() if k != "tour"},
                   "excess_pct": (r["length"] / inst.optimum - 1) * 100,
                   "mst_ratio": r["length"] / mst}
            rows.append(row)
            if verbose:
                print(f"tsplib {name:9s} n={n:4d} {algo:8s} length={r['length']:7d} "
                      f"opt={inst.optimum:6d} excess={row['excess_pct']:6.2f}% "
                      f"calls={r['dist_calls']:>9,} time={r['time_ms']:8.2f} ms", flush=True)
    return rows


def tsplib(reps: int) -> Path:
    """실험 C: TSPLIB 5개 인스턴스에서 알려진 최적값 대비 초과율."""
    return save(tsplib_rows(reps), "tsplib", slopes=False)


START_FIELDS = ["group", "kind", "n", "instance", "metric", "variant", "start", "length", "quality"]
START_SUMMARY_FIELDS = ["group", "kind", "n", "instance", "metric", "variant", "starts", "min", "median",
                        "max", "range", "start0", "start0_rank", "corr_nn_two_opt"]
START_SAMPLE = 20  # 큰 n에서 쓰는 시작점 수 (점 0 포함)


def start_tasks(small_sizes=(8, 10, 12, 14), large_sizes=LARGE_SIZES, instances: int = 10,
                tsplib_names=TSPLIB_INSTANCES) -> list[tuple]:
    """실험 D 작업 목록: (group, kind, n, instance)."""
    tasks = [("small", kind, n, i) for kind in ("uniform", "clustered", "grid")
             for n in small_sizes for i in range(instances)]
    tasks += [("tsplib", "tsplib", None, name) for name in tsplib_names]
    tasks += [("large", kind, n, i) for kind in ("uniform", "clustered", "grid")
              for n in large_sizes for i in range(instances)]
    return tasks


def start_task(task: tuple) -> tuple[list[dict], list[dict]]:
    """실험 D 작업 하나: (원자료 행, 요약 행). 여러 프로세스에서 부를 수 있게 최상위 함수로 둔다."""
    group, kind, n, instance = task
    dist, optimum, mst, with_two_opt = None, None, None, True
    if group == "tsplib":
        inst = load_tsplib(instance)
        points, n, optimum, dist = inst.points, len(inst.points), inst.optimum, tsplib_distance
        starts = list(range(n))
    else:
        points = make_instance(kind, n, instance)
        if kind == "grid":
            optimum = grid_optimum(n)
        elif n <= 14:
            optimum = tour_length(points, held_karp(points))
        else:
            mst = mst_length(points)
        starts = sample_starts(n, START_SAMPLE if group == "large" else n, f"starts-{kind}-{n}-{instance}")
        with_two_opt = n <= TWO_OPT_MAX_N
    kwargs = {"dist": dist} if dist else {}
    rows = run_instance(points, starts, optimum=optimum, mst=mst, with_two_opt=with_two_opt, **kwargs)
    meta = {"group": group, "kind": kind, "n": n, "instance": instance,
            "metric": "excess_pct" if optimum else "mst_ratio"}
    return [{**meta, **r} for r in rows], [{**meta, **s} for s in summarize_instance(rows)]


def start_rows(tasks, workers: int | None = None, verbose: bool = True) -> tuple[list[dict], list[dict]]:
    """작업을 실행해 (원자료, 인스턴스별 요약)을 모은다. 시간을 재지 않으므로 병렬로 돌린다."""
    raw, summary = [], []
    if workers == 1:
        results = map(start_task, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 2))
        results = pool.map(start_task, tasks)
    try:
        for done, (r, s) in enumerate(results, 1):
            raw += r
            summary += s
            if verbose and (done % 20 == 0 or done == len(tasks)):
                print(f"start: {done}/{len(tasks)} 작업 완료", flush=True)
    finally:
        if workers != 1:
            pool.shutdown()
    return raw, summary


def print_start_summary(summary: list[dict]) -> None:
    """(group, kind, n, variant)별 평균 중앙값·평균 범위, TSPLIB는 인스턴스별로 출력."""
    groups = {}
    for s in summary:
        key = (s["group"], s["kind"], s["n"], s["variant"]) if s["group"] != "tsplib" else             (s["group"], s["instance"], s["n"], s["variant"])
        groups.setdefault(key, []).append(s)
    print(f"{'group':7s} {'kind':9s} {'n':>5s} {'variant':14s} {'median':>8s} {'range':>8s} {'start0 rank':>12s}")
    for (group, kind, n, variant), ss in groups.items():
        unit = "%" if ss[0]["metric"] == "excess_pct" else "x"
        rank = statistics.median(s["start0_rank"] for s in ss)
        print(f"{group:7s} {kind:9s} {n:5d} {variant:14s} {statistics.mean(s['median'] for s in ss):7.2f}{unit} "
              f"{statistics.mean(s['range'] for s in ss):7.2f}{unit} {rank:6.1f}/{ss[0]['starts']}")


def start(reps: int) -> Path:
    """실험 D: 최근접 이웃·2-opt의 시작점 민감도. reps는 쓰지 않는다 (시간을 재지 않음)."""
    raw, summary = start_rows(start_tasks())
    path = RESULTS / "start_sensitivity.csv"
    write_csv(raw, path, START_FIELDS)
    write_csv(summary, RESULTS / "start_sensitivity_summary.csv", START_SUMMARY_FIELDS)
    print_start_summary(summary)
    return path


# 이름 → (함수, 기본 반복 횟수)
EXPERIMENTS = {"smoke": (smoke, 3), "small": (small, 5), "large": (large, 5), "tsplib": (tsplib, 5),
               "start": (start, 1)}


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
