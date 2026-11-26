"""실험 실행 스크립트.

    py bench.py smoke      작은 설정으로 러너가 동작하는지 확인 → results/smoke.csv

실험 A·B·C·D는 이후 이슈에서 이 스크립트에 추가한다.
모든 실험은 results/environment.json에 측정 환경을 함께 남긴다.
"""

import argparse
import sys
from pathlib import Path

from tsp.bench import measure, write_csv, write_environment
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


def reference_values(kind: str, points) -> dict:
    """품질 기준값: 최적 길이(알 수 있을 때)와 MST 길이."""
    ref = {"mst_length": mst_length(points), "optimum": None}
    if kind == "grid":
        ref["optimum"] = grid_optimum(len(points))
    elif len(points) <= 14:
        ref["optimum"] = tour_length(points, held_karp(points))
    return ref


def run_cases(experiment: str, kinds, sizes, instances: int, reps: int, algorithms=APPROX,
              verbose: bool = True) -> list[dict]:
    rows = []
    for kind in kinds:
        for n in sizes:
            for index in range(instances):
                points = make_instance(kind, n, index)
                ref = reference_values(kind, points)
                for algo in algorithms:
                    r = measure(algo, points, reps=reps)
                    row = {"experiment": experiment, "kind": kind, "n": n, "instance": index,
                           "algorithm": algo, **ref, **{k: v for k, v in r.items() if k != "tour"}}
                    if ref["optimum"]:
                        row["excess_pct"] = (r["length"] / ref["optimum"] - 1) * 100
                    row["mst_ratio"] = r["length"] / ref["mst_length"]
                    rows.append(row)
                    if verbose:
                        print(f"{experiment} {kind:9s} n={n:5d} #{index} {algo:8s} "
                              f"time={r['time_ms']:9.2f} ms calls={r['dist_calls']:>10,} "
                              f"ratio={row['mst_ratio']:.3f}", flush=True)
    return rows


def smoke(reps: int) -> Path:
    rows = run_cases("smoke", ("uniform", "clustered", "grid"), (12, 100), instances=2, reps=reps)
    path = RESULTS / "smoke.csv"
    write_csv(rows, path, FIELDS)
    return path


EXPERIMENTS = {"smoke": smoke}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TSP 근사 해법 비교 실험")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--reps", type=int, default=3, help="시간 측정 반복 횟수 (중앙값 사용)")
    args = parser.parse_args(argv)
    write_environment(RESULTS / "environment.json")
    path = EXPERIMENTS[args.experiment](args.reps)
    print(f"저장: {path.relative_to(RESULTS.parent).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
