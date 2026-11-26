"""벤치마크 러너: 알고리즘 하나를 입력 하나에 돌려 시간·거리 계산 횟수·경로 길이를 잰다.

측정 규칙 (명세 4장 "거리 계산 횟수의 정의"):
    - 실행 시간: 세지 않는 거리 함수로 reps번 돌려 중앙값을 쓴다. 측정 중에는 가비지 컬렉터를 끈다.
    - 거리 계산 횟수: 시간을 재지 않는 별도 실행 한 번에서 CountingDistance로 센다.
      알고리즘이 결정적이므로 한 번이면 충분하고, 세는 오버헤드가 시간에 섞이지 않는다.
    - 경로 길이: 시간 측정 구간 밖에서, 세지 않는 거리 함수로 계산한다.
"""

import csv
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

from .common import CountingDistance, euclidean, tour_length, validate_tour
from .held_karp import held_karp
from .mst_approx import mst_approx
from .nearest_neighbor import nearest_neighbor
from .two_opt import two_opt, two_opt_with_stats

# 이름 → (경로만 돌려주는 함수, 통계까지 돌려주는 함수 또는 None)
ALGORITHMS = {
    "nn": (nearest_neighbor, None),
    "mst": (mst_approx, None),
    "two_opt": (two_opt, two_opt_with_stats),
    "held_karp": (held_karp, None),
}

STAT_FIELDS = ("passes", "improvements", "checks")


def measure(name: str, points, reps: int = 5, dist=euclidean, length_dist=None, **options) -> dict:
    """알고리즘 name을 points에 돌려 측정 결과를 돌려준다.

    dist는 알고리즘이 쓰는 거리, length_dist는 결과 길이를 잴 거리(기본 dist와 같음)이다.
    options는 알고리즘에 그대로 넘긴다 (예: start, root, initial).

    반환: tour, length, time_ms(중앙값), time_min_ms, dist_calls, reps, 2-opt면 passes·improvements·checks
    """
    if reps < 1:
        raise ValueError("reps는 1 이상이어야 합니다")
    solve, solve_with_stats = ALGORITHMS[name]
    length_dist = length_dist or dist

    # 1) 거리 계산 횟수 (시간을 재지 않는 실행)
    counter = CountingDistance(dist)
    stats = {}
    if solve_with_stats is not None:
        tour, stats = solve_with_stats(points, counter, **options)
    else:
        tour = solve(points, counter, **options)
    validate_tour(tour, len(points))

    # 2) 실행 시간 (세지 않는 거리, reps번 중앙값)
    times = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            start = time.perf_counter()
            timed_tour = solve(points, dist, **options)
            times.append((time.perf_counter() - start) * 1000)
            if timed_tour != tour:
                raise AssertionError(f"{name}: 실행마다 경로가 다릅니다 (결정적이어야 함)")
    finally:
        if gc_was_enabled:
            gc.enable()

    # 3) 경로 길이 (측정 구간 밖)
    result = {
        "tour": tour,
        "length": tour_length(points, tour, length_dist),
        "time_ms": statistics.median(times),
        "time_min_ms": min(times),
        "dist_calls": counter.count,
        "reps": reps,
    }
    for field in STAT_FIELDS:
        if field in stats:
            result[field] = stats[field]
    return result


def environment() -> dict:
    """측정 환경 정보 (README·보고서에 적을 값).

    공개 저장소에 올라가므로 사용자 이름이 들어가는 경로(실행 파일 위치 등)는 넣지 않는다.
    """
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def write_environment(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    """rows를 CSV로 저장한다. 없는 열은 빈칸, fields에 없는 키는 무시한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in fields})


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return value
