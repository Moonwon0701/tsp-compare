"""results/*.csv에서 보고서용 그래프(results/*.png)를 그린다.

    py plot.py            모든 그래프
    py plot.py small      실험 A 그래프만 (이름: small, large, tsplib, start)

색은 알고리즘마다 고정한다 (최근접 이웃 = 파랑, MST 2-근사 = 주황, 2-opt = 청록).
이 세 색은 색각 이상 시뮬레이션까지 포함한 팔레트 검증(모든 쌍)을 통과했다.
글꼴은 Pretendard를 쓰고, 설치되어 있지 않으면 맑은 고딕 등 한글 글꼴로 대신한다.
청록은 밝은 배경 대비가 3:1보다 낮아서, 선 끝에 이름을 직접 적어 보완한다.
두 단위를 한 그래프의 두 y축에 겹치지 않고, 단위가 다르면 패널을 나눈다.
"""

import csv
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

RESULTS = Path(__file__).resolve().parent / "results"

ALGOS = ("nn", "mst", "two_opt")
COLORS = {"nn": "#2a78d6", "mst": "#eb6834", "two_opt": "#1baf7a"}
LABELS = {"nn": "최근접 이웃", "mst": "MST 2-근사", "two_opt": "2-opt"}
KIND_LABELS = {"uniform": "무작위", "clustered": "군집형", "grid": "격자"}
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

FONT_PREFERENCE = ("Pretendard", "Malgun Gothic", "AppleGothic", "NanumGothic")


def _register_user_fonts() -> None:
    """사용자 폴더에만 설치된 Pretendard는 matplotlib가 못 찾을 수 있어 파일을 직접 등록한다."""
    import os
    dirs = [Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
            Path.home() / "Library" / "Fonts", Path.home() / ".fonts", Path.home() / ".local" / "share" / "fonts"]
    for d in dirs:
        if d.is_dir():
            for f in list(d.glob("Pretendard*.otf")) + list(d.glob("Pretendard*.ttf")):
                font_manager.fontManager.addfont(str(f))


_register_user_fonts()
_FONTS = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in FONT_PREFERENCE if f in _FONTS), "sans-serif")
plt.rcParams.update({
    "font.family": FONT,
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_3, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlesize": 12, "axes.titleweight": "normal", "axes.titlecolor": INK,
    "font.size": 10, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


# ── 데이터 읽기 ─────────────────────────────────────────────────────────────

def load(name: str) -> list[dict]:
    with open(RESULTS / name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k == "n" and v not in ("", None):
                r[k] = int(v)
            elif k not in ("experiment", "kind", "instance", "algorithm", "group", "metric", "variant") and v != "":
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def pick(rows, **cond):
    return [r for r in rows if all(r[k] == v for k, v in cond.items())]


def save(fig, name: str) -> Path:
    path = RESULTS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: results/{name}")
    return path


def end_label(ax, x, y, text, color, dy=0):
    """선 끝에 이름을 직접 적는다 (글자는 잉크색, 앞의 짧은 선분이 색으로 정체를 알려 준다)."""
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center",
                fontsize=9, color=INK_2)


def legend_below(fig, axes_with_handles):
    handles, labels = axes_with_handles.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))


def log_x(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


# ── 실험 A ────────────────────────────────────────────────────────────────

def plot_small():
    """실험 A: 작은 n에서 최적값 대비 평균 초과율 (유형별 패널), Held-Karp 시간 증가."""
    summary = load("small_summary.csv")
    sizes = sorted({r["n"] for r in summary})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, kind in zip(axes, KIND_LABELS):
        for algo in ALGOS:
            rs = sorted(pick(summary, kind=kind, algorithm=algo), key=lambda r: r["n"])
            ys = [r["mean_excess_pct"] for r in rs]
            ax.plot(sizes, ys, color=COLORS[algo], marker="o", label=LABELS[algo])
        ax.set_title(KIND_LABELS[kind])
        ax.set_xticks(sizes)
        ax.set_xlabel("점 개수 n")
    axes[0].set_ylabel("최적 대비 평균 초과율 (%)")
    for algo in ALGOS:  # 마지막 패널에만 직접 이름
        r = pick(summary, kind="grid", algorithm=algo, n=sizes[-1])[0]
        end_label(axes[-1], sizes[-1], r["mean_excess_pct"], LABELS[algo], COLORS[algo])
    fig.suptitle("실험 A · 작은 n에서 Held-Karp 최적값 대비 초과율 (인스턴스 10개 평균)", x=0.01, ha="left", fontsize=12)
    legend_below(fig, axes[0])
    fig.tight_layout(rect=(0, 0.06, 0.97, 0.95))
    save(fig, "small_excess.png")

    # Held-Karp 시간: 측정값과 n²·2ⁿ 비례 곡선(n=14에 맞춤)
    hk = {}
    for r in pick(summary, algorithm="held_karp"):
        hk.setdefault(r["n"], []).append(r["median_time_ms"])
    ns = sorted(hk)
    times = [statistics.median(hk[n]) for n in ns]
    scale = times[-1] / (ns[-1] ** 2 * 2 ** ns[-1])
    ref_n = [n / 10 for n in range(ns[0] * 10, ns[-1] * 10 + 1)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ref_n, [scale * n ** 2 * 2 ** n for n in ref_n], color=INK_3, linewidth=1.2, linestyle="--")
    ax.plot(ns, times, color=INK, marker="o")
    for n, t in zip(ns, times):
        ax.annotate(f"{t:.1f} ms", (n, t), xytext=(8, -10), textcoords="offset points", ha="left", fontsize=9, color=INK_2)
    ref_at = 13
    ax.annotate("점선: n²·2ⁿ에 비례 (n=14에 맞춤)", (ref_at, scale * ref_at ** 2 * 2 ** ref_at), xytext=(-12, 10),
                textcoords="offset points", ha="right", fontsize=9, color=INK_3)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xlabel("점 개수 n")
    ax.set_ylabel("실행 시간 (ms, 로그 눈금)")
    ax.set_title("실험 A · Held-Karp 실행 시간 (세 유형 중앙값, 보조 지표)", loc="left")
    fig.tight_layout()
    save(fig, "small_held_karp_time.png")


# ── 실험 B ────────────────────────────────────────────────────────────────

def plot_large():
    """실험 B: 품질(무작위·군집형은 MST 대비 비율, 격자는 초과율)과 거리 계산·시간의 log-log 기울기."""
    summary = load("large_summary.csv")
    slopes = {(r["kind"], r["algorithm"]): r for r in load("large_slopes.csv")}
    sizes = sorted({r["n"] for r in summary})

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    panels = (("uniform", "mean_mst_ratio", "MST 대비 비율"), ("clustered", "mean_mst_ratio", "MST 대비 비율"),
              ("grid", "mean_excess_pct", "최적(n) 대비 초과율 (%)"))
    for ax, (kind, col, ylabel) in zip(axes, panels):
        for algo in ALGOS:
            rs = sorted(pick(summary, kind=kind, algorithm=algo), key=lambda r: r["n"])
            ax.plot([r["n"] for r in rs], [r[col] for r in rs], color=COLORS[algo], marker="o", label=LABELS[algo])
            end_label(ax, rs[-1]["n"], rs[-1][col], LABELS[algo], COLORS[algo])
        log_x(ax, sizes)
        ax.set_title(KIND_LABELS[kind])
        ax.set_xlabel("점 개수 n (로그 눈금)")
        ax.set_ylabel(ylabel)
        if col == "mean_mst_ratio":
            ax.set_ylim(1.0, 1.75)
        else:
            ax.set_ylim(0, 50)
    fig.suptitle("실험 B · 큰 n에서의 품질 (인스턴스 10개 평균, 2-opt는 n ≤ 1000)", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.06, "MST 대비 비율은 최적 대비 비율의 상한이라 실제 초과율보다 크게 나온다. 실험 A의 초과율과 직접 비교하지 않는다.",
             fontsize=8.5, color=INK_3)
    legend_below(fig, axes[0])
    fig.tight_layout(rect=(0, 0.06, 0.97, 0.95))
    save(fig, "large_quality.png")

    # 거리 계산 횟수(주 지표)와 실행 시간(보조 지표): 무작위 입력, 패널 둘
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, key, ylabel, title in (
            (axes[0], "median_dist_calls", "calls_slope", "거리 계산 횟수 (로그 눈금)", "거리 계산 횟수 (주 지표)"),
            (axes[1], "median_time_ms", "time_slope", "실행 시간 ms (로그 눈금)", "실행 시간 (보조 지표)")):
        for algo in ALGOS:
            rs = sorted(pick(summary, kind="uniform", algorithm=algo), key=lambda r: r["n"])
            slope = slopes[("uniform", algo)][key]
            ax.plot([r["n"] for r in rs], [r[col] for r in rs], color=COLORS[algo], marker="o", label=LABELS[algo])
            if col == "median_dist_calls" and algo == "mst":
                continue  # 최근접 이웃과 값이 정확히 같아 선이 겹치므로 이름표 하나로 합친다
            text = f"{LABELS[algo]} {slope:.2f}"
            if col == "median_dist_calls" and algo == "nn":
                text = f"최근접 이웃 = MST 2-근사 {slope:.2f}"
            dy = {"nn": -8, "mst": 8}.get(algo, 0) if col == "median_time_ms" else 0
            end_label(ax, rs[-1]["n"], rs[-1][col], text, COLORS[algo], dy)
        log_x(ax, sizes)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.set_xlabel("점 개수 n (로그 눈금)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    fig.suptitle("실험 B · 무작위 입력에서 n에 따른 비용 증가 (선 끝 숫자는 log-log 기울기, O(n²)이면 2)",
                 x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.06, "최근접 이웃과 MST 2-근사의 거리 계산 횟수는 n(n-1)/2로 정확히 같아 왼쪽 그래프에서 두 선이 겹친다.",
             fontsize=8.5, color=INK_3)
    legend_below(fig, axes[0])
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=6)
    save(fig, "large_cost_loglog.png")


# ── 실험 C + D (TSPLIB) ────────────────────────────────────────────────────

D_VARIANTS = (("nn", "nn", "최근접 이웃"), ("two_opt_start", "two_opt", "2-opt (시작점 변경)"),
              ("two_opt_rot", "two_opt", "2-opt (회전만)"))


def plot_tsplib():
    """실험 C(점 0 시작 결과)를 실험 D의 시작점별 분포 위에 표시한다."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = load("tsplib.csv")
    dsum = [r for r in load("start_sensitivity_summary.csv") if r["group"] == "tsplib"]
    names = [r["instance"] for r in pick(rows, algorithm="nn")]
    width = 0.2
    alpha = {"nn": 0.6, "two_opt_start": 0.6, "two_opt_rot": 0.3}

    fig, ax = plt.subplots(figsize=(11, 4.6))
    for j, name in enumerate(names):
        mst = pick(rows, instance=name, algorithm="mst")[0]
        ax.plot(j - 1.5 * width, mst["excess_pct"], marker="D", markersize=7, color=COLORS["mst"],
                markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="none")
        for k, (variant, algo, _label) in enumerate(D_VARIANTS):
            s = next(r for r in dsum if r["instance"] == name and r["variant"] == variant)
            x = j + (k - 0.5) * width
            ax.vlines(x, s["min"], s["max"], color=COLORS[algo], linewidth=7, alpha=alpha[variant])
            ax.hlines(s["median"], x - 0.07, x + 0.07, color=INK, linewidth=1.5)
            ax.plot(x, s["start0"], marker="o", markersize=7, color=COLORS[algo], markeredgecolor=SURFACE,
                    markeredgewidth=1.5, linestyle="none")
    handles = [
        Line2D([], [], marker="D", color=COLORS["mst"], markeredgecolor=SURFACE, linestyle="none", label="MST 2-근사 (점 0 시작)"),
        Patch(color=COLORS["nn"], alpha=alpha["nn"], label="최근접 이웃: 모든 시작점의 범위"),
        Patch(color=COLORS["two_opt"], alpha=alpha["two_opt_start"], label="2-opt (시작점 변경): 범위"),
        Patch(color=COLORS["two_opt"], alpha=alpha["two_opt_rot"], label="2-opt (회전만): 범위"),
        Line2D([], [], color=INK, linewidth=1.5, label="범위의 중앙값"),
        Line2D([], [], marker="o", color=INK_2, markeredgecolor=SURFACE, linestyle="none", label="점 0 시작 = 실험 C의 값"),
    ]
    ax.set_xticks(range(len(names)))
    sizes = {n: int(pick(rows, instance=n, algorithm="nn")[0]["n"]) for n in names}
    ax.set_xticklabels([n + "\n(n=" + str(sizes[n]) + ")" for n in names])
    ax.set_ylabel("최적 대비 초과율 (%)")
    ax.set_ylim(0, 50)
    ax.grid(axis="x", visible=False)
    ax.set_title("실험 C · D · TSPLIB 초과율: 점 0에서 시작한 결과(점)와 모든 시작점의 범위(막대)", loc="left")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    fig.tight_layout()
    save(fig, "tsplib_excess.png")


# ── 실험 D (n에 따른 퍼짐) ─────────────────────────────────────────────────

def plot_start():
    """실험 D: 격자 입력에서 시작점에 따른 초과율 범위 (격자는 모든 n에서 최적값을 안다).

    작은 n은 모든 시작점, 큰 n은 20개만 쓰므로 범위의 뜻이 달라 패널을 나눈다.
    """
    dsum = [r for r in load("start_sensitivity_summary.csv") if r["kind"] == "grid"]
    styles = {"nn": ("-", "o"), "two_opt_start": ("-", "o"), "two_opt_rot": ("--", "s")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, group, title in ((axes[0], "small", "작은 n · 모든 시작점"), (axes[1], "large", "큰 n · 시작점 20개 (점 0 포함)")):
        rows = [r for r in dsum if r["group"] == group]
        sizes = sorted({r["n"] for r in rows})
        for variant, algo, label in D_VARIANTS:
            pts = []
            for n in sizes:
                rs = [r["range"] for r in rows if r["variant"] == variant and r["n"] == n]
                if rs:
                    pts.append((n, statistics.mean(rs)))
            ls, mk = styles[variant]
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=COLORS[algo], linestyle=ls, marker=mk, label=label)
            if group == "large":
                if variant == "nn":
                    end_label(ax, pts[-1][0], pts[-1][1], label, COLORS[algo])
                else:
                    # 두 2-opt 선은 오른쪽 끝에서 거의 겹치고 최근접 이웃 이름표와도 가까워,
                    # 선이 벌어져 있는 왼쪽 첫 점에 위(시작점 변경)·아래(회전만)로 나눠 적는다
                    dy = 12 if variant == "two_opt_start" else -14
                    ax.annotate(label, (pts[0][0], pts[0][1]), xytext=(4, dy), textcoords="offset points",
                                va="center", ha="left", fontsize=9, color=INK_2)
        if group == "large":
            log_x(ax, sizes)
            ax.set_xlabel("점 개수 n (로그 눈금)")
        else:
            ax.set_xticks(sizes)
            ax.set_xlabel("점 개수 n")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("시작점에 따른 초과율 범위 (%p)")
    axes[0].set_ylim(bottom=0)
    fig.suptitle("실험 D · 격자 입력에서 시작점에 따른 퍼짐 (최대 - 최소, 인스턴스 10개 평균)", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.06, "작은 n의 2-opt는 2×k 격자(n=8, 10, 14)에서 시작점과 관계없이 최적을 찾아 범위가 0이다. 큰 n에서 2-opt는 n ≤ 1000.",
             fontsize=8.5, color=INK_3)
    legend_below(fig, axes[0])
    fig.tight_layout(rect=(0, 0.06, 0.93, 0.94))
    save(fig, "start_spread.png")


PLOTS = {"small": plot_small, "large": plot_large, "tsplib": plot_tsplib, "start": plot_start}


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(PLOTS)
    unknown = [n for n in names if n not in PLOTS]
    if unknown:
        print(f"알 수 없는 그래프: {', '.join(unknown)} (가능: {', '.join(PLOTS)})")
        return 2
    for name in names:
        PLOTS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
