from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    ROOT
    / "data"
    / "close_distance_highdim_optimality"
    / "close_distance_highdim_optimality_v2_raw.csv"
)
OUT_DIR = ROOT / "figures" / "close_distance_highdim_optimality" / "presentation"


METHODS = [
    ("dopt", "D-opt", "Classical"),
    ("aopt", "A-opt", "Classical"),
    ("iopt", "I-opt", "Classical"),
    ("latin", "Latin", "Baseline"),
    ("dopt_err", "D + error reg.", "Proposed"),
    ("aopt_err", "A + error reg.", "Proposed"),
    ("iopt_err", "I + error reg.", "Proposed"),
]

PAIR_METHODS = [
    ("dopt", "dopt_err", "D-opt"),
    ("aopt", "aopt_err", "A-opt"),
    ("iopt", "iopt_err", "I-opt"),
]

GROUP_COLORS = {
    "Classical": "#C95C3F",
    "Baseline": "#7A7F86",
    "Proposed": "#16857B",
}

PAIR_COLORS = {
    "Classical": "#C95C3F",
    "Proposed": "#16857B",
}


def gain_col(method: str) -> str:
    return f"{method}_gain_vs_random_median_pct"


def load_used_results() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing result file: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    used = raw[raw["status"] == "used_close"].copy()
    if used.empty:
        raise ValueError("No used_close rows found in the v2 results.")
    return used


def nonidentity_summary(used: pd.DataFrame) -> pd.DataFrame:
    nonidentity = used[used["activation"] != "identity"].copy()
    rows = []
    for method, label, group in METHODS:
        values = nonidentity[gain_col(method)].dropna()
        rows.append(
            {
                "method": method,
                "label": label,
                "group": group,
                "reps": len(values),
                "mean_gain": float(values.mean()),
                "median_gain": float(values.median()),
                "win_rate": float((values > 0).mean() * 100.0),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def label_bars(ax, bars, fmt="{:.1f}%"):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        value = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2.0
        if value >= 0:
            y = value + 0.025 * span
            va = "bottom"
        else:
            y = value - 0.04 * span
            va = "top"
        ax.text(
            x,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=10,
            color="#202124",
            fontweight="bold" if value > 25 else "normal",
        )


def make_summary_slide(used: pd.DataFrame) -> Path:
    summary = nonidentity_summary(used)
    labels = summary["label"].tolist()
    colors = [GROUP_COLORS[group] for group in summary["group"]]
    x = np.arange(len(summary))

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Error-aware optimal design restores robust NN-to-FullPR transfer",
        fontsize=23,
        fontweight="bold",
        x=0.5,
        y=0.96,
    )
    fig.text(
        0.5,
        0.91,
        "Nonlinear activations only; p = 5, 10, 20; close-distance runs; gain vs same-budget random median",
        ha="center",
        fontsize=12,
        color="#5F6368",
    )

    ax = axes[0]
    bars = ax.bar(x, summary["median_gain"], color=colors, width=0.72)
    ax.axhline(0, color="#202124", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Median MSE gain (%)", fontsize=12)
    ax.set_title("Typical transfer performance", fontsize=15, fontweight="bold")
    ax.set_ylim(-25, 50)
    style_axis(ax)
    label_bars(ax, bars)
    ax.axvspan(3.5, 6.5, color="#16857B", alpha=0.08, zorder=0)
    ax.text(
        5.0,
        46,
        "Proposed error-regularized criteria",
        ha="center",
        va="top",
        color="#0D5F58",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[1]
    bars = ax.bar(x, summary["win_rate"], color=colors, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Win rate over random median (%)", fontsize=12)
    ax.set_title("Reliability across runs", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 116)
    style_axis(ax)
    label_bars(ax, bars)
    ax.axvspan(3.5, 6.5, color="#16857B", alpha=0.08, zorder=0)
    ax.text(
        5.0,
        112,
        "97-100% wins",
        ha="center",
        va="top",
        color="#0D5F58",
        fontsize=12,
        fontweight="bold",
    )

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=11)
        for color in [GROUP_COLORS["Classical"], GROUP_COLORS["Baseline"], GROUP_COLORS["Proposed"]]
    ]
    fig.legend(
        handles,
        ["Classical optimal criteria", "Latin baseline", "Our error-regularized criteria"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.20, wspace=0.23)

    out = OUT_DIR / "optimality_v2_advantage_summary.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def make_dimension_slide(used: pd.DataFrame) -> Path:
    nonidentity = used[used["activation"] != "identity"].copy()
    rows = []
    for p, sub in nonidentity.groupby("p"):
        for classic, proposed, label in PAIR_METHODS:
            rows.append(
                {
                    "p": int(p),
                    "criterion": label,
                    "classic": float(sub[gain_col(classic)].median()),
                    "proposed": float(sub[gain_col(proposed)].median()),
                }
            )
    stats = pd.DataFrame(rows)
    p_values = sorted(stats["p"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "The error term fixes the failure mode at every dimension tested",
        fontsize=23,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.91,
        "Median MSE gain on nonlinear activations; ordinary criteria versus error-regularized versions",
        ha="center",
        fontsize=12,
        color="#5F6368",
    )

    width = 0.34
    x = np.arange(len(p_values))
    for ax, (_, _, title) in zip(axes, PAIR_METHODS):
        sub = stats[stats["criterion"] == title].sort_values("p")
        classic = sub["classic"].to_numpy()
        proposed = sub["proposed"].to_numpy()
        b1 = ax.bar(
            x - width / 2,
            classic,
            width,
            color=PAIR_COLORS["Classical"],
            label="Ordinary",
        )
        b2 = ax.bar(
            x + width / 2,
            proposed,
            width,
            color=PAIR_COLORS["Proposed"],
            label="With error reg.",
        )
        ax.axhline(0, color="#202124", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"p={p}" for p in p_values])
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_ylim(-130, 60)
        style_axis(ax)
        label_bars(ax, b1)
        label_bars(ax, b2)
        for i, (old, new) in enumerate(zip(classic, proposed)):
            ax.annotate(
                "",
                xy=(i + width / 2, new),
                xytext=(i - width / 2, old),
                arrowprops=dict(arrowstyle="->", color="#5F6368", lw=1.4, alpha=0.75),
            )
    axes[0].set_ylabel("Median MSE gain (%)", fontsize=12)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.18, wspace=0.10)

    out = OUT_DIR / "optimality_v2_advantage_by_dimension.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def make_case_heatmap(used: pd.DataFrame) -> Path:
    nonidentity = used[used["activation"] != "identity"].copy()
    case_order = [
        "Quadratic",
        "Smooth",
        "Strong nonlinear",
        "Sparse cubic",
        "Sparse quartic",
        "Random nonlinear",
    ]
    records = []
    for case, sub in nonidentity.groupby("case_label"):
        if case not in case_order:
            continue
        ordinary = np.nanmedian(
            np.column_stack(
                [
                    sub[gain_col("dopt")],
                    sub[gain_col("aopt")],
                    sub[gain_col("iopt")],
                ]
            )
        )
        proposed = np.nanmedian(
            np.column_stack(
                [
                    sub[gain_col("dopt_err")],
                    sub[gain_col("aopt_err")],
                    sub[gain_col("iopt_err")],
                ]
            )
        )
        records.append({"case": case, "Ordinary D/A/I": ordinary, "Our error-reg. D/A/I": proposed})
    stats = pd.DataFrame(records).set_index("case").reindex(case_order)

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    fig.patch.set_facecolor("white")
    vals = stats.to_numpy(dtype=float)
    vmax = max(50.0, float(np.nanmax(np.abs(vals))))
    im = ax.imshow(vals, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(stats.shape[1]))
    ax.set_xticklabels(stats.columns, fontsize=12, fontweight="bold")
    ax.set_yticks(np.arange(stats.shape[0]))
    ax.set_yticklabels(stats.index, fontsize=12)
    ax.set_title(
        "Across problem families, the proposed criteria turn red cells blue",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            val = vals[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#202124",
                    fontweight="bold",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.04)
    cbar.set_label("Median MSE gain (%)", fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(
        0.5,
        0.04,
        "Nonlinear activations only; cases with no close-distance nonlinear runs are omitted.",
        ha="center",
        fontsize=10,
        color="#5F6368",
    )
    fig.subplots_adjust(left=0.24, right=0.90, top=0.82, bottom=0.14)

    out = OUT_DIR / "optimality_v2_advantage_by_case.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    used = load_used_results()
    paths = [
        make_summary_slide(used),
        make_dimension_slide(used),
        make_case_heatmap(used),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
