from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures" / "close_distance"

BASE_RAW = ROOT / "data" / "base_geometry" / "results_raw.csv"
MW_FINAL = ROOT / "data" / "measure_weighted" / "measure_weighted_grand_mean_final_combined.csv"
MW_RAW = ROOT / "data" / "measure_weighted" / "measure_weighted_grand_mean_raw_combined.csv"

CASE_COLORS = {
    "paper_poly3": "#3B6EA8",
    "paper_poly4": "#D9822B",
    "smooth_nonlinear": "#2F8F5B",
    "smooth_nonlinear_rand": "#8C5FBF",
}
STRATEGY_ORDER = ["random", "dopt", "latin", "measure_weighted"]
STRATEGY_LABELS = {
    "random": "Random",
    "dopt": "D-optimal",
    "latin": "Latin hypercube",
    "measure_weighted": "Measure-weighted",
}
STRATEGY_COLORS = {
    "random": "#6C6C6C",
    "dopt": "#D55E00",
    "latin": "#009E73",
    "measure_weighted": "#0072B2",
}


def sem(values):
    x = pd.Series(values).dropna()
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def clean_base_geometry(df):
    df = df.copy()
    group_median = df.groupby(["case", "activation", "hidden"])["NN_mse_vs_y"].transform("median")
    keep = df["NN_mse_vs_y"] <= 10.0 * group_median.clip(lower=1e-12)
    return df[keep].copy()


def save_close_distance_performance():
    df = clean_base_geometry(pd.read_csv(BASE_RAW))
    df = df.dropna(subset=["shape_NN_FPR_in", "abs_rmse_gap", "FPR_mse_vs_NN"])
    close_threshold = float(df["shape_NN_FPR_in"].quantile(0.25))
    df["distance_regime"] = np.where(
        df["shape_NN_FPR_in"] <= close_threshold,
        "Closest NN-FPR quartile",
        "Other runs",
    )
    close = df[df["distance_regime"] == "Closest NN-FPR quartile"]
    other = df[df["distance_regime"] == "Other runs"]
    rho, p_value = spearmanr(df["shape_NN_FPR_in"], df["abs_rmse_gap"])
    p_text = "p < 1e-300" if p_value == 0 else f"p = {p_value:.1e}"

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))

    ax = axes[0]
    ax.scatter(
        other["shape_NN_FPR_in"],
        other["abs_rmse_gap"],
        s=18,
        color="#B7B7B7",
        alpha=0.34,
        edgecolors="none",
        label="Other runs",
    )
    for case, sub in close.groupby("case"):
        ax.scatter(
            sub["shape_NN_FPR_in"],
            sub["abs_rmse_gap"],
            s=28,
            color=CASE_COLORS.get(case, "#333333"),
            alpha=0.78,
            edgecolors="white",
            linewidths=0.25,
            label=case,
        )
    ax.axvline(close_threshold, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("In-domain NN-FPR shape distance")
    ax.set_ylabel("Absolute RMSE gap between NN and FPR")
    ax.set_title("Small NN-FPR distance corresponds to small performance gap")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    text = (
        f"close threshold = {close_threshold:.2e}\n"
        f"Spearman rho = {rho:.2f}, {p_text}\n"
        f"median gap, close = {close['abs_rmse_gap'].median():.2e}\n"
        f"median gap, other = {other['abs_rmse_gap'].median():.2e}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#D0D0D0", alpha=0.88),
    )

    ax = axes[1]
    groups = [close["abs_rmse_gap"], other["abs_rmse_gap"]]
    box = ax.boxplot(
        groups,
        tick_labels=["Closest\nquartile", "Other\nruns"],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], ["#3B6EA8", "#B7B7B7"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_yscale("log")
    ax.set_ylabel("Absolute RMSE gap, log scale")
    ax.set_title("Performance gaps are compressed in the close regime")
    ax.grid(True, axis="y", alpha=0.25)
    medians = [g.median() for g in groups]
    for x, med in zip([1, 2], medians):
        ax.text(x, med * 1.22, f"median\n{med:.2e}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("FPR and NN are performance-close when their response surfaces are distance-close", fontsize=14)
    fig.tight_layout()
    out = OUT_DIR / "fpr_nn_close_distance_performance.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out, close_threshold, close, other


def save_close_distance_design_optimization():
    final = pd.read_csv(MW_FINAL)
    raw = pd.read_csv(MW_RAW)
    close_threshold = float(final["fpr_distance"].quantile(0.50))
    key_cols = ["p_dim", "case", "data_seed", "init_seed"]
    close_keys = (
        final[final["fpr_distance"] <= close_threshold][key_cols]
        .drop_duplicates()
        .assign(close_regime=True)
    )
    final_close = final.merge(close_keys, on=key_cols, how="inner")
    raw_close = raw.merge(close_keys, on=key_cols, how="inner")

    bar_stats = (
        final_close.groupby(["strategy", "strategy_label"], as_index=False)
        .agg(
            mean_gain=("gain_vs_random_pct", "mean"),
            se_gain=("gain_vs_random_pct", sem),
            positive_rate=("gain_vs_random_pct", lambda x: float((x > 0).mean())),
            runs=("gain_vs_random_pct", "size"),
        )
    )
    bar_stats["strategy"] = pd.Categorical(bar_stats["strategy"], STRATEGY_ORDER, ordered=True)
    bar_stats = bar_stats.sort_values("strategy")

    random_ref = raw_close[raw_close["strategy"] == "random"][
        key_cols + ["round", "test_mse"]
    ].rename(columns={"test_mse": "random_mse_same_round"})
    learning = raw_close.merge(random_ref, on=key_cols + ["round"], how="left")
    learning["gain_vs_random_round_pct"] = (
        (learning["random_mse_same_round"] - learning["test_mse"])
        / learning["random_mse_same_round"]
        * 100.0
    )
    curve_stats = (
        learning.groupby(["strategy", "strategy_label", "round"], as_index=False)
        .agg(mean_gain=("gain_vs_random_round_pct", "mean"), se_gain=("gain_vs_random_round_pct", sem))
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))

    ax = axes[0]
    x = np.arange(len(bar_stats))
    colors = [STRATEGY_COLORS[str(s)] for s in bar_stats["strategy"]]
    ax.bar(x, bar_stats["mean_gain"], color=colors, alpha=0.88)
    ax.errorbar(
        x,
        bar_stats["mean_gain"],
        yerr=1.96 * bar_stats["se_gain"],
        fmt="none",
        ecolor="black",
        linewidth=1.0,
        capsize=4,
    )
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[str(s)] for s in bar_stats["strategy"]], rotation=18, ha="right")
    ax.set_ylabel("Final gain vs random (%)")
    ax.set_title("Final optimization in the NN-FPR-close regime")
    ax.grid(True, axis="y", alpha=0.25)
    for i, row in enumerate(bar_stats.itertuples(index=False)):
        ax.text(
            i,
            row.mean_gain + (1.1 if row.mean_gain >= 0 else -1.1),
            f"{row.mean_gain:.1f}%\nwin {100 * row.positive_rate:.0f}%",
            ha="center",
            va="bottom" if row.mean_gain >= 0 else "top",
            fontsize=8,
        )

    ax = axes[1]
    for strategy in STRATEGY_ORDER:
        sub = curve_stats[curve_stats["strategy"] == strategy].sort_values("round")
        if sub.empty:
            continue
        x_round = sub["round"].to_numpy(dtype=float)
        y = sub["mean_gain"].to_numpy(dtype=float)
        ci = 1.96 * sub["se_gain"].to_numpy(dtype=float)
        ax.plot(
            x_round,
            y,
            marker="o",
            linewidth=2.0,
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
        )
        ax.fill_between(
            x_round,
            y - ci,
            y + ci,
            color=STRATEGY_COLORS[strategy],
            alpha=0.12,
            linewidth=0,
        )
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Batch round")
    ax.set_ylabel("Gain vs same-round random (%)")
    ax.set_title("Optimization trajectory across batch updates")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    title = (
        "Design strategies when the pilot FPR-NN distance is small "
        f"(fpr_distance <= {close_threshold:.2e})"
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out = OUT_DIR / "fpr_nn_close_distance_design_optimization.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return out, close_threshold, bar_stats


def save_close_distance_design_by_case():
    final = pd.read_csv(MW_FINAL)
    case_distance = (
        final.groupby(["p_dim", "case", "case_label"], as_index=False)["fpr_distance"]
        .mean()
        .sort_values("fpr_distance")
    )
    close_case_threshold = float(case_distance["fpr_distance"].median())
    close_cases = case_distance[case_distance["fpr_distance"] <= close_case_threshold]
    plot_df = final.merge(
        close_cases[["p_dim", "case"]],
        on=["p_dim", "case"],
        how="inner",
    )
    stats = (
        plot_df.groupby(["p_dim", "case_label", "strategy", "strategy_label"], as_index=False)
        .agg(mean_gain=("gain_vs_random_pct", "mean"), se_gain=("gain_vs_random_pct", sem))
    )
    stats["panel"] = stats["case_label"] + "\np=" + stats["p_dim"].astype(str)
    panel_order = (
        close_cases.assign(panel=close_cases["case_label"] + "\np=" + close_cases["p_dim"].astype(str))
        .sort_values(["fpr_distance", "p_dim", "case_label"])["panel"]
        .tolist()
    )

    n = len(panel_order)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 3.65 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, panel in zip(axes, panel_order):
        sub = stats[stats["panel"] == panel].copy()
        sub["strategy"] = pd.Categorical(sub["strategy"], STRATEGY_ORDER, ordered=True)
        sub = sub.sort_values("strategy")
        x = np.arange(len(sub))
        colors = [STRATEGY_COLORS[str(s)] for s in sub["strategy"]]
        ax.bar(x, sub["mean_gain"], color=colors, alpha=0.88)
        ax.errorbar(
            x,
            sub["mean_gain"],
            yerr=1.96 * sub["se_gain"],
            fmt="none",
            ecolor="black",
            linewidth=0.9,
            capsize=3,
        )
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_title(panel)
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[str(s)] for s in sub["strategy"]], rotation=24, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel("Final gain vs random (%)")
    fig.suptitle(
        "Design-strategy gains by case in the closest half of mean FPR-NN distances",
        fontsize=14,
    )
    fig.tight_layout()
    out = OUT_DIR / "fpr_nn_close_distance_design_by_case.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out, close_case_threshold, close_cases


def write_summary(perf_threshold, close, other, design_threshold, bar_stats,
                  case_threshold, close_cases):
    lines = [
        "# Close-distance figure summary",
        "",
        "## FPR/NN distance close -> performance close",
        "",
        f"- Source data: `{BASE_RAW.relative_to(ROOT)}`.",
        f"- Close regime: bottom quartile of `shape_NN_FPR_in`, threshold `{perf_threshold:.6e}`.",
        f"- Rows after NN non-convergence cleaning: `{len(close) + len(other)}`.",
        f"- Close-regime rows: `{len(close)}`.",
        f"- Median `abs_rmse_gap` in close regime: `{close['abs_rmse_gap'].median():.6e}`.",
        f"- Median `abs_rmse_gap` outside close regime: `{other['abs_rmse_gap'].median():.6e}`.",
        "",
        "## Design optimization under close FPR/NN distance",
        "",
        f"- Source data: `{MW_FINAL.relative_to(ROOT)}` and `{MW_RAW.relative_to(ROOT)}`.",
        f"- Close regime: lower half of `fpr_distance`, threshold `{design_threshold:.6e}`.",
        f"- Case-level close regime for the by-case figure: lower half of mean `fpr_distance`, threshold `{case_threshold:.6e}`.",
        "",
        "| Strategy | Mean final gain vs random | 95% CI half-width | Positive rate | Runs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in bar_stats.itertuples(index=False):
        lines.append(
            f"| {row.strategy_label} | {row.mean_gain:.3f}% | "
            f"{1.96 * row.se_gain:.3f}% | {100 * row.positive_rate:.1f}% | {int(row.runs)} |"
        )
    lines.extend([
        "",
        "Case/p panels used in the by-case design figure:",
        "",
        "| p | Case | Mean FPR distance |",
        "|---:|---|---:|",
    ])
    for row in close_cases.sort_values("fpr_distance").itertuples(index=False):
        lines.append(f"| {int(row.p_dim)} | {row.case_label} | {row.fpr_distance:.6e} |")
    out = OUT_DIR / "close_distance_figure_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perf_path, perf_threshold, close, other = save_close_distance_performance()
    design_path, design_threshold, bar_stats = save_close_distance_design_optimization()
    case_path, case_threshold, close_cases = save_close_distance_design_by_case()
    summary_path = write_summary(
        perf_threshold, close, other, design_threshold, bar_stats,
        case_threshold, close_cases)
    print(perf_path)
    print(design_path)
    print(case_path)
    print(summary_path)


if __name__ == "__main__":
    main()
