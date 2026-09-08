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
TABLE_DIR = ROOT / "data" / "close_distance_highdim_optimality" / "presentation"


ACTIVATION_COLORS = {
    "identity": "#7A7F86",
    "relu": "#2C7FB8",
    "sigmoid": "#7B3294",
    "softplus": "#008837",
    "tanh": "#D55E00",
}
CASE_COLORS = {
    "Quadratic": "#4C78A8",
    "Smooth": "#54A24B",
    "Strong nonlinear": "#B279A2",
    "Local interior": "#72B7B2",
    "High-frequency": "#E45756",
    "Sparse cubic": "#F58518",
    "Sparse quartic": "#9467BD",
    "Random nonlinear": "#8C564B",
}
METHOD_COLORS = {
    "D-opt": "#C95C3F",
    "D + error reg.": "#16857B",
    "Latin": "#7A7F86",
}
FAMILY_MARKERS = {
    "iterative_dopt": "o",
    "measure_weighted": "s",
    "gpu_geometry": "^",
}


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing v2 result file: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    base = raw.drop_duplicates(["activation", "p", "case", "data_seed", "init_seed"]).copy()
    close_base = base[base["status"] == "used_close"].copy()
    close_rows = raw[raw["status"] == "used_close"].copy()
    if close_base.empty or close_rows.empty:
        raise ValueError("No close-distance rows found.")
    close_base["nn_rmse"] = np.sqrt(close_base["oracle_mse_vs_y"])
    close_base["fpr_rmse"] = np.sqrt(close_base["full_fpr_mse_vs_y"])
    close_base["rmse_gap"] = (close_base["nn_rmse"] - close_base["fpr_rmse"]).abs()
    close_base["mimic_rmse"] = np.sqrt(close_base["full_fpr_mse_vs_nn"])
    return close_base, close_rows


def load_base_results() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing v2 result file: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    base = raw.drop_duplicates(["activation", "p", "case", "data_seed", "init_seed"]).copy()
    if base.empty:
        raise ValueError("No base rows found.")
    base["abs_mse_gap"] = (
        base["oracle_mse_vs_y"] - base["full_fpr_mse_vs_y"]
    ).abs()
    base["nn_rmse"] = np.sqrt(base["oracle_mse_vs_y"])
    base["fpr_rmse"] = np.sqrt(base["full_fpr_mse_vs_y"])
    base["rmse_gap"] = (base["nn_rmse"] - base["fpr_rmse"]).abs()
    base["is_close_run"] = base["status"] == "used_close"
    return base


def style_axis(ax, y_grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid:
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.7)
    else:
        ax.grid(color="#D8D8D8", linewidth=0.8, alpha=0.55)
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
            y = value - 0.035 * span
            va = "top"
        ax.text(
            x,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=10,
            color="#202124",
            fontweight="bold" if abs(value) >= 25 else "normal",
        )


def save_distance_relationship_table(base: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for (activation, p), sub in base.groupby(["activation", "p"]):
        x = sub["shape_nn_fpr"].clip(lower=1e-16)
        y = sub["abs_mse_gap"].clip(lower=1e-16)
        rows.append(
            {
                "activation": activation,
                "p": int(p),
                "runs": len(sub),
                "close_runs": int(sub["is_close_run"].sum()),
                "close_pass_rate_pct": sub["is_close_run"].mean() * 100.0,
                "spearman_distance_abs_mse_gap": x.corr(y, method="spearman"),
                "log_pearson_distance_abs_mse_gap": np.corrcoef(
                    np.log10(x), np.log10(y)
                )[0, 1],
                "median_shape_distance": x.median(),
                "median_abs_mse_gap": y.median(),
                "median_abs_rmse_gap": sub["rmse_gap"].median(),
            }
        )
    overall = {
        "activation": "all",
        "p": "all",
        "runs": len(base),
        "close_runs": int(base["is_close_run"].sum()),
        "close_pass_rate_pct": base["is_close_run"].mean() * 100.0,
        "spearman_distance_abs_mse_gap": base["shape_nn_fpr"]
        .clip(lower=1e-16)
        .corr(base["abs_mse_gap"].clip(lower=1e-16), method="spearman"),
        "log_pearson_distance_abs_mse_gap": np.corrcoef(
            np.log10(base["shape_nn_fpr"].clip(lower=1e-16)),
            np.log10(base["abs_mse_gap"].clip(lower=1e-16)),
        )[0, 1],
        "median_shape_distance": base["shape_nn_fpr"].median(),
        "median_abs_mse_gap": base["abs_mse_gap"].median(),
        "median_abs_rmse_gap": base["rmse_gap"].median(),
    }
    out = TABLE_DIR / "distance_abs_mse_relationship_by_activation_p.csv"
    pd.concat([pd.DataFrame([overall]), pd.DataFrame(rows)], ignore_index=True).to_csv(
        out, index=False, encoding="utf-8-sig"
    )
    return out


def make_distance_abs_mse_relationship_figure(base: pd.DataFrame) -> Path:
    p_values = sorted(base["p"].dropna().unique())
    activations = ["identity", "relu", "sigmoid", "softplus", "tanh"]
    x_col = "shape_nn_fpr"
    y_col = "abs_mse_gap"

    fig, axes = plt.subplots(
        len(activations),
        len(p_values),
        figsize=(16, 18),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "NN-FPR distance tracks absolute MSE gap across settings",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.958,
        "Each point is one data case / data seed / NN initialization; y = |NN test MSE - FullPR test MSE|",
        ha="center",
        fontsize=12,
        color="#5F6368",
    )

    xmin, xmax = 1e-16, max(0.50, float(base[x_col].max()) * 2.00)
    ymin, ymax = 1e-12, max(0.40, float(base[y_col].max()) * 2.00)
    threshold = float(base["close_threshold"].dropna().iloc[0])

    for i, activation in enumerate(activations):
        for j, p in enumerate(p_values):
            ax = axes[i, j]
            sub = base[(base["activation"] == activation) & (base["p"] == p)].copy()
            ax.axvspan(xmin, threshold, color="#16857B", alpha=0.06, linewidth=0)
            ax.axvline(threshold, color="#202124", linestyle="--", linewidth=0.9, alpha=0.75)
            for case_label, case_sub in sub.groupby("case_label"):
                x = case_sub[x_col].clip(lower=xmin)
                y = case_sub[y_col].clip(lower=ymin)
                ax.scatter(
                    x,
                    y,
                    s=34,
                    color=CASE_COLORS.get(case_label, "#444444"),
                    alpha=0.82 if activation != "identity" else 0.50,
                    edgecolors="none",
                )

            finite = sub[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
            finite = finite[(finite[x_col] > 0) & (finite[y_col] > 0)]
            rho = np.nan
            if len(finite) >= 3:
                rho = finite[x_col].corr(finite[y_col], method="spearman")
                lx = np.log10(finite[x_col].clip(lower=xmin))
                ly = np.log10(finite[y_col].clip(lower=ymin))
                slope, intercept = np.polyfit(lx, ly, 1)
                xp = np.geomspace(
                    max(xmin, float(finite[x_col].min()) * 0.90),
                    min(xmax, float(finite[x_col].max()) * 1.10),
                    80,
                )
                yp = 10 ** (intercept + slope * np.log10(xp))
                ax.plot(xp, yp, color="#202124", linewidth=1.4, alpha=0.80)
            ax.text(
                0.04,
                0.92,
                f"rho={rho:.2f}" if np.isfinite(rho) else "rho=n/a",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="#202124",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.grid(color="#D8D8D8", linewidth=0.65, alpha=0.55)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == 0:
                ax.set_title(f"p={int(p)}", fontsize=14, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{activation}\nAbsolute MSE gap", fontsize=11)
            if i == len(activations) - 1:
                ax.set_xlabel("NN-FPR shape distance", fontsize=11)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=8)
        for color in CASE_COLORS.values()
    ]
    fig.legend(
        handles,
        list(CASE_COLORS.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.075,
        0.936,
        "Shaded region: close-distance subset used in the design comparison",
        ha="left",
        fontsize=10,
        color="#0D5F58",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.925, bottom=0.095, hspace=0.22, wspace=0.08)

    out = OUT_DIR / "distance_abs_mse_relationship_by_p_activation_case.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def make_distance_abs_mse_relationship_compact(base: pd.DataFrame, include_identity: bool = True) -> Path:
    activation_order = ["identity", "relu", "sigmoid", "softplus", "tanh"]
    plot_base = base.copy()
    if not include_identity:
        plot_base = plot_base[plot_base["activation"] != "identity"].copy()
        activation_order = [a for a in activation_order if a != "identity"]
        if plot_base.empty:
            raise ValueError("No non-identity rows found.")

    p_values = sorted(plot_base["p"].dropna().unique())
    x_col = "shape_nn_fpr"
    y_col = "abs_mse_gap"
    if include_identity:
        xmin = 1e-16
        ymin = 1e-12
    else:
        xmin = max(1e-8, float(plot_base.loc[plot_base[x_col] > 0, x_col].min()) * 0.60)
        ymin = max(1e-8, float(plot_base.loc[plot_base[y_col] > 0, y_col].min()) * 0.60)
    xmax = max(0.50, float(plot_base[x_col].max()) * 2.00)
    ymax = max(0.40, float(plot_base[y_col].max()) * 2.00)
    threshold = float(plot_base["close_threshold"].dropna().iloc[0])

    fig, axes = plt.subplots(1, len(p_values), figsize=(16, 9), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    title = (
        "Smaller NN-FPR distance corresponds to smaller absolute MSE gap"
        if include_identity
        else "Distance predicts absolute MSE gap without identity controls"
    )
    subtitle = (
        "All dataset settings, activations, and seeds; x = shape distance, y = |NN test MSE - FullPR test MSE|"
        if include_identity
        else "Nonlinear activations only: relu, sigmoid, softplus, tanh; x = shape distance, y = |NN test MSE - FullPR test MSE|"
    )
    fig.suptitle(
        title,
        fontsize=22,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.91,
        subtitle,
        ha="center",
        fontsize=12,
        color="#5F6368",
    )

    for ax, p in zip(axes, p_values):
        sub = plot_base[plot_base["p"] == p].copy()
        ax.axvspan(xmin, threshold, color="#16857B", alpha=0.06, linewidth=0)
        ax.axvline(threshold, color="#202124", linestyle="--", linewidth=1.0, alpha=0.75)
        for activation in activation_order:
            act_sub = sub[sub["activation"] == activation]
            if act_sub.empty:
                continue
            for family, fam_sub in act_sub.groupby("dataset_family"):
                ax.scatter(
                    fam_sub[x_col].clip(lower=xmin),
                    fam_sub[y_col].clip(lower=ymin),
                    s=46 if activation != "identity" else 34,
                    marker=FAMILY_MARKERS.get(family, "o"),
                    color=ACTIVATION_COLORS.get(activation, "#444444"),
                    alpha=0.80 if activation != "identity" else 0.45,
                    edgecolors="none",
                )

        finite = sub[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        finite = finite[(finite[x_col] > 0) & (finite[y_col] > 0)]
        rho = finite[x_col].corr(finite[y_col], method="spearman")
        lx = np.log10(finite[x_col].clip(lower=xmin))
        ly = np.log10(finite[y_col].clip(lower=ymin))
        slope, intercept = np.polyfit(lx, ly, 1)
        xp = np.geomspace(max(xmin, float(finite[x_col].min()) * 0.90), min(xmax, float(finite[x_col].max()) * 1.10), 120)
        yp = 10 ** (intercept + slope * np.log10(xp))
        ax.plot(xp, yp, color="#202124", linewidth=1.6, alpha=0.82)
        ax.text(
            0.05,
            0.92,
            f"p={int(p)}\nrho={rho:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="#202124",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("NN-FPR shape distance", fontsize=11)
        ax.grid(color="#D8D8D8", linewidth=0.7, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Absolute MSE gap", fontsize=12)

    activation_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ACTIVATION_COLORS[a], markersize=8)
        for a in activation_order
    ]
    family_handles = [
        plt.Line2D([0], [0], marker=m, color="#202124", linestyle="none", markersize=8)
        for m in FAMILY_MARKERS.values()
    ]
    legend1 = fig.legend(
        activation_handles,
        activation_order,
        loc="lower center",
        bbox_to_anchor=(0.36, 0.045),
        ncol=len(activation_order),
        frameon=False,
        fontsize=10,
        title="Activation",
        title_fontsize=10,
    )
    fig.add_artist(legend1)
    fig.legend(
        family_handles,
        ["iterative", "measure-weighted", "geometry"],
        loc="lower center",
        bbox_to_anchor=(0.72, 0.045),
        ncol=3,
        frameon=False,
        fontsize=10,
        title="Dataset family",
        title_fontsize=10,
    )
    fig.text(
        0.075,
        0.865,
        "Shaded region: close-distance subset used in the design comparison",
        ha="left",
        fontsize=10,
        color="#0D5F58",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.18, wspace=0.10)

    suffix = "compact" if include_identity else "nonidentity_compact"
    out = OUT_DIR / f"distance_abs_mse_relationship_{suffix}.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def save_prediction_tables(close_base: pd.DataFrame) -> tuple[Path, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    by_activation = (
        close_base.groupby("activation")
        .agg(
            runs=("activation", "size"),
            median_shape_distance=("shape_nn_fpr", "median"),
            median_nn_rmse=("nn_rmse", "median"),
            median_fpr_rmse=("fpr_rmse", "median"),
            median_abs_rmse_gap=("rmse_gap", "median"),
            median_mimic_rmse=("mimic_rmse", "median"),
        )
        .reset_index()
    )
    by_family = (
        close_base.groupby("dataset_family")
        .agg(
            runs=("dataset_family", "size"),
            median_shape_distance=("shape_nn_fpr", "median"),
            median_nn_rmse=("nn_rmse", "median"),
            median_fpr_rmse=("fpr_rmse", "median"),
            median_abs_rmse_gap=("rmse_gap", "median"),
            median_mimic_rmse=("mimic_rmse", "median"),
        )
        .reset_index()
    )
    p1 = TABLE_DIR / "close_distance_prediction_by_activation.csv"
    p2 = TABLE_DIR / "close_distance_prediction_by_family.csv"
    by_activation.to_csv(p1, index=False, encoding="utf-8-sig")
    by_family.to_csv(p2, index=False, encoding="utf-8-sig")
    return p1, p2


def make_prediction_closeness_figure(close_base: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 9),
        gridspec_kw={"width_ratios": [0.95, 1.30, 0.95]},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "When NN and FPR are close, their predictive performance is comparable",
        fontsize=22,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.91,
        "Close-distance subset only: shape distance <= 0.005; y scaled to [-1, 1]",
        ha="center",
        fontsize=12,
        color="#5F6368",
    )

    ax = axes[0]
    activations = ["identity", "relu", "sigmoid", "softplus", "tanh"]
    rng = np.random.default_rng(2026)
    for i, activation in enumerate(activations):
        sub = close_base[close_base["activation"] == activation]
        if sub.empty:
            continue
        x = np.full(len(sub), i, dtype=float) + rng.uniform(-0.17, 0.17, size=len(sub))
        ax.scatter(
            x,
            sub["shape_nn_fpr"],
            s=28 if activation != "identity" else 24,
            alpha=0.70 if activation != "identity" else 0.42,
            color=ACTIVATION_COLORS.get(activation, "#444444"),
            edgecolors="none",
        )
        ax.plot(
            [i - 0.22, i + 0.22],
            [sub["shape_nn_fpr"].median()] * 2,
            color="#202124",
            linewidth=2,
        )
    threshold = float(close_base["close_threshold"].dropna().iloc[0])
    ax.axhline(threshold, color="#202124", linewidth=1.2, linestyle="--")
    ax.text(
        len(activations) - 0.55,
        threshold * 1.25,
        "close threshold",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#202124",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1.5e-2)
    ax.set_xticks(np.arange(len(activations)))
    ax.set_xticklabels(activations, rotation=28, ha="right")
    ax.set_ylabel("NN-FPR shape distance")
    ax.set_title("Only close-distance runs", fontsize=15, fontweight="bold")
    style_axis(ax)

    ax = axes[1]
    max_rmse = float(np.nanquantile(close_base[["nn_rmse", "fpr_rmse"]], 0.995)) * 1.08
    max_rmse = max(max_rmse, 0.08)
    for activation, sub in close_base.groupby("activation"):
        ax.scatter(
            sub["nn_rmse"],
            sub["fpr_rmse"],
            s=42 if activation != "identity" else 34,
            alpha=0.78 if activation != "identity" else 0.48,
            color=ACTIVATION_COLORS.get(activation, "#444444"),
            label=activation,
            edgecolors="none",
        )
    ax.plot([0, max_rmse], [0, max_rmse], color="#202124", linewidth=1.3)
    ax.fill_between(
        [0, max_rmse],
        [0, max_rmse * 0.92],
        [max_rmse * 0.08, max_rmse],
        color="#16857B",
        alpha=0.08,
        linewidth=0,
    )
    ax.set_xlim(0, max_rmse)
    ax.set_ylim(0, max_rmse)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Test RMSE parity against true y", fontsize=15, fontweight="bold")
    ax.set_xlabel("NN test RMSE")
    ax.set_ylabel("FullPR test RMSE")
    style_axis(ax, y_grid=False)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    ax = axes[2]
    nonidentity = close_base[close_base["activation"] != "identity"]
    summary = pd.DataFrame(
        [
            {
                "group": "All close runs",
                "runs": len(close_base),
                "gap": close_base["rmse_gap"].median(),
                "mimic": close_base["mimic_rmse"].median(),
            },
            {
                "group": "Nonlinear close runs",
                "runs": len(nonidentity),
                "gap": nonidentity["rmse_gap"].median(),
                "mimic": nonidentity["mimic_rmse"].median(),
            },
        ]
    )
    x = np.arange(len(summary))
    bars = ax.bar(x, summary["gap"], width=0.58, color=["#7A7F86", "#16857B"])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row.group}\nN={int(row.runs)}" for row in summary.itertuples()],
        fontsize=11,
    )
    ax.set_ylabel("Median absolute RMSE gap")
    ax.set_title("Partial data summary", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(0.03, float(summary["gap"].max()) * 1.65))
    style_axis(ax)
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ax.get_ylim()[1] * 0.035,
            f"{row.gap:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#202124",
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ax.get_ylim()[1] * 0.86,
            f"mimic RMSE\n{row.mimic:.4f}",
            ha="center",
            va="top",
            fontsize=10,
            color="#5F6368",
        )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.16, wspace=0.24)

    out = OUT_DIR / "close_distance_prediction_closeness.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def save_dopt_advantage_table(close_rows: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    nonidentity = close_rows[close_rows["activation"] != "identity"].copy()
    rows = []
    for p, sub in nonidentity.groupby("p"):
        rows.append(
            {
                "p": int(p),
                "reps": len(sub),
                "dopt_median_gain_pct": sub["dopt_gain_vs_random_median_pct"].median(),
                "dopt_error_median_gain_pct": sub[
                    "dopt_err_gain_vs_random_median_pct"
                ].median(),
                "latin_median_gain_pct": sub["latin_gain_vs_random_median_pct"].median(),
                "dopt_win_rate_pct": (sub["dopt_gain_vs_random_median_pct"] > 0).mean()
                * 100,
                "dopt_error_win_rate_pct": (
                    sub["dopt_err_gain_vs_random_median_pct"] > 0
                ).mean()
                * 100,
                "latin_win_rate_pct": (sub["latin_gain_vs_random_median_pct"] > 0).mean()
                * 100,
            }
        )
    out = TABLE_DIR / "close_distance_error_dopt_advantage_by_p.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def make_error_dopt_advantage_figure(close_rows: pd.DataFrame) -> Path:
    nonidentity = close_rows[close_rows["activation"] != "identity"].copy()
    rows = []
    for p, sub in nonidentity.groupby("p"):
        for method, label in [
            ("dopt", "D-opt"),
            ("dopt_err", "D + error reg."),
            ("latin", "Latin"),
        ]:
            values = sub[f"{method}_gain_vs_random_median_pct"]
            rows.append(
                {
                    "p": int(p),
                    "method": label,
                    "median_gain": float(values.median()),
                    "win_rate": float((values > 0).mean() * 100.0),
                }
            )
    stats = pd.DataFrame(rows)
    p_values = sorted(stats["p"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Under close distance, error-aware D-optimal design wins consistently",
        fontsize=22,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.91,
        "Nonlinear activations only; close-distance runs; gain is MSE reduction vs same-budget random median",
        ha="center",
        fontsize=12,
        color="#5F6368",
    )

    width = 0.25
    offsets = {"D-opt": -width, "D + error reg.": 0.0, "Latin": width}
    x = np.arange(len(p_values))
    for ax, metric, ylabel, title, ylim in [
        (
            axes[0],
            "median_gain",
            "Median MSE gain (%)",
            "Predictive gain",
            (-130, 55),
        ),
        (
            axes[1],
            "win_rate",
            "Win rate over random median (%)",
            "Reliability",
            (0, 108),
        ),
    ]:
        for method in ["D-opt", "D + error reg.", "Latin"]:
            sub = stats[stats["method"] == method].sort_values("p")
            bars = ax.bar(
                x + offsets[method],
                sub[metric],
                width,
                color=METHOD_COLORS[method],
                label=method,
            )
            label_bars(ax, bars)
        ax.axhline(0, color="#202124", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"p={p}" for p in p_values])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_ylim(*ylim)
        style_axis(ax)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.18, wspace=0.22)

    out = OUT_DIR / "close_distance_error_dopt_advantage.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_results()
    close_base, close_rows = load_results()
    paths = [
        make_distance_abs_mse_relationship_compact(base),
        make_distance_abs_mse_relationship_compact(base, include_identity=False),
        make_distance_abs_mse_relationship_figure(base),
        make_prediction_closeness_figure(close_base),
        make_error_dopt_advantage_figure(close_rows),
        save_distance_relationship_table(base),
        *save_prediction_tables(close_base),
        save_dopt_advantage_table(close_rows),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
