from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "measure_weighted"
FIG_DIR = ROOT / "figures" / "measure_weighted"
NOTES_DIR = ROOT / "reports" / "notes"
PREFIXES = {
    20: "measure_weighted_sampling_grand_mean_p20",
    50: "measure_weighted_sampling_grand_mean_p50",
}
MIN_PREFIXES = {
    20: "measure_weighted_sampling_grand_p20",
    50: "measure_weighted_sampling_grand_p50",
}

STRATEGY_ORDER = ["measure_weighted", "random", "dopt", "latin"]
STRATEGY_LABELS = {
    "measure_weighted": "Measure-weighted",
    "random": "Random",
    "dopt": "D-optimal",
    "latin": "Latin hypercube",
}
STRATEGY_COLORS = {
    "measure_weighted": "#0072B2",
    "random": "#6C6C6C",
    "dopt": "#D55E00",
    "latin": "#009E73",
}
CASE_LABELS = {
    "highdim_poly2": "Quadratic",
    "highdim_smooth": "Smooth",
    "highdim_strong": "Strong nonlinear",
    "highdim_local": "Local interior",
    "highdim_highfreq": "High-frequency",
}
CASE_ORDER = [
    "highdim_poly2",
    "highdim_smooth",
    "highdim_strong",
    "highdim_local",
    "highdim_highfreq",
]


def load_kind(kind, prefixes):
    root_paths = [ROOT / f"{prefix}_{kind}.csv" for prefix in prefixes.values()]
    if not any(path.exists() for path in root_paths):
        combined_path = DATA_DIR / f"measure_weighted_grand_mean_{kind}_combined.csv"
        if combined_path.exists():
            return pd.read_csv(combined_path)

    frames = []
    for p, prefix in prefixes.items():
        path = ROOT / f"{prefix}_{kind}.csv"
        if not path.exists():
            data_path = DATA_DIR / f"{prefix}_{kind}.csv"
            if data_path.exists():
                path = data_path
        df = pd.read_csv(path)
        df["p_dim"] = p
        df["source_prefix"] = prefix
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def summarize_final(final):
    summary = (
        final.groupby(["p_dim", "strategy", "strategy_label"], as_index=False)
        .agg(
            final_mse_mean=("test_mse", "mean"),
            final_mse_std=("test_mse", "std"),
            final_mse_se=("test_mse", sem),
            gain_vs_random_mean=("gain_vs_random_pct", "mean"),
            gain_vs_random_std=("gain_vs_random_pct", "std"),
            gain_vs_random_se=("gain_vs_random_pct", sem),
            positive_gain_rate=("gain_vs_random_pct", lambda x: float((x > 0).mean())),
            runs=("test_mse", "size"),
        )
        .sort_values(["p_dim", "final_mse_mean"])
    )
    return summary


def summarize_cases(final):
    return (
        final.groupby(["p_dim", "case", "case_label", "strategy", "strategy_label"], as_index=False)
        .agg(
            final_mse_mean=("test_mse", "mean"),
            gain_vs_random_mean=("gain_vs_random_pct", "mean"),
            positive_gain_rate=("gain_vs_random_pct", lambda x: float((x > 0).mean())),
            runs=("test_mse", "size"),
        )
        .sort_values(["p_dim", "case", "final_mse_mean"])
    )


def pairwise(final):
    rows = []
    key = ["p_dim", "case", "data_seed", "init_seed"]
    mw = final[final["strategy"] == "measure_weighted"][key + ["test_mse"]].rename(
        columns={"test_mse": "mw_mse"}
    )
    for comp in ["random", "dopt", "latin"]:
        other = final[final["strategy"] == comp][key + ["test_mse"]].rename(
            columns={"test_mse": f"{comp}_mse"}
        )
        merged = mw.merge(other, on=key, how="inner")
        diff = merged[f"{comp}_mse"] - merged["mw_mse"]
        pct = diff / merged[f"{comp}_mse"] * 100.0
        for p_dim, sub_idx in merged.groupby("p_dim").groups.items():
            sub_diff = diff.loc[sub_idx]
            sub_pct = pct.loc[sub_idx]
            rows.append(
                {
                    "p_dim": p_dim,
                    "comparison": f"measure_weighted_vs_{comp}",
                    "mean_mse_diff": float(sub_diff.mean()),
                    "median_mse_diff": float(sub_diff.median()),
                    "win_rate": float((sub_diff > 0).mean()),
                    "mean_pct_gain": float(sub_pct.mean()),
                    "pct_gain_se": sem(sub_pct),
                    "runs": len(sub_pct),
                }
            )
        rows.append(
            {
                "p_dim": "all",
                "comparison": f"measure_weighted_vs_{comp}",
                "mean_mse_diff": float(diff.mean()),
                "median_mse_diff": float(diff.median()),
                "win_rate": float((diff > 0).mean()),
                "mean_pct_gain": float(pct.mean()),
                "pct_gain_se": sem(pct),
                "runs": len(pct),
            }
        )
    return pd.DataFrame(rows)


def plot_gain_by_p(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, p_dim in zip(axes, [20, 50]):
        sub = summary[summary["p_dim"] == p_dim].copy()
        sub["strategy"] = pd.Categorical(sub["strategy"], STRATEGY_ORDER, ordered=True)
        sub = sub.sort_values("strategy")
        x = np.arange(len(sub))
        colors = [STRATEGY_COLORS[s] for s in sub["strategy"]]
        ax.bar(x, sub["gain_vs_random_mean"], color=colors, alpha=0.88)
        ax.errorbar(
            x,
            sub["gain_vs_random_mean"],
            yerr=1.96 * sub["gain_vs_random_se"],
            fmt="none",
            ecolor="black",
            linewidth=1,
            capsize=4,
        )
        ax.axhline(0, color="black", linewidth=0.9, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in sub["strategy"]], rotation=18, ha="right")
        ax.set_title(f"p={p_dim}")
        ax.set_ylabel("Mean final gain vs random (%)")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "measure_weighted_grand_gain_by_p.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_learning_gain(raw):
    random_ref = raw[raw["strategy"] == "random"][
        ["p_dim", "case", "data_seed", "init_seed", "round", "test_mse"]
    ].rename(columns={"test_mse": "random_mse"})
    merged = raw.merge(
        random_ref,
        on=["p_dim", "case", "data_seed", "init_seed", "round"],
        how="left",
    )
    merged["gain_vs_random_round"] = (
        (merged["random_mse"] - merged["test_mse"]) / merged["random_mse"] * 100.0
    )
    stats = (
        merged.groupby(["p_dim", "strategy", "round"], as_index=False)
        .agg(mean=("gain_vs_random_round", "mean"), se=("gain_vs_random_round", sem))
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, p_dim in zip(axes, [20, 50]):
        sub = stats[stats["p_dim"] == p_dim]
        for strategy in STRATEGY_ORDER:
            one = sub[sub["strategy"] == strategy].sort_values("round")
            if one.empty:
                continue
            x = one["round"].to_numpy()
            y = one["mean"].to_numpy()
            se_arr = one["se"].to_numpy()
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
            )
            ax.fill_between(
                x,
                y - 1.96 * se_arr,
                y + 1.96 * se_arr,
                color=STRATEGY_COLORS[strategy],
                alpha=0.12,
                linewidth=0,
            )
        ax.axhline(0, color="black", linewidth=0.9, alpha=0.6)
        ax.set_title(f"p={p_dim}")
        ax.set_xlabel("Batch round")
        ax.set_ylabel("Gain vs random at same round (%)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "measure_weighted_grand_learning_gain.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_case_gain_heatmap(case_summary):
    mw = case_summary[case_summary["strategy"] == "measure_weighted"].copy()
    table = (
        mw.pivot(index="case", columns="p_dim", values="gain_vs_random_mean")
        .reindex(CASE_ORDER)
        .rename(index=CASE_LABELS)
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    vals = table.to_numpy(dtype=float)
    vmax = max(1.0, np.nanmax(np.abs(vals)))
    im = ax.imshow(vals, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([f"p={c}" for c in table.columns])
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = vals[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9)
    ax.set_title("Measure-weighted final gain vs random")
    fig.colorbar(im, ax=ax, label="Gain (%)")
    fig.tight_layout()
    out = FIG_DIR / "measure_weighted_grand_case_gain_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_weight_heatmap(distances):
    table = (
        distances.groupby(["case", "p_dim"])["dopt_weight"]
        .mean()
        .unstack("p_dim")
        .reindex(CASE_ORDER)
        .rename(index=CASE_LABELS)
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    vals = table.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([f"p={c}" for c in table.columns])
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = vals[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Mean D-optimal fraction learned from NN-PR distance")
    fig.colorbar(im, ax=ax, label="D-optimal fraction")
    fig.tight_layout()
    out = FIG_DIR / "measure_weighted_grand_weight_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def draw_box(ax, xy, wh, text, fc="#F4F6F9", ec="#2E74B5", fontsize=10):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#1F4D78"),
    )


def plot_flowchart(lang="cn"):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if lang == "en":
        title = "Measure-weighted batch sampling workflow"
        note = "Larger weight means more D-optimal points in the next batch; smaller weight preserves more random exploration."
        suffix = "en"
        boxes = {
            "split": ((0.05, 0.76), (0.18, 0.12), "Original dataset\nTrain/test split"),
            "pilot": ((0.30, 0.76), (0.18, 0.12), "Uniform sample from train\nPilot subset"),
            "init": ((0.55, 0.76), (0.18, 0.12), "Train initial NN\nSingle hidden-layer tanh"),
            "dist": ((0.78, 0.76), (0.18, 0.12), "Measure distance\nNN-FPR / NN-TYPR"),
            "norm": ((0.78, 0.53), (0.18, 0.13), "Normalize to [0, 1]\nTransfer weight"),
            "mix": ((0.55, 0.53), (0.18, 0.13), "Fixed batch size\nD-optimal + random mix"),
            "train": ((0.30, 0.53), (0.18, 0.13), "Add new batch\nRetrain NN"),
            "eval": ((0.05, 0.53), (0.18, 0.13), "Evaluate on test set\nMSE / learning curve"),
            "loop": ((0.30, 0.29), (0.43, 0.12), "Repeat batch rounds\nUpdate training set and continue sampling"),
            "controls": ((0.05, 0.16), (0.43, 0.10), "Controls: all-random / all-D-optimal / Latin hypercube"),
            "claim": ((0.55, 0.16), (0.41, 0.10), "Paper interpretation: transfer legacy methods when distance is small\npreserve random exploration when distance is large"),
        }
    else:
        title = "距离测度加权的批次采样流程"
        note = "weight 越大，下一批中 D-optimal 点越多；weight 越小，随机探索比例越高。"
        suffix = "cn"
        boxes = {
            "split": ((0.05, 0.76), (0.18, 0.12), "原始数据集\nTrain/Test 拆分"),
            "pilot": ((0.30, 0.76), (0.18, 0.12), "训练集均匀抽样\nPilot subset"),
            "init": ((0.55, 0.76), (0.18, 0.12), "训练初始 NN\n单隐层 tanh"),
            "dist": ((0.78, 0.76), (0.18, 0.12), "计算距离\nNN-FPR / NN-TYPR"),
            "norm": ((0.78, 0.53), (0.18, 0.13), "归一化到 [0,1]\n得到 transfer weight"),
            "mix": ((0.55, 0.53), (0.18, 0.13), "固定 batch size\nD-optimal + 随机混合"),
            "train": ((0.30, 0.53), (0.18, 0.13), "加入新批次\n重新训练 NN"),
            "eval": ((0.05, 0.53), (0.18, 0.13), "测试集评估\nMSE / 学习曲线"),
            "loop": ((0.30, 0.29), (0.43, 0.12), "重复多个 batch round\n更新训练集并继续采样"),
            "controls": ((0.05, 0.16), (0.43, 0.10), "对照组：全随机 / 全 D-optimal / Latin hypercube"),
            "claim": ((0.55, 0.16), (0.41, 0.10), "论文解释：距离小则迁移旧技术\n距离大则保留随机探索，降低负迁移"),
        }
    for key, (xy, wh, text) in boxes.items():
        fill = "#E8EEF5" if key in {"dist", "norm", "mix"} else "#F7F9FC"
        draw_box(ax, xy, wh, text, fc=fill)

    arrow(ax, (0.23, 0.82), (0.30, 0.82))
    arrow(ax, (0.48, 0.82), (0.55, 0.82))
    arrow(ax, (0.73, 0.82), (0.78, 0.82))
    arrow(ax, (0.87, 0.76), (0.87, 0.66))
    arrow(ax, (0.78, 0.60), (0.73, 0.60))
    arrow(ax, (0.55, 0.60), (0.48, 0.60))
    arrow(ax, (0.30, 0.60), (0.23, 0.60))
    arrow(ax, (0.39, 0.53), (0.43, 0.41))
    arrow(ax, (0.51, 0.41), (0.62, 0.53))
    arrow(ax, (0.26, 0.29), (0.25, 0.26))
    arrow(ax, (0.51, 0.29), (0.67, 0.26))

    ax.text(
        0.5,
        0.95,
        title,
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
        color="#0B2545",
    )
    ax.text(
        0.5,
        0.04,
        note,
        ha="center",
        va="center",
        fontsize=11,
        color="#555555",
    )
    fig.tight_layout()
    out = FIG_DIR / f"measure_weighted_sampling_flowchart_{suffix}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def markdown_table(df, cols, formats=None):
    formats = formats or {}
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if col in formats:
                vals.append(formats[col](val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary, case_summary, pairwise_df, distances, plot_paths):
    mw_summary = summary[summary["strategy"] == "measure_weighted"].copy()
    min_summary = []
    for p, prefix in MIN_PREFIXES.items():
        path = ROOT / f"{prefix}_summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            row = df[df["strategy"] == "measure_weighted"].iloc[0].to_dict()
            row["p_dim"] = p
            min_summary.append(row)
    min_df = pd.DataFrame(min_summary)

    lines = [
        "# Grand measure-weighted sampling experiment",
        "",
        "## Scope",
        "",
        (
            "This grand benchmark uses n=10000, p in {20, 50}, five dataset families, "
            "five data seeds, two initial sampling seeds, four sampling strategies, "
            "six batch updates, and a fixed batch size of 500. The main version uses "
            "the conservative combined distance mean(d_FPR, d_TYPR) to decide the "
            "D-optimal fraction."
        ),
        "",
        "## Main summary by dimension",
        "",
        markdown_table(
            summary[["p_dim", "strategy_label", "final_mse_mean", "gain_vs_random_mean", "positive_gain_rate", "runs"]],
            ["p_dim", "strategy_label", "final_mse_mean", "gain_vs_random_mean", "positive_gain_rate", "runs"],
            {
                "final_mse_mean": lambda x: f"{x:.6f}",
                "gain_vs_random_mean": lambda x: f"{x:.3f}%",
                "positive_gain_rate": lambda x: f"{100*x:.1f}%",
            },
        ),
        "",
        "## Measure-weighted pairwise comparisons",
        "",
        markdown_table(
            pairwise_df[["p_dim", "comparison", "win_rate", "mean_pct_gain", "runs"]],
            ["p_dim", "comparison", "win_rate", "mean_pct_gain", "runs"],
            {
                "win_rate": lambda x: f"{100*x:.1f}%",
                "mean_pct_gain": lambda x: f"{x:.3f}%",
            },
        ),
        "",
        "## Measure-weighted gains by case",
        "",
        markdown_table(
            case_summary[case_summary["strategy"] == "measure_weighted"][
                ["p_dim", "case_label", "final_mse_mean", "gain_vs_random_mean", "positive_gain_rate"]
            ],
            ["p_dim", "case_label", "final_mse_mean", "gain_vs_random_mean", "positive_gain_rate"],
            {
                "final_mse_mean": lambda x: f"{x:.6f}",
                "gain_vs_random_mean": lambda x: f"{x:.3f}%",
                "positive_gain_rate": lambda x: f"{100*x:.1f}%",
            },
        ),
        "",
        "## Weight calibration",
        "",
        markdown_table(
            distances.groupby(["p_dim", "case", "case_label"], as_index=False)
            .agg(
                fpr_distance_mean=("fpr_distance", "mean"),
                tpr_distance_mean=("tpr_distance", "mean"),
                dopt_weight_mean=("dopt_weight", "mean"),
            )
            [["p_dim", "case_label", "fpr_distance_mean", "tpr_distance_mean", "dopt_weight_mean"]],
            ["p_dim", "case_label", "fpr_distance_mean", "tpr_distance_mean", "dopt_weight_mean"],
            {
                "fpr_distance_mean": lambda x: f"{x:.5f}",
                "tpr_distance_mean": lambda x: f"{x:.5f}",
                "dopt_weight_mean": lambda x: f"{x:.3f}",
            },
        ),
        "",
        "## Sensitivity note",
        "",
    ]
    if not min_df.empty:
        lines.extend(
            [
                (
                    "The earlier min(d_FPR, d_TYPR) rule was intentionally less conservative. "
                    "In p=50 it caused negative transfer for the measure-weighted policy "
                    "because a single small PR distance could trigger too much D-optimal sampling. "
                    "The mean-distance rule reduces that over-transfer."
                ),
                "",
                markdown_table(
                    min_df[["p_dim", "strategy_label", "final_mse_mean", "gain_vs_random_mean", "positive_gain_rate"]],
                    ["p_dim", "strategy_label", "final_mse_mean", "gain_vs_random_mean", "positive_gain_rate"],
                    {
                        "final_mse_mean": lambda x: f"{x:.6f}",
                        "gain_vs_random_mean": lambda x: f"{x:.3f}%",
                        "positive_gain_rate": lambda x: f"{100*x:.1f}%",
                    },
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
        ]
    )
    for path in plot_paths:
        lines.append(f"- {path.name}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The measure is most useful as a transfer gate rather than as a universal "
                "winner over every control. It improves over random in p=20 and becomes "
                "competitive in p=50 after conservative distance aggregation. In high-frequency "
                "or high-dimensional settings, the learned weight exposes when old PR/D-optimal "
                "technology should be used cautiously."
            ),
            "",
        ]
    )
    out = NOTES_DIR / "measure_weighted_grand_mean_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_kind("raw", PREFIXES)
    final = load_kind("final", PREFIXES)
    distances = load_kind("distances", PREFIXES)

    raw.to_csv(DATA_DIR / "measure_weighted_grand_mean_raw_combined.csv", index=False)
    final.to_csv(DATA_DIR / "measure_weighted_grand_mean_final_combined.csv", index=False)
    distances.to_csv(DATA_DIR / "measure_weighted_grand_mean_distances_combined.csv", index=False)

    summary = summarize_final(final)
    case_summary = summarize_cases(final)
    pairwise_df = pairwise(final)
    summary.to_csv(DATA_DIR / "measure_weighted_grand_mean_strategy_summary.csv", index=False)
    case_summary.to_csv(DATA_DIR / "measure_weighted_grand_mean_case_summary.csv", index=False)
    pairwise_df.to_csv(DATA_DIR / "measure_weighted_grand_mean_pairwise.csv", index=False)

    paths = [
        plot_gain_by_p(summary),
        plot_learning_gain(raw),
        plot_case_gain_heatmap(case_summary),
        plot_weight_heatmap(distances),
        plot_flowchart("cn"),
        plot_flowchart("en"),
    ]
    report = write_report(summary, case_summary, pairwise_df, distances, paths)
    print(f"Wrote {report}")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
