from __future__ import annotations

import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SOURCE_DIR = Path(
    r"C:/Users/sheesh/Documents/xwechat_files/wxid_xvkf6py2gc8o22_ee18/msg/file/2026-09"
)
WORKSPACE = Path(__file__).resolve().parent
FIG_DIR = WORKSPACE / "figures" / "geom_noltpr_fivep"
REPORT_PATH = WORKSPACE / "reports" / "notes" / "geom_gpu_noltpr_fivep_report.md"

SAFE_MAX_ORDER = 3
FULLPR_FEATURE_CAP = 30_000
N_ACTIVE = 8
N_TRAIN = 7_500


VARIABLE_DEFINITIONS = [
    ("case", "数据生成机制/实验场景。`highdim_poly3` 是三阶多项式真函数；`highdim_poly4` 是四阶多项式真函数；`highdim_nonlinear` 是含正弦与交互项的非多项式真函数。"),
    ("activation", "NN 使用的激活函数，例如 `softplus`、`tanh`、`sigmoid`。它决定网络非线性的形状，也影响泰勒类近似的稳定性。"),
    ("hidden", "隐藏层宽度配置。`64` 表示单隐层 64 个神经元；`128,64` 表示两隐层；`256,128,64` 表示三隐层。"),
    ("n_hidden", "隐藏层层数，由 `hidden` 的长度得到。当前实验中为 1、2、3。"),
    ("q", "Taylor-PR 的泰勒展开阶数参数。当前数据中为 3，但 GPU 脚本又受 `SAFE_MAX_ORDER=3` 限制。"),
    ("data_seed", "数据集随机种子。控制 X、真实函数系数、噪声等数据生成随机性。"),
    ("init_seed", "NN 初始化随机种子。控制网络初始权重和训练随机性；同一配置下重复多次用于估计稳定性。"),
    ("special", "FullPR 是否额外加入特殊函数特征。为 True 时，每个输入变量额外加入 `sin(x_i)` 和 `exp(x_i)` 两类特征。"),
    ("n", "总样本数。当前实验为 10000，训练/测试拆分后约 7500/2500。"),
    ("p", "输入维度，也就是自变量个数。当前实验扫 10、20、50、100、200。"),
    ("max_order", "本次 FullPR/FPR 实际使用的最高多项式总阶数。它不是简单随 p 增加，而是由隐藏层数、`SAFE_MAX_ORDER` 和特征数上限共同决定。"),
    ("NN_mse_vs_y", "NN 预测相对真实测试标签 y 的均方误差。越小表示 NN 本身预测越准。"),
    ("act_io_mean", "各隐藏层从 pre-activation `u` 到 activation 输出 `g(u)` 的形状/响应变化距离的平均值。反映平均每层注入了多少非线性。"),
    ("act_io_sum", "各隐藏层 `u -> g(u)` 距离之和。比 `act_io_mean` 更强调深度累积的总非线性。"),
    ("TPR_mse_vs_NN", "Taylor-PR 预测相对 NN 输出的均方误差。衡量 Taylor-PR 是否能复刻 NN。当前只有单隐层时计算。"),
    ("TPR_mse_vs_y", "Taylor-PR 预测相对真实 y 的均方误差。衡量 Taylor-PR 自身预测性能。"),
    ("LTPR_mse_vs_NN", "LayerTaylor-PR 预测相对 NN 输出的均方误差。当前这批 `noltpr` 实验中全部为空，因为运行时设置 `--ltpr-max-p 0` 跳过了 LTPR。"),
    ("LTPR_mse_vs_y", "LayerTaylor-PR 预测相对真实 y 的均方误差。本批数据为空，不能据此下 LTPR 结论。"),
    ("LTPR_n_terms", "LayerTaylor-PR 展开后得到的多项式项数。本批数据为空。"),
    ("shape_NN_LTPR", "NN 与 LayerTaylor-PR 在测试数据区响应曲面 `[X, yhat]` 上的 Procrustes 形状距离。本批数据为空。"),
    ("FPR_mse_vs_y", "FullPR/FPR 预测相对真实 y 的均方误差。越小表示 FPR 自身预测越准。"),
    ("FPR_mse_vs_NN", "FPR 预测相对 NN 输出的均方误差。衡量 FPR 是否能复刻 NN，是“像不像 NN”的直接误差指标。"),
    ("FPR_r2_vs_y", "FPR 相对真实 y 的 R2。越接近 1 表示解释真实 y 的方差比例越高；负值表示比常数均值预测还差。"),
    ("shape_NN_FPR_in", "NN 与 FPR 在测试数据区响应曲面 `[X, yhat]` 上的 Procrustes 形状距离。越小表示二者在数据区形状越接近。"),
    ("mahal_NN_FPR", "NN 与 FPR 输出差异的 Mahalanobis 平均距离。它考虑输出差异的协方差结构，比普通欧氏差异更带统计尺度。"),
    ("shape_NN_FPR_ext", "NN 与 FPR 在外推区响应曲面上的形状距离。外推区由更宽输入范围采样得到，衡量离开训练/测试分布后的形状一致性。"),
    ("abs_rmse_gap", "`abs(sqrt(NN_mse_vs_y) - sqrt(FPR_mse_vs_y))`。这是 NN 与 FPR 相对真实 y 的 RMSE 性能差距，越小表示性能越接近。"),
    ("NN_FPR_log10", "`log10(NN_mse_vs_y / FPR_mse_vs_y)`。大于 0 表示 NN 的 MSE 大于 FPR，即 FPR 更好；小于 0 表示 NN 更好。"),
    ("FPR_mse_delta_vs_NN", "`FPR_mse_vs_y - NN_mse_vs_y`。负值表示 FPR 比 NN 更准，正值表示 FPR 更差。"),
    ("FPR_improve_vs_NN_pct", "`(NN_mse_vs_y - FPR_mse_vs_y) / NN_mse_vs_y * 100%`。正值表示 FPR 相对 NN 改善，负值表示退化。"),
    ("FPR_better_than_NN", "布尔指标。True 表示该 run 中 FPR 的 MSE 小于 NN 的 MSE。"),
    ("TPR_mse_delta_vs_NN", "`TPR_mse_vs_y - NN_mse_vs_y`。负值表示 TPR 比 NN 更准。"),
    ("TPR_improve_vs_NN_pct", "`(NN_mse_vs_y - TPR_mse_vs_y) / NN_mse_vs_y * 100%`。正值表示 TPR 相对 NN 改善。"),
    ("TPR_better_than_NN", "布尔指标。True 表示 TPR 的 MSE 小于 NN 的 MSE。当前只对单隐层有效。"),
    ("LTPR_mse_delta_vs_NN", "`LTPR_mse_vs_y - NN_mse_vs_y`。本批数据为空。"),
    ("LTPR_improve_vs_NN_pct", "`(NN_mse_vs_y - LTPR_mse_vs_y) / NN_mse_vs_y * 100%`。本批数据为空。"),
    ("LTPR_better_than_NN", "布尔指标。True 表示 LTPR 的 MSE 小于 NN 的 MSE。本批数据为空/不可用。"),
    ("well_specified", "是否为设定匹配的对照。当前代码定义为 `case == highdim_poly3` 且 `n_hidden == 3`，因为三隐层对应三阶 FPR，在低/中维时可以命中三阶真函数。"),
]


def read_raw() -> pd.DataFrame:
    files = sorted(
        SOURCE_DIR.glob("geom_gpu_noltpr_p*_raw.csv"),
        key=lambda path: int(re.search(r"p(\d+)_raw", path.name).group(1)),
    )
    frames = []
    for path in files:
        p_value = int(re.search(r"p(\d+)_raw", path.name).group(1))
        frames.append(pd.read_csv(path).assign(source_file=path.name, source_p=p_value))
    if not frames:
        raise FileNotFoundError(f"No raw CSV files found in {SOURCE_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df["FPR_NN_mse_ratio"] = df["FPR_mse_vs_y"] / df["NN_mse_vs_y"]
    df["TPR_NN_mse_ratio"] = df["TPR_mse_vs_y"] / df["NN_mse_vs_y"]
    return df


def read_correlations() -> pd.DataFrame:
    files = sorted(
        SOURCE_DIR.glob("geom_gpu_noltpr_p*_correlations.csv"),
        key=lambda path: int(re.search(r"p(\d+)_correlations", path.name).group(1)),
    )
    frames = []
    for path in files:
        p_value = int(re.search(r"p(\d+)_correlations", path.name).group(1))
        frames.append(pd.read_csv(path).assign(source_file=path.name, source_p=p_value))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "NA"
    if abs(x) >= 1000 or (0 < abs(x) < 0.001):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.1f}%"


def feature_count(p: int, order: int, include_special: bool = True) -> int:
    return math.comb(p + order, order) - 1 + (2 * p if include_special else 0)


def chosen_order(p: int, n_hidden: int, include_special: bool = True) -> int:
    requested = min(n_hidden, SAFE_MAX_ORDER)
    for order in range(requested, 0, -1):
        if feature_count(p, order, include_special) <= FULLPR_FEATURE_CAP:
            return order
    return 1


def screened_fpr3_count(p: int, k: int = N_ACTIVE, include_special: bool = True) -> int:
    # Degree <=3 interactions are built only on k screened variables.
    # Non-selected variables keep main effects, and optional special features are kept for all variables.
    return (math.comb(k + 3, 3) - 1) + max(p - k, 0) + (2 * p if include_special else 0)


def save_dimension_summary(df: pd.DataFrame) -> Path:
    summary = df.groupby("source_p").agg(
        FPR_ratio=("FPR_NN_mse_ratio", "median"),
        TPR_ratio=("TPR_NN_mse_ratio", "median"),
        FPR_better=("FPR_better_than_NN", "mean"),
        TPR_better=("TPR_better_than_NN", "mean"),
        shape_in=("shape_NN_FPR_in", "median"),
        abs_gap=("abs_rmse_gap", "median"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    x = summary["source_p"].to_numpy()
    axes[0].plot(x, summary["FPR_ratio"], marker="o", label="FPR / NN")
    axes[0].plot(x, summary["TPR_ratio"], marker="s", label="TPR / NN")
    axes[0].axhline(1, color="0.25", linestyle="--", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Input dimension p")
    axes[0].set_ylabel("Median MSE ratio vs NN (log)")
    axes[0].set_title("Model error relative to NN")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, 100 * summary["FPR_better"], marker="o", label="FPR better")
    axes[1].plot(x, 100 * summary["TPR_better"], marker="s", label="TPR better")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Input dimension p")
    axes[1].set_ylabel("Runs better than NN (%)")
    axes[1].set_title("Win rate against NN")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Dimension trend: FPR is close at low p, degrades at high p", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "dimension_trend.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_case_order_heatmap(df: pd.DataFrame) -> Path:
    pivot = (
        df.groupby(["case", "max_order", "source_p"])["FPR_NN_mse_ratio"]
        .median()
        .unstack("source_p")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=150)
    values = np.log10(pivot.to_numpy(dtype=float))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{case}, order {order}" for case, order in pivot.index])
    ax.set_xlabel("Input dimension p")
    ax.set_title("Median log10(FPR MSE / NN MSE) by case and actual FPR order")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            raw_value = pivot.iloc[i, j]
            if pd.notna(raw_value):
                ax.text(j, i, fmt(raw_value, 3), ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("log10 ratio; blue = FPR better, red = NN better")
    fig.tight_layout()
    path = FIG_DIR / "case_order_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_shape_perf_scatter(df: pd.DataFrame) -> Path:
    p_values = sorted(df["source_p"].dropna().unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), dpi=150, sharex=False, sharey=False)
    axes = axes.ravel()
    colors = {"highdim_poly3": "#4C78A8", "highdim_poly4": "#F58518", "highdim_nonlinear": "#54A24B"}
    for ax, p_value in zip(axes, p_values):
        sub = df[df["source_p"] == p_value].dropna(subset=["shape_NN_FPR_in", "abs_rmse_gap"])
        for case, g in sub.groupby("case"):
            ax.scatter(
                g["shape_NN_FPR_in"],
                g["abs_rmse_gap"],
                s=16,
                alpha=0.55,
                label=case,
                color=colors.get(case),
                edgecolors="none",
            )
        rho, pv = spearmanr(sub["shape_NN_FPR_in"], sub["abs_rmse_gap"]) if len(sub) >= 5 else (np.nan, np.nan)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"p={int(p_value)}; Spearman rho={rho:.2f}")
        ax.set_xlabel("shape_NN_FPR_in (log)")
        ax.set_ylabel("abs_rmse_gap (log)")
        ax.grid(alpha=0.25)
    for ax in axes[len(p_values):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Core diagnostic: smaller NN-FPR shape distance usually means smaller performance gap",
        y=0.99,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    path = FIG_DIR / "shape_distance_vs_performance_gap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_comparability_plot() -> Path:
    p_values = np.array([10, 20, 50, 100, 200])
    full_order3 = np.array([feature_count(int(p), 3, True) for p in p_values])
    full_order2 = np.array([feature_count(int(p), 2, True) for p in p_values])
    screened_order3 = np.array([screened_fpr3_count(int(p), N_ACTIVE, True) for p in p_values])
    actual_three_hidden = np.array([chosen_order(int(p), 3, True) for p in p_values])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)
    axes[0].plot(p_values, full_order3, marker="o", label="Full degree-3 FPR")
    axes[0].plot(p_values, full_order2, marker="s", label="Full degree-2 FPR")
    axes[0].plot(p_values, screened_order3, marker="^", label=f"Screened degree-3 FPR (k={N_ACTIVE})")
    axes[0].axhline(FULLPR_FEATURE_CAP, color="0.25", linestyle="--", linewidth=1, label="feature cap")
    axes[0].axhline(N_TRAIN, color="0.5", linestyle=":", linewidth=1, label="train n")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Input dimension p")
    axes[0].set_ylabel("Number of FPR features (log)")
    axes[0].set_title("Why full degree-3 is not comparable at high p")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(p_values, actual_three_hidden, marker="o")
    axes[1].set_xscale("log")
    axes[1].set_yticks([1, 2, 3])
    axes[1].set_ylim(0.8, 3.2)
    axes[1].set_xlabel("Input dimension p")
    axes[1].set_ylabel("Actual FPR order for 3-hidden NN")
    axes[1].set_title("Current code silently lowers order at p=100/200")
    axes[1].grid(alpha=0.25)
    for x, y in zip(p_values, actual_three_hidden):
        axes[1].text(x, y + 0.08, f"order {int(y)}", ha="center", fontsize=8)

    fig.tight_layout()
    path = FIG_DIR / "high_p_comparability_feature_counts.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    rows = df[columns].astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def build_report(df: pd.DataFrame, corr: pd.DataFrame, fig_paths: list[Path]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dim_summary = df.groupby("source_p").agg(
        n=("case", "size"),
        cases=("case", "nunique"),
        NN=("NN_mse_vs_y", "median"),
        FPR=("FPR_mse_vs_y", "median"),
        TPR=("TPR_mse_vs_y", "median"),
        FPR_ratio=("FPR_NN_mse_ratio", "median"),
        TPR_ratio=("TPR_NN_mse_ratio", "median"),
        FPR_better=("FPR_better_than_NN", "mean"),
        TPR_better=("TPR_better_than_NN", "mean"),
        shape_in=("shape_NN_FPR_in", "median"),
        abs_gap=("abs_rmse_gap", "median"),
    ).reset_index()
    dim_display = dim_summary.copy()
    for col in ["NN", "FPR", "TPR", "FPR_ratio", "TPR_ratio", "shape_in", "abs_gap"]:
        dim_display[col] = dim_display[col].map(fmt)
    for col in ["FPR_better", "TPR_better"]:
        dim_display[col] = dim_display[col].map(pct)

    case_summary = df.groupby(["case", "source_p"]).agg(
        n=("case", "size"),
        FPR_ratio=("FPR_NN_mse_ratio", "median"),
        FPR_better=("FPR_better_than_NN", "mean"),
        shape_in=("shape_NN_FPR_in", "median"),
        abs_gap=("abs_rmse_gap", "median"),
    ).reset_index()
    case_display = case_summary.copy()
    for col in ["FPR_ratio", "shape_in", "abs_gap"]:
        case_display[col] = case_display[col].map(fmt)
    case_display["FPR_better"] = case_display["FPR_better"].map(pct)

    order_rows = []
    for p in sorted(df["source_p"].unique()):
        for n_hidden in [1, 2, 3]:
            order = chosen_order(int(p), n_hidden, True)
            order_rows.append({
                "p": int(p),
                "n_hidden": n_hidden,
                "actual_max_order": order,
                "feature_count": feature_count(int(p), order, True),
            })
    order_display = pd.DataFrame(order_rows)

    comp_rows = []
    for p in [10, 20, 50, 100, 200]:
        comp_rows.append({
            "p": p,
            "full_degree_3_features": feature_count(p, 3, True),
            "current_3_hidden_order": chosen_order(p, 3, True),
            f"screened_degree_3_k{N_ACTIVE}_features": screened_fpr3_count(p, N_ACTIVE, True),
        })
    comp_display = pd.DataFrame(comp_rows)

    if not corr.empty:
        core_corr = corr[
            (corr["group"] == "ALL")
            & (corr["relation"] == "NN-FullPR closeness -> perf gap")
        ][["source_p", "spearman_rho", "p_value", "n"]].copy()
    else:
        core_corr = []
        for p, sub in df.groupby("source_p"):
            m = sub[["shape_NN_FPR_in", "abs_rmse_gap"]].dropna()
            rho, pv = spearmanr(m["shape_NN_FPR_in"], m["abs_rmse_gap"])
            core_corr.append({"source_p": p, "spearman_rho": rho, "p_value": pv, "n": len(m)})
        core_corr = pd.DataFrame(core_corr)
    core_display = core_corr.copy()
    core_display["spearman_rho"] = core_display["spearman_rho"].map(lambda x: fmt(x, 3))
    core_display["p_value"] = core_display["p_value"].map(lambda x: fmt(x, 3))

    rel_figs = [
        Path(os.path.relpath(path, REPORT_PATH.parent)).as_posix()
        for path in fig_paths
    ]
    variable_lines = "\n".join(f"- `{name}`：{desc}" for name, desc in VARIABLE_DEFINITIONS)

    report = f"""# GPU 高维 NN-FPR 几何等价实验小报告

## 1. 本轮实验要回答的问题

本轮实验的核心问题是：在不同输入维度 `p`、不同真实函数设定、不同激活函数和不同网络深度下，`FPR` 与 `NN` 的距离是否能作为性能接近或形状接近的诊断指标。

这里应区分两个命题：

1. **形状距离命题**：`shape_NN_FPR_in` 越小，NN 和 FPR 的响应曲面越接近。
2. **性能诊断命题**：`shape_NN_FPR_in` 越小，NN 与 FPR 相对真实 y 的性能差距 `abs_rmse_gap` 通常越小。

当前数据支持第二个命题的“诊断”版本，而不支持把它写成“距离小必然保证 FPR 更好或完全替代 NN”。

## 2. 数据范围和完整性

- 原始结果文件：`geom_gpu_noltpr_p10_raw.csv`、`geom_gpu_noltpr_p20_raw.csv`、`geom_gpu_noltpr_p50_raw.csv`、`geom_gpu_noltpr_p100_raw.csv`、`geom_gpu_noltpr_p200_raw.csv`。
- 相关性文件：p=10、20、50、100 有 correlation 文件；p=200 当前没有 correlation 文件。
- `LTPR` 相关列在本轮实验中全部为空，因为运行脚本使用了 `--ltpr-max-p 0`，即高维实验完全跳过 LayerTaylor-PR。
- p=200 只有 231 行，并且缺少 `highdim_nonlinear`，因此 p=200 可以作为高维趋势参考，但不应与 p=10/20/50/100 做完全公平的全配置比较。

## 3. 每个变量的含义

{variable_lines}

## 4. 维度层面的总体结论

{markdown_table(dim_display, ["source_p", "n", "cases", "NN", "FPR", "TPR", "FPR_ratio", "TPR_ratio", "FPR_better", "TPR_better", "shape_in", "abs_gap"])}

解读：

- p=10 和 p=20 时，FPR 的中位 MSE 大约是 NN 的 1.13 倍，胜率接近 46%-47%。这说明低维时 FPR 与 NN 已经处在可以正面对比的区域。
- p=50 时，FPR 中位误差比升到 1.28，胜率降到 35.7%，开始出现高维退化。
- p=100 和 p=200 时，FPR 明显退化，中位误差比分别约为 3.37 和 2.75，胜率降到 17.0% 和 5.2%。
- TPR 在单隐层上才有结果。随着 p 变高，TPR/NN 的中位误差比下降，但胜率仍然较低，说明它更像一个局部解释工具，而不是稳定预测模型。

![Dimension trend]({rel_figs[0]})

## 5. 分 case 的结果

{markdown_table(case_display, ["case", "source_p", "n", "FPR_ratio", "FPR_better", "shape_in", "abs_gap"])}

解读：

- `highdim_poly3` 在低维且三阶 FPR 可用时表现最好，符合“真函数阶数与 FPR 阶数匹配”的预期。
- `highdim_poly4` 是严格欠设定：即使三隐层最多也只给到三阶 FPR，无法完整表达四阶真函数，所以高维后 NN 更稳。
- `highdim_nonlinear` 不是多项式真函数，FPR 的优势取决于数据区局部是否可由低阶/特殊函数特征近似；它更适合用来测试“距离指标是否能诊断可迁移性”，而不是测试 FPR 是否必然胜过 NN。

![Case-order heatmap]({rel_figs[1]})

## 6. 核心诊断：形状距离是否代表性能接近

相关性摘要如下，使用的是 Spearman 秩相关：

{markdown_table(core_display, ["source_p", "spearman_rho", "p_value", "n"])}

这组结果非常关键：`shape_NN_FPR_in -> abs_rmse_gap` 在 p=10/20/50/100 上都显著为正，相关强度约 0.61 到 0.81。也就是说，数据区内 NN-FPR 形状距离越小，NN 与 FPR 相对真实 y 的性能差距通常越小。

这支持你的核心想法：**FPR 和 NN 的距离可以作为性能接近和形状接近的强诊断指标。**

但应注意：

- 它是诊断指标，不是充分条件。
- 它更能预测“NN 和 FPR 的性能差距是否小”，不等同于预测“FPR 是否一定比 NN 更准”。
- 当高维下 FPR 阶数被降阶、特征空间欠表达、或 NN 训练不稳定时，距离-性能关系仍可能存在，但胜负关系会受其他因素影响。

![Shape distance vs performance gap]({rel_figs[2]})

## 7. 当前代码中的可比性问题

当前 GPU 代码中，FPR 的实际阶数由以下逻辑决定：

```python
requested = min(n_hidden, SAFE_MAX_ORDER)
for order in range(requested, 0, -1):
    if estimate_fullpr_feature_count(p, order, include_special) <= FULLPR_FEATURE_CAP:
        return order
```

因此，FPR 阶数不会随 `p` 增高而增加。相反，当 p 太高导致完整交互特征数超过上限时，代码会自动降阶。

当前三种深度下的实际 FPR 阶数和特征数为：

{markdown_table(order_display, ["p", "n_hidden", "actual_max_order", "feature_count"])}

这带来一个解释风险：p=10/20/50 的三隐层 NN 对应三阶 FPR，但 p=100/200 的三隐层 NN 对应二阶 FPR。因此高维结果同时混入了两个变化：

1. 输入维度 p 变高；
2. 三隐层设置下 FPR 从三阶被压到二阶。

这会削弱“不同 p 之间三隐层结果可比”的严谨性。

## 8. 推荐的高维可比方案：Screened FPR-3

最可行的方案是：**固定 NN 深度仍为 1/2/3，但把三隐层对应的 FPR 始终保持为三阶；为了避免高维完整三阶组合爆炸，只在筛选出的 k 个变量上生成三阶交互，对所有变量保留主效应和特殊函数特征。**

具体做法：

1. 先用训练集或 pilot subset 选择固定数量的活跃变量，例如 `k=8`。合成实验中可以用真实设定的前 8 个 active variables 做 sanity check；正式论文实验中更好用可观测规则，例如边际相关、Lasso、随机森林重要性或 NN 输入梯度筛选。
2. 对这 k 个变量生成完整三阶多项式交互。
3. 对所有 p 个变量保留一阶主效应。
4. 如果 `special=True`，仍对所有 p 个变量保留 `sin(x_i)` 和 `exp(x_i)`。
5. 这样三隐层设置在 p=10/20/50/100/200 下都是真正的“三阶 FPR”，只是三阶交互被限制在同样规模的候选活跃子空间内。

特征数对比如下：

{markdown_table(comp_display, ["p", "full_degree_3_features", "current_3_hidden_order", f"screened_degree_3_k{N_ACTIVE}_features"])}

为什么这个方案可比：

- 它保证三隐层设置在所有 p 下都对应三阶 FPR，不再出现 p=100/200 被降到二阶的问题。
- 它保持每个 p 下的高阶交互容量大致同一量级，高维增加主要体现在噪声变量和筛选难度，而不是完整三阶特征数爆炸。
- 它适合你的科学问题：你关心的是“NN-FPR 距离能否诊断性能/形状接近”，而不是单纯测试一个无法估计的百万级完整三阶设计矩阵。

![High-p comparability]({rel_figs[3]})

## 9. 建议的论文表述

可以写：

> Across dimensions, cases, activations and network depths, the in-domain NN-FPR response-surface distance is strongly associated with the NN-FPR performance gap. This supports using NN-FPR geometric distance as a practical diagnostic for whether a polynomial surrogate is close to the neural network in both shape and predictive behavior.

中文对应：

> 在不同维度、不同真函数设定、不同激活函数和不同网络深度下，数据区内 NN-FPR 响应曲面距离与 NN-FPR 性能差距稳定正相关。这说明 NN-FPR 几何距离可以作为一个实际可观测的诊断指标，用于判断多项式代理是否在形状和预测行为上接近神经网络。

但不要写成：

> 只要 NN-FPR 距离小，FPR 就一定优于 NN。

更严谨的限定是：

> 距离小通常意味着二者性能差距小；是否 FPR 胜过 NN，还取决于真函数是否落在 FPR 特征族内、FPR 阶数容量、输入维度、样本量、激活函数和 NN 训练稳定性。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = read_raw()
    corr = read_correlations()
    fig_paths = [
        save_dimension_summary(df),
        save_case_order_heatmap(df),
        save_shape_perf_scatter(df),
        save_comparability_plot(),
    ]
    build_report(df, corr, fig_paths)
    print(REPORT_PATH)
    for path in fig_paths:
        print(path)


if __name__ == "__main__":
    main()
