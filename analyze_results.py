"""
analyze_results.py
─────────────────────────────────────────────────────────────────
独立分析脚本：读取已有的 results_raw.csv（无需重新训练），做三件事
  1. 数据清洗：剔除未收敛的 NN run、标记 LayerTaylor-PR 的爆炸失效
  2. 稳健重绘：坐标轴按分位数裁剪，LTPR 的 MSE 用 log 轴，主体数据看得清
  3. 失效率统计：单独报告 LayerTaylor-PR 在各配置下的失效比例

为什么需要它：
  - 逐层泰勒展开（LayerTaylor-PR）在突触电位冲出泰勒收敛域时会灾难性发散，
    MSE 可达 1e8。这些爆炸值若直接画图，会把所有正常点压成贴地的一条线，
    且污染相关性。本脚本把"失效"识别出来单独统计，而非混入数值分析。
  - 深层网络偶发不收敛，NN_mse 远超同组其他 run，也需剔除。

用法：
  python analyze_results.py                      # 读 results_raw.csv
  python analyze_results.py --csv my_raw.csv     # 指定文件
  python analyze_results.py --nn-thresh 10 --ltpr-thresh 100
"""

import argparse
import math

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ACTS = ["softplus", "tanh", "sigmoid"]
COLORS = {"softplus": "#2a9d8f", "tanh": "#e76f51", "sigmoid": "#264653"}

COMPARE_MODELS = [
    ("FPR", "FullPR", "FPR_mse_vs_y"),
    ("TPR", "Taylor-PR", "TPR_mse_vs_y"),
    ("LTPR", "LayerTaylor-PR", "LTPR_mse_vs_y"),
]


# ─────────────────────────────────────────────
# 1. 数据清洗
# ─────────────────────────────────────────────
def clean(df, nn_thresh=10.0, ltpr_thresh=100.0):
    """
    返回 (df_clean, report)。
      - NN 未收敛：组内（case×activation×hidden）NN_mse > nn_thresh × 组中位数 → 剔除
      - LTPR 失效：LTPR_mse_vs_NN > ltpr_thresh → 标记 ltpr_failed=True（不剔除整行，
        只在涉及 LTPR 的分析里排除该值），同时用于失效率统计
    """
    df = df.copy()
    report = {}

    # —— NN 未收敛剔除 ——
    grp = df.groupby(["case", "activation", "hidden"])["NN_mse_vs_y"]
    med = grp.transform("median")
    nn_bad = df["NN_mse_vs_y"] > nn_thresh * med.clip(lower=1e-12)
    report["nn_unconverged_removed"] = int(nn_bad.sum())
    df = df[~nn_bad].copy()

    # —— LTPR 失效标记 ——
    if "LTPR_mse_vs_NN" in df.columns:
        df["ltpr_failed"] = df["LTPR_mse_vs_NN"] > ltpr_thresh
        report["ltpr_failed_count"] = int(df["ltpr_failed"].sum())
        report["ltpr_total_valid"] = int(df["LTPR_mse_vs_NN"].notna().sum())
    else:
        df["ltpr_failed"] = False
        report["ltpr_failed_count"] = 0
        report["ltpr_total_valid"] = 0

    return df, report


def ltpr_failure_table(df):
    """按配置统计 LayerTaylor-PR 失效率。"""
    if "ltpr_failed" not in df.columns:
        return pd.DataFrame()
    g = df.groupby(["case", "activation", "hidden"]).agg(
        n=("ltpr_failed", "size"),
        n_failed=("ltpr_failed", "sum"))
    g["fail_rate"] = (g["n_failed"] / g["n"]).round(3)
    return g.reset_index()


def add_nn_baseline_columns(df):
    """Add model-vs-NN deltas. Positive improvement means the model beats NN."""
    df = df.copy()
    if "NN_mse_vs_y" not in df.columns:
        return df
    nn = df["NN_mse_vs_y"].replace(0, np.nan)
    for short, _name, mse_col in COMPARE_MODELS:
        if mse_col not in df.columns:
            continue
        df[f"{short}_mse_delta_vs_NN"] = df[mse_col] - df["NN_mse_vs_y"]
        df[f"{short}_improve_vs_NN_pct"] = 100.0 * (df["NN_mse_vs_y"] - df[mse_col]) / nn
        df[f"{short}_better_than_NN"] = df[mse_col] < df["NN_mse_vs_y"]
    return df


def nn_baseline_table(df):
    """Summarize who beats NN for each comparable model."""
    recs = []
    for case, gdf in [("ALL", df)] + [(c, df[df["case"] == c]) for c in sorted(df["case"].unique())]:
        for short, name, mse_col in COMPARE_MODELS:
            pct_col = f"{short}_improve_vs_NN_pct"
            win_col = f"{short}_better_than_NN"
            if pct_col not in gdf.columns or win_col not in gdf.columns:
                continue
            m = gdf[[pct_col, win_col]].dropna()
            if len(m) == 0:
                continue
            recs.append({
                "case": case,
                "model": name,
                "n": len(m),
                "win_rate_vs_NN": float(m[win_col].mean()),
                "median_improve_vs_NN_pct": float(m[pct_col].median()),
                "mean_improve_vs_NN_pct": float(m[pct_col].mean()),
            })
    return pd.DataFrame(recs)


# ─────────────────────────────────────────────
# 2. 稳健绘图工具
# ─────────────────────────────────────────────
def _qlim(series, lo=0.01, hi=0.99):
    """按分位数给出坐标轴范围，避免极端值拉伸。"""
    s = series.dropna()
    if len(s) < 5:
        return None
    a, b = s.quantile(lo), s.quantile(hi)
    if a == b:
        return None
    pad = (b - a) * 0.05
    return (a - pad, b + pad)


def scatter(ax, df, x, y, logy=False, clip=True, drop_ltpr_fail=False):
    sub = df
    if drop_ltpr_fail and "ltpr_failed" in df.columns:
        sub = df[~df["ltpr_failed"]]
    for a in ACTS:
        s = sub[sub["activation"] == a][[x, y]].dropna()
        if logy:
            s = s[s[y] > 0]
        ax.scatter(s[x], s[y], s=14, alpha=0.5, color=COLORS[a], label=a)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    if logy:
        ax.set_yscale("log")
    elif clip:
        yl = _qlim(sub[y])
        if yl:
            ax.set_ylim(yl)
    if clip:
        xl = _qlim(sub[x])
        if xl:
            ax.set_xlim(xl)


# ─────────────────────────────────────────────
# 3. 按数据集出图
# ─────────────────────────────────────────────
def plots_for_case(df, case, out_prefix):
    sub = df[df["case"] == case]
    if len(sub) == 0:
        return

    # 图A：核心几何关系（清洗 + 裁剪）
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    scatter(ax[0, 0], sub, "act_io_scaled_sum", "shape_NN_FPR_in")
    ax[0, 0].set_title("act-change(sum) -> NN-FullPR closeness(in)", fontsize=9)
    scatter(ax[0, 1], sub, "act_io_scaled_sum", "shape_NN_FPR_ext")
    ax[0, 1].set_title("act-change(sum) -> closeness(extrap)", fontsize=9)
    scatter(ax[0, 2], sub, "shape_NN_FPR_in", "abs_rmse_gap")
    ax[0, 2].set_title("NN-FullPR closeness -> perf gap (CORE)", fontsize=9)
    scatter(ax[1, 0], sub, "NN_mse_vs_y", "FPR_mse_vs_y")
    lim = max(_qlim(sub["NN_mse_vs_y"])[1] if _qlim(sub["NN_mse_vs_y"]) else 1,
              _qlim(sub["FPR_mse_vs_y"])[1] if _qlim(sub["FPR_mse_vs_y"]) else 1)
    ax[1, 0].plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
    ax[1, 0].set_title("NN MSE vs FullPR MSE (y=x ref)", fontsize=9)
    scatter(ax[1, 1], sub, "shape_NN_LTPR", "abs_rmse_gap", drop_ltpr_fail=True)
    ax[1, 1].set_title("NN-LayerTaylorPR closeness -> perf gap (LTPR ok only)", fontsize=9)
    scatter(ax[1, 2], sub, "shape_NN_LTPR", "shape_NN_FPR_in", drop_ltpr_fail=True)
    ax[1, 2].set_title("LayerTaylorPR vs FullPR closeness (LTPR ok only)", fontsize=9)
    fig.suptitle(f"[{case}] Cleaned geometry & performance", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_{case}_clean_geom.png", dpi=130)
    plt.close(fig)

    # 图B：LTPR 的 MSE 用 log 轴展示其失效尾部
    if "LTPR_mse_vs_NN" in sub.columns and sub["LTPR_mse_vs_NN"].notna().any():
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        scatter(ax[0], sub, "shape_NN_LTPR", "LTPR_mse_vs_NN", logy=True, clip=False)
        ax[0].set_title("LTPR MSE vs NN (log) - failure tail visible", fontsize=9)
        scatter(ax[1], sub, "act_io_scaled_sum", "LTPR_mse_vs_NN", logy=True, clip=False)
        ax[1].set_title("act-change -> LTPR MSE (log)", fontsize=9)
        fig.suptitle(f"[{case}] LayerTaylor-PR failure (log scale)", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{out_prefix}_{case}_ltpr_failure.png", dpi=130)
        plt.close(fig)

    plot_nn_baseline(sub, f"{out_prefix}_{case}_nn_baseline.png",
                     f"[{case}] Models vs NN baseline")


def plot_nn_baseline(sub, path, title):
    rows = []
    for short, name, _mse_col in COMPARE_MODELS:
        pct_col = f"{short}_improve_vs_NN_pct"
        win_col = f"{short}_better_than_NN"
        if pct_col not in sub.columns or win_col not in sub.columns:
            continue
        for act in ACTS:
            s = sub[sub["activation"] == act][[pct_col, win_col]].dropna()
            if len(s) == 0:
                continue
            rows.append((name, act, float(s[win_col].mean() * 100.0),
                         float(s[pct_col].median()), len(s)))
    if not rows:
        return

    models = list(dict.fromkeys(r[0] for r in rows))
    x = np.arange(len(models))
    width = 0.24
    offsets = np.linspace(-width, width, len(ACTS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for i, act in enumerate(ACTS):
        vals_win, vals_imp = [], []
        for model in models:
            hit = [r for r in rows if r[0] == model and r[1] == act]
            vals_win.append(hit[0][2] if hit else np.nan)
            vals_imp.append(hit[0][3] if hit else np.nan)
        axes[0].bar(x + offsets[i], vals_win, width=width, color=COLORS[act],
                    alpha=0.75, label=act)
        axes[1].bar(x + offsets[i], vals_imp, width=width, color=COLORS[act],
                    alpha=0.75, label=act)

    axes[0].axhline(50, color="k", lw=0.8, ls="--", alpha=0.5)
    axes[0].set_ylabel("win rate vs NN (%)")
    axes[0].set_title("How often model MSE < NN MSE", fontsize=10)
    axes[1].axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    axes[1].set_ylabel("median improvement vs NN (%)")
    axes[1].set_title("Positive means better than NN", fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────
# 4. 清洗后的相关性（稳健：Spearman 对单调失效不敏感，但仍排除 NN 未收敛）
# ─────────────────────────────────────────────
def correlations(df):
    pairs = [
        ("act_io_scaled_sum", "shape_NN_FPR_in",  "act-change(sum) -> NN-FullPR closeness(in)"),
        ("act_io_scaled_sum", "shape_NN_FPR_ext", "act-change(sum) -> closeness(extrap)"),
        ("shape_NN_FPR_in",   "abs_rmse_gap",     "NN-FullPR closeness -> perf gap"),
        ("shape_NN_LTPR",     "abs_rmse_gap",     "NN-LayerTaylorPR closeness -> perf gap"),
        ("shape_NN_LTPR",     "shape_NN_FPR_in",  "LayerTaylorPR vs FullPR closeness"),
    ]
    recs = []
    for gname, gdf in [("ALL", df)] + [(c, df[df.case == c]) for c in sorted(df.case.unique())]:
        for x, y, label in pairs:
            m = gdf.copy()
            # 涉及 LTPR 的关系排除失效值
            if "LTPR" in label and "ltpr_failed" in m.columns:
                m = m[~m["ltpr_failed"]]
            m = m[[x, y]].dropna()
            if len(m) >= 5 and m[x].std() > 0 and m[y].std() > 0:
                rho, p = spearmanr(m[x], m[y])
            else:
                rho, p = float("nan"), float("nan")
            recs.append({"group": gname, "relation": label,
                         "spearman_rho": rho, "p_value": p, "n": len(m)})
    return pd.DataFrame(recs)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="清洗并重绘 results_raw.csv（无需重训）")
    ap.add_argument("--csv", default="results_raw.csv")
    ap.add_argument("--out-prefix", default="clean")
    ap.add_argument("--nn-thresh", type=float, default=10.0,
                    help="NN_mse 超过组中位数的几倍判为未收敛")
    ap.add_argument("--ltpr-thresh", type=float, default=100.0,
                    help="LTPR_mse_vs_NN 超过此值判为泰勒展开失效")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"读入 {args.csv}：{len(df)} 行")

    df_clean, report = clean(df, args.nn_thresh, args.ltpr_thresh)
    df_clean = add_nn_baseline_columns(df_clean)
    print(f"\n清洗报告：")
    print(f"  剔除未收敛 NN run：{report['nn_unconverged_removed']} 个")
    print(f"  LayerTaylor-PR 失效（MSE>{args.ltpr_thresh}）："
          f"{report['ltpr_failed_count']}/{report['ltpr_total_valid']}")
    print(f"  清洗后剩余：{len(df_clean)} 行")

    # 失效率表
    ft = ltpr_failure_table(df_clean)
    if len(ft):
        print(f"\nLayerTaylor-PR 失效率（前 10 高）：")
        top = ft.sort_values("fail_rate", ascending=False).head(10)
        print(top.to_string(index=False))
        ft.to_csv(f"{args.out_prefix}_ltpr_failure_rate.csv",
                  index=False, encoding="utf-8-sig")

    bt = nn_baseline_table(df_clean)
    if len(bt):
        print(f"\nModels vs NN baseline:")
        print(bt.to_string(index=False))
        bt.to_csv(f"{args.out_prefix}_nn_baseline_summary.csv",
                  index=False, encoding="utf-8-sig")

    # 相关性
    corr = correlations(df_clean)
    print(f"\n清洗后相关性（Spearman ρ；LTPR 关系已排除失效值）：")
    prev = None
    for _, c in corr.iterrows():
        if prev is not None and c.group != prev:
            print("  " + "-" * 80)
        prev = c.group
        sig = "*" if (not math.isnan(c.p_value) and c.p_value < 0.05) else " "
        rho = f"{c.spearman_rho:.3f}" if not math.isnan(c.spearman_rho) else "—"
        p = f"{c.p_value:.2g}" if not math.isnan(c.p_value) else "—"
        print(f"  {c.group:<22}{c.relation:<46}{rho:>7}{sig} p={p:>9} n={c.n}")
    corr.to_csv(f"{args.out_prefix}_correlations.csv", index=False, encoding="utf-8-sig")

    # 出图
    for case in sorted(df_clean["case"].unique()):
        plots_for_case(df_clean, case, args.out_prefix)
    print(f"\n已输出：")
    print(f"  {args.out_prefix}_<case>_clean_geom.png   （每数据集：清洗后核心关系）")
    print(f"  {args.out_prefix}_<case>_ltpr_failure.png （每数据集：LTPR 失效 log 图）")
    print(f"  {args.out_prefix}_<case>_nn_baseline.png  （每数据集：以 NN 为基准的胜负图）")
    print(f"  {args.out_prefix}_correlations.csv")
    print(f"  {args.out_prefix}_ltpr_failure_rate.csv")
    print(f"  {args.out_prefix}_nn_baseline_summary.csv")
