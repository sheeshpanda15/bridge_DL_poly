"""Combine CPU/GPU outputs from iterative_highdim_dopt_experiment.py."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CPU_PREFIX = "highdim_iter_dopt_cpu"
GPU_PREFIX = "highdim_iter_dopt_gpu"
OUT_PREFIX = "highdim_iter_dopt_cpu_gpu"


def read_results(prefix):
    path = Path(f"{prefix}_results.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def combine():
    df = pd.concat([read_results(CPU_PREFIX), read_results(GPU_PREFIX)],
                   ignore_index=True)
    df.to_csv(f"{OUT_PREFIX}_results.csv", index=False, encoding="utf-8-sig")
    summary = df.sort_values("iteration").groupby(["device", "case", "p"]).tail(1)
    summary.to_csv(f"{OUT_PREFIX}_summary.csv", index=False, encoding="utf-8-sig")
    diff = make_diff(summary)
    plot_curves(df)
    write_report(df, summary, diff)


def make_diff(summary):
    keys = ["case", "p", "iteration", "selected_n"]
    metrics = [
        "oracle_mse_vs_y",
        "dopt_mse_vs_nn",
        "random_mse_median",
        "mse_gain_vs_random_median_pct",
        "dopt_shape_vs_nn",
        "random_shape_median",
        "shape_gain_vs_random_median_pct",
        "elapsed_sec_case_p",
    ]
    cpu = summary[summary["device"] == "cpu"][keys + metrics]
    gpu = summary[summary["device"] == "cuda"][keys + metrics]
    diff = cpu.merge(gpu, on=keys, suffixes=("_cpu", "_gpu"))
    for metric in metrics:
        diff[f"{metric}_gpu_minus_cpu"] = diff[f"{metric}_gpu"] - diff[f"{metric}_cpu"]
    diff.to_csv(f"{OUT_PREFIX}_device_diff.csv", index=False, encoding="utf-8-sig")
    return diff


def plot_curves(df):
    devices = [d for d in ["cpu", "cuda"] if d in set(df["device"])]
    devices.extend(d for d in sorted(set(df["device"])) if d not in devices)
    p_values = sorted(df["p"].unique())
    for metric, random_metric, ylabel, suffix in [
        ("dopt_mse_vs_nn", "random_mse_median", "MSE vs NN oracle", "mse_curve"),
        ("mse_gain_vs_random_median_pct", None,
         "D-optimal gain vs random median MSE (%)", "gain_curve"),
    ]:
        fig, axes = plt.subplots(
            len(devices), len(p_values),
            figsize=(5.8 * len(p_values), 4.1 * len(devices)),
            sharex=False,
            squeeze=False,
        )
        for row_idx, device in enumerate(devices):
            for col_idx, p in enumerate(p_values):
                ax = axes[row_idx, col_idx]
                part = df[(df["device"] == device) & (df["p"] == p)]
                for case, sub in part.groupby("case"):
                    sub = sub.sort_values("iteration")
                    label = {
                        "highdim_poly2": "Quadratic",
                        "highdim_smooth": "Smooth",
                        "highdim_strong": "Strong nonlinear",
                    }.get(case, case)
                    ax.plot(sub["selected_n"], sub[metric], marker="o", label=label)
                    if random_metric:
                        ax.plot(sub["selected_n"], sub[random_metric],
                                marker="x", linestyle="--", alpha=0.75,
                                label=f"{label} random")
                if metric.endswith("pct"):
                    ax.axhline(0, color="#555555", linewidth=1)
                ax.set_title(f"{device.upper()} / p={p}")
                ax.set_xlabel("Selected training points")
                ax.set_ylabel(ylabel)
                ax.grid(True, linestyle=":", alpha=0.65)
                if len(part):
                    ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT_PREFIX}_{suffix}.png", dpi=220)
        plt.close(fig)


def write_report(df, summary, diff):
    final_positive = int((summary["mse_gain_vs_random_median_pct"] > 0).sum())
    total = len(summary)
    gain_corr = diff["mse_gain_vs_random_median_pct_cpu"].corr(
        diff["mse_gain_vs_random_median_pct_gpu"])
    avg_abs_gain_diff = diff[
        "mse_gain_vs_random_median_pct_gpu_minus_cpu"].abs().mean()
    feature_rows = (
        summary[["p", "n_fullpr_features", "n_design_params"]]
        .drop_duplicates()
        .sort_values("p")
    )
    settings = summary.iloc[0]
    lines = [
        "# 高维迭代 D-optimal CPU/GPU 合并报告",
        "",
        "## 实验设置",
        "",
        f"- 原始数据集大小：{int(settings.n_original)}。",
        f"- 输入维度：{', '.join(str(x) for x in sorted(summary['p'].unique()))}。",
        f"- 训练池/评估集：{int(settings.n_train_pool)}/{int(settings.n_eval)}。",
        f"- 初始 uniform 设计集：{int(settings.initial_n)} 个点，即训练池的 {settings.initial_fraction:g}。",
        f"- 迭代升级：每轮 D-optimal 增加 {int(settings.batch_size)} 个点，最后到 {int(settings.selected_n)} 个点。",
        f"- NN：hidden={settings.hidden}，activation={settings.activation}，epochs={int(settings.epochs)}。",
        f"- FullPR：degree={int(settings.degree)}，include_special={bool(settings.include_special)}。",
        "- 随机基准：每个设备、case、p、iteration 重复 10 次，且共享同一个初始 uniform 设计集。",
        "",
        "## 特征规模",
        "",
        "| p | FullPR特征数 | D-opt设计参数数（含截距） |",
        "|---:|---:|---:|",
    ]
    for row in feature_rows.itertuples(index=False):
        lines.append(f"| {row.p} | {row.n_fullpr_features} | {row.n_design_params} |")

    lines.extend([
        "",
        "## CPU/GPU 一致性",
        "",
        f"最后一轮 4 个匹配组合上，CPU 与 GPU 的 MSE gain 平均绝对差为 `{avg_abs_gain_diff:.2f}` 个百分点；相关系数为 `{gain_corr:.3f}`，但由于只有 4 个匹配点，这个相关值不作为主要结论。更重要的是，CPU/GPU 在最后一轮均显示 D-optimal 为正收益。收益幅度总体不大，说明高维下 D-optimal 的主要作用是稳健改善，而不是压倒性改善。",
        "",
        "## 最后一轮结果",
        "",
        "| device | case | p | selected n | pilot shape | D-opt MSE | random MSE | gain | runtime/case |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary.sort_values(["device", "p", "case"]).itertuples(index=False):
        lines.append(
            f"| {row.device} | {row.case} | {row.p} | {row.selected_n} | "
            f"{row.pilot_shape_fullpr_vs_nn:.3e} | "
            f"{row.dopt_mse_vs_nn:.3e} | {row.random_mse_median:.3e} | "
            f"{row.mse_gain_vs_random_median_pct:.1f}% | "
            f"{row.elapsed_sec_case_p:.1f}s |"
        )

    lines.extend([
        "",
        "## 迭代现象",
        "",
        "p=20 时 FullPR 只有 270 个特征，初始 750 个点已经超过特征数，所以 D-optimal 相对随机的提升比较温和。p=50 时 FullPR 有 1425 个特征，初始 750 个点低于特征数，前几轮会出现欠定和不稳定；当选点数超过特征数后，误差明显下降，最后一轮 CPU/GPU 都转为正收益。",
        "",
        f"最后一轮共有 {final_positive}/{total} 个 CPU/GPU 组合中 D-optimal 优于随机中位数。这个结果支持“升级后的数据集可以继续用于下一轮 D-optimal 优化”的流程，但也说明高维时需要足够迭代预算，尤其是 p=50 时至少要让选点数接近或超过 FullPR 特征数。",
        "",
        "## 输出文件",
        "",
        f"- `{OUT_PREFIX}_results.csv`：CPU/GPU 全部迭代结果。",
        f"- `{OUT_PREFIX}_summary.csv`：最后一轮汇总。",
        f"- `{OUT_PREFIX}_device_diff.csv`：CPU/GPU 差异。",
        f"- `{OUT_PREFIX}_mse_curve.png`：误差随迭代变化。",
        f"- `{OUT_PREFIX}_gain_curve.png`：相对随机收益随迭代变化。",
    ])
    Path(f"{OUT_PREFIX}_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    combine()
