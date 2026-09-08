"""Combine CPU/GPU outputs from iterative_highdim_dopt_experiment.py."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CPU_PREFIX = "highdim_iter_dopt_cpu"
GPU_PREFIX = "highdim_iter_dopt_gpu"
OUT_PREFIX = "highdim_iter_dopt_cpu_gpu"
CASE_LABELS = {
    "highdim_poly2": "Quadratic",
    "highdim_smooth": "Smooth",
    "highdim_strong": "Strong nonlinear",
}


def _slug(value):
    return str(value).replace(".", "p").replace("-", "m").replace(",", "_")


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
    for metric, random_metric, ylabel, suffix in [
        ("dopt_mse_vs_nn", "random_mse_median", "MSE vs NN oracle", "mse_curve"),
        ("mse_gain_vs_random_median_pct", None,
         "D-optimal gain vs random median MSE (%)", "gain_curve"),
    ]:
        for device in devices:
            for p in sorted(df["p"].unique()):
                part = df[(df["device"] == device) & (df["p"] == p)]
                for case, sub in part.groupby("case"):
                    sub = sub.sort_values("iteration")
                    if sub.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(6.2, 4.8))
                    label = CASE_LABELS.get(case, case)
                    ax.plot(sub["selected_n"], sub[metric], marker="o",
                            label="D-optimal")
                    if random_metric:
                        ax.plot(sub["selected_n"], sub[random_metric],
                                marker="x", linestyle="--", alpha=0.75,
                                label="Random median")
                    if metric.endswith("pct"):
                        ax.axhline(0, color="#555555", linewidth=1)
                    ax.set_title(f"{device.upper()} / {label} / p={p}")
                    ax.set_xlabel("Selected training points")
                    ax.set_ylabel(ylabel)
                    ax.grid(True, linestyle=":", alpha=0.65)
                    ax.legend(frameon=False, fontsize=8)
                    fig.tight_layout()
                    fig.savefig(
                        f"{OUT_PREFIX}_{device}_{_slug(case)}_p{int(p)}_{suffix}.png",
                        dpi=220,
                    )
                    plt.close(fig)


def write_report(df, summary, diff):
    final_positive = int((summary["mse_gain_vs_random_median_pct"] > 0).sum())
    total = len(summary)
    gain_corr = diff["mse_gain_vs_random_median_pct_cpu"].corr(
        diff["mse_gain_vs_random_median_pct_gpu"])
    avg_abs_gain_diff = diff[
        "mse_gain_vs_random_median_pct_gpu_minus_cpu"].abs().mean()
    n_matched = len(diff)
    feature_rows = (
        summary[["p", "n_fullpr_features", "n_design_params"]]
        .drop_duplicates()
        .sort_values("p")
    )
    feature_note = "；".join(
        f"p={int(row.p)} 时 FullPR={int(row.n_fullpr_features)}、"
        f"D-opt参数={int(row.n_design_params)}"
        for row in feature_rows.itertuples(index=False)
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
        f"最后一轮 {n_matched} 个匹配组合上，CPU 与 GPU 的 MSE gain 平均绝对差为 `{avg_abs_gain_diff:.2f}` 个百分点；相关系数为 `{gain_corr:.3f}`。这个相关值应结合匹配组合数量一起解读；更重要的是观察 CPU/GPU 的收益符号和排序是否一致。收益幅度总体不大时，说明高维下 D-optimal 的主要作用更接近稳健改善，而不是压倒性改善。",
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
        f"不同维度下需要把初始样本数、最终选点数和 FullPR 特征规模一起看。当前特征规模为：{feature_note}。当初始设计集已经超过特征数时，D-optimal 相对随机的提升通常较温和；当初始设计集低于特征数时，前几轮可能欠定或波动，随后要看升级后的设计集继续进入下一轮 D-optimal 后是否稳定。",
        "",
        f"最后一轮共有 {final_positive}/{total} 个 CPU/GPU 组合中 D-optimal 优于随机中位数。这个结果支持“升级后的数据集可以继续用于下一轮 D-optimal 优化”的流程，但也说明高维时需要足够迭代预算，尤其要让选点数尽量接近或超过对应维度下的 FullPR 特征数。",
        "",
        "## 输出文件",
        "",
        f"- `{OUT_PREFIX}_results.csv`：CPU/GPU 全部迭代结果。",
        f"- `{OUT_PREFIX}_summary.csv`：最后一轮汇总。",
        f"- `{OUT_PREFIX}_device_diff.csv`：CPU/GPU 差异。",
        f"- `{OUT_PREFIX}_<device>_<case>_p<p>_mse_curve.png`：每个 device/case/p 单独保存的误差随迭代变化。",
        f"- `{OUT_PREFIX}_<device>_<case>_p<p>_gain_curve.png`：每个 device/case/p 单独保存的相对随机收益随迭代变化。",
    ])
    Path(f"{OUT_PREFIX}_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    combine()
