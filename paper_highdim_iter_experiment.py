"""
Paper-scale high-dimensional iterative D-optimal experiment.

This runner wraps iterative_highdim_dopt_experiment.py with multiple data seeds,
multiple NN initialization seeds, multiple initial uniform fractions, and both
CPU/GPU devices. It writes raw rows, aggregate statistics, plots, and a Chinese
report suitable for the paper draft.
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

from iterative_highdim_dopt_experiment import (
    CASE_LABELS,
    batch_dopt_select,
    build_fullpr_features,
    design_matrix_for_dopt,
    fit_surrogate,
    get_device,
    limited_shape_distance,
    make_highdim_dataset,
    nn_predict,
    pilot_distance,
    random_baseline,
    train_oracle_nn,
)


CASE_ORDER = ["highdim_poly2", "highdim_smooth", "highdim_strong"]


def stable_seed(base, device_name, case, p, data_seed, init_seed, fraction):
    device_offset = 0 if device_name == "cpu" else 500_000
    case_offset = (CASE_ORDER.index(case) + 1) * 10_000 if case in CASE_ORDER else 90_000
    frac_offset = int(round(fraction * 100_000))
    return (
        base
        + device_offset
        + case_offset
        + int(p) * 100
        + int(data_seed) * 1_000
        + int(init_seed) * 10
        + frac_offset
    )


def summarize_random_count(args):
    n_devices = len(args.devices)
    n_oracles = (
        n_devices
        * len(args.p_values)
        * len(args.cases)
        * len(args.data_seeds)
        * len(args.init_seeds)
    )
    n_trajectories = n_oracles * len(args.initial_fractions)
    n_iter_points = args.iterations + 1
    return {
        "n_oracles": n_oracles,
        "n_trajectories": n_trajectories,
        "n_dopt_fits": n_trajectories * n_iter_points,
        "n_random_fits": n_trajectories * n_iter_points * args.random_repeats,
    }


def run_design_path(case, p, device_name, data_seed, init_seed, initial_fraction,
                    shared, args):
    X_train = shared["X_train"]
    X_test = shared["X_test"]
    y_test = shared["y_test"]
    F_train = shared["F_train"]
    F_eval = shared["F_eval"]
    Z = shared["Z"]
    pred_oracle_train = shared["pred_oracle_train"]
    pred_oracle_eval = shared["pred_oracle_eval"]
    oracle_mse_vs_y = shared["oracle_mse_vs_y"]

    rng = np.random.default_rng(
        stable_seed(args.seed, device_name, case, p, data_seed, init_seed,
                    initial_fraction)
    )
    initial_n = max(int(round(len(X_train) * initial_fraction)), args.min_initial)
    initial_n = min(initial_n, len(X_train))
    initial_idx = rng.choice(len(X_train), size=initial_n, replace=False)

    selected_mask = np.zeros(len(X_train), dtype=bool)
    selected_mask[initial_idx] = True
    pilot = pilot_distance(
        F_train, pred_oracle_train, X_train, initial_idx,
        args.ridge_alpha, rng, args.shape_points)

    rows = []
    path_start = time.perf_counter()
    for iteration in range(args.iterations + 1):
        selected_idx = np.flatnonzero(selected_mask)
        selected_n = len(selected_idx)
        pred_sur, _ = fit_surrogate(
            F_train[selected_idx], pred_oracle_train[selected_idx],
            F_eval, args.ridge_alpha)
        dopt_mse = mean_squared_error(pred_oracle_eval, pred_sur)
        dopt_shape = limited_shape_distance(
            X_test, pred_oracle_eval, pred_sur, rng, args.shape_points)
        dopt_mse_vs_y = mean_squared_error(y_test, pred_sur)
        rb = random_baseline(
            F_train, pred_oracle_train, F_eval, pred_oracle_eval, X_test,
            initial_idx, selected_n, args.random_repeats, args.ridge_alpha,
            rng, args.shape_points)

        rows.append({
            "case": case,
            "case_label": CASE_LABELS.get(case, case),
            "p": p,
            "device": device_name,
            "cuda_name": shared["cuda_name"],
            "data_seed": data_seed,
            "init_seed": init_seed,
            "n_original": args.n,
            "n_train_pool": len(X_train),
            "n_eval": len(X_test),
            "hidden": ",".join(str(x) for x in args.hidden),
            "activation": args.activation,
            "epochs": args.epochs,
            "degree": args.degree,
            "include_special": args.include_special,
            "n_fullpr_features": F_train.shape[1],
            "n_design_params": Z.shape[1],
            "initial_fraction": initial_fraction,
            "initial_n": initial_n,
            "batch_size": args.batch_size,
            "iteration": iteration,
            "selected_n": selected_n,
            "oracle_mse_vs_y": oracle_mse_vs_y,
            "surrogate_mse_vs_y": dopt_mse_vs_y,
            "dopt_mse_vs_nn": dopt_mse,
            "dopt_shape_vs_nn": dopt_shape,
            **pilot,
            **rb,
            "mse_gain_vs_random_median_pct": (
                100.0 * (rb["random_mse_median"] - dopt_mse)
                / rb["random_mse_median"]
            ),
            "shape_gain_vs_random_median_pct": (
                100.0 * (rb["random_shape_median"] - dopt_shape)
                / rb["random_shape_median"]
            ),
            "elapsed_sec_path": time.perf_counter() - path_start,
        })

        if iteration == args.iterations:
            break
        add_idx = batch_dopt_select(
            Z, selected_mask, args.batch_size, args.dopt_ridge,
            args.dopt_chunk_size)
        selected_mask[add_idx] = True
    return rows


def run_oracle(case, p, data_seed, init_seed, device_name, args):
    device = get_device(device_name)
    X_train, X_test, y_train, y_test = make_highdim_dataset(
        case, args.n, p, data_seed, args.noise_sd)
    model = train_oracle_nn(
        X_train, y_train, X_test, y_test, tuple(args.hidden),
        args.activation, args.epochs, init_seed, device)
    pred_oracle_train = nn_predict(model, X_train, device)
    pred_oracle_eval = nn_predict(model, X_test, device)
    F_train, F_eval = build_fullpr_features(
        X_train, X_test, args.degree, args.include_special)
    Z = design_matrix_for_dopt(F_train)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "F_train": F_train,
        "F_eval": F_eval,
        "Z": Z,
        "pred_oracle_train": pred_oracle_train,
        "pred_oracle_eval": pred_oracle_eval,
        "oracle_mse_vs_y": mean_squared_error(y_test, pred_oracle_eval),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
    }


def aggregate_results(df):
    group_cols = [
        "device", "case", "case_label", "p", "initial_fraction", "iteration",
        "selected_n", "n_original", "n_train_pool", "n_eval", "hidden",
        "activation", "epochs", "degree", "include_special",
        "n_fullpr_features", "n_design_params", "initial_n", "batch_size",
    ]
    metric_cols = [
        "oracle_mse_vs_y",
        "pilot_mse_fullpr_vs_nn",
        "pilot_shape_fullpr_vs_nn",
        "dopt_mse_vs_nn",
        "random_mse_median",
        "mse_gain_vs_random_median_pct",
        "dopt_shape_vs_nn",
        "random_shape_median",
        "shape_gain_vs_random_median_pct",
    ]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    for metric in metric_cols:
        count = agg[f"{metric}_count"].clip(lower=1)
        agg[f"{metric}_sem"] = agg[f"{metric}_std"].fillna(0.0) / np.sqrt(count)
        agg[f"{metric}_ci95"] = 1.96 * agg[f"{metric}_sem"]

    pos = (
        df.assign(positive_gain=df["mse_gain_vs_random_median_pct"] > 0)
        .groupby(group_cols)["positive_gain"]
        .mean()
        .reset_index(name="positive_rate")
    )
    return agg.merge(pos, on=group_cols, how="left")


def plot_mean_curves(agg, out_prefix):
    for device in sorted(agg["device"].unique()):
        part = agg[agg["device"] == device]
        fig, axes = plt.subplots(
            len(sorted(part["p"].unique())),
            len(sorted(part["initial_fraction"].unique())),
            figsize=(13.5, 8.2),
            squeeze=False,
            sharex=False,
        )
        for i, p in enumerate(sorted(part["p"].unique())):
            for j, frac in enumerate(sorted(part["initial_fraction"].unique())):
                ax = axes[i, j]
                sub = part[(part["p"] == p) & (part["initial_fraction"] == frac)]
                for case in CASE_ORDER:
                    line = sub[sub["case"] == case].sort_values("selected_n")
                    if line.empty:
                        continue
                    x = line["selected_n"].to_numpy(dtype=float)
                    y = line["mse_gain_vs_random_median_pct_mean"].to_numpy(dtype=float)
                    ci = line["mse_gain_vs_random_median_pct_ci95"].to_numpy(dtype=float)
                    label = CASE_LABELS.get(case, case)
                    ax.plot(x, y, marker="o", label=label)
                    ax.fill_between(x, y - ci, y + ci, alpha=0.16)
                ax.axhline(0, color="#555555", linewidth=1)
                ax.set_title(f"{device.upper()} / p={p} / init={frac:g}")
                ax.set_xlabel("Selected training points")
                ax.set_ylabel("Gain vs random median MSE (%)")
                ax.grid(True, linestyle=":", alpha=0.65)
                ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_{device}_gain_mean_ci.png", dpi=220)
        plt.close(fig)

    for device in sorted(agg["device"].unique()):
        part = agg[agg["device"] == device]
        fig, axes = plt.subplots(
            len(sorted(part["p"].unique())),
            len(sorted(part["initial_fraction"].unique())),
            figsize=(13.5, 8.2),
            squeeze=False,
            sharex=False,
        )
        for i, p in enumerate(sorted(part["p"].unique())):
            for j, frac in enumerate(sorted(part["initial_fraction"].unique())):
                ax = axes[i, j]
                sub = part[(part["p"] == p) & (part["initial_fraction"] == frac)]
                for case in CASE_ORDER:
                    line = sub[sub["case"] == case].sort_values("selected_n")
                    if line.empty:
                        continue
                    x = line["selected_n"].to_numpy(dtype=float)
                    y = line["dopt_mse_vs_nn_mean"].to_numpy(dtype=float)
                    ci = line["dopt_mse_vs_nn_ci95"].to_numpy(dtype=float)
                    yr = line["random_mse_median_mean"].to_numpy(dtype=float)
                    label = CASE_LABELS.get(case, case)
                    ax.plot(x, y, marker="o", label=label)
                    ax.fill_between(x, y - ci, y + ci, alpha=0.16)
                    ax.plot(x, yr, linestyle="--", alpha=0.7,
                            label=f"{label} random")
                ax.set_title(f"{device.upper()} / p={p} / init={frac:g}")
                ax.set_xlabel("Selected training points")
                ax.set_ylabel("MSE vs NN oracle")
                ax.grid(True, linestyle=":", alpha=0.65)
                ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_{device}_mse_mean_ci.png", dpi=220)
        plt.close(fig)


def plot_final_boxplots(df, out_prefix):
    final_iter = int(df["iteration"].max())
    final = df[df["iteration"] == final_iter].copy()
    final["label"] = (
        final["device"].str.upper()
        + "\np="
        + final["p"].astype(str)
        + "\n"
        + final["case_label"].astype(str)
        + "\ninit="
        + final["initial_fraction"].map(lambda x: f"{x:g}")
    )
    order = final.sort_values(["device", "p", "case", "initial_fraction"])["label"].unique()
    data = [
        final[final["label"] == label]["mse_gain_vs_random_median_pct"].to_numpy()
        for label in order
    ]
    fig, ax = plt.subplots(figsize=(max(14, 0.55 * len(order)), 6.2))
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#80b1d3")
        patch.set_alpha(0.72)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_ylabel("Final gain vs random median MSE (%)")
    ax.set_title("Final-iteration gain distribution across data/init seeds")
    ax.tick_params(axis="x", labelrotation=75)
    ax.grid(True, axis="y", linestyle=":", alpha=0.65)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_final_gain_boxplot.png", dpi=220)
    plt.close(fig)


def write_report(df, agg, args, counts, out_prefix):
    final_iter = int(df["iteration"].max())
    final = agg[agg["iteration"] == final_iter].copy()
    final["case"] = pd.Categorical(final["case"], CASE_ORDER, ordered=True)
    final = final.sort_values(["device", "p", "case", "initial_fraction"])
    raw_final = df[df["iteration"] == final_iter]
    overall_positive = float((raw_final["mse_gain_vs_random_median_pct"] > 0).mean())
    mean_final_gain = float(raw_final["mse_gain_vs_random_median_pct"].mean())
    device_names = sorted(df["device"].astype(str).unique())
    cuda_names = sorted(
        x for x in df["cuda_name"].dropna().astype(str).unique()
        if x and x.lower() != "nan"
    )
    cuda_desc = ", ".join(cuda_names) if cuda_names else "not used"

    lines = [
        "# 论文级高维迭代 D-optimal 实验报告",
        "",
        "## 实验目标",
        "",
        "本实验检验一个实际流程：在原始高维数据集上先均匀抽取小比例初始设计集，用该设计集估计 NN 与 FullPR surrogate 的距离；随后把当前已经升级后的设计集作为下一轮 D-optimal 的条件信息，迭代加入新样本并重新拟合 surrogate。实验关注 D-optimal 升级是否稳定优于同预算随机升级。",
        "",
        "## 实验矩阵",
        "",
        f"- 原始数据集大小：{args.n}。",
        f"- 输入维度：{', '.join(str(x) for x in args.p_values)}。",
        f"- 数据场景：{', '.join(args.cases)}。",
        f"- data seeds：{', '.join(str(x) for x in args.data_seeds)}。",
        f"- NN init seeds：{', '.join(str(x) for x in args.init_seeds)}。",
        f"- 初始 uniform 比例：{', '.join(f'{x:g}' for x in args.initial_fractions)}。",
        f"- 迭代轮数：{args.iterations}；每轮增加 {args.batch_size} 个点。",
        f"- 随机基准：每个迭代点重复 {args.random_repeats} 次。",
        f"- 设备：{', '.join(device_names)}；GPU：{cuda_desc}。",
        f"- NN：hidden={','.join(str(x) for x in args.hidden)}，activation={args.activation}，epochs={args.epochs}。",
        f"- FullPR：degree={args.degree}，include_special={args.include_special}。",
        "",
        "## 实际计算量",
        "",
        f"- NN oracle 训练次数：{counts['n_oracles']}。",
        f"- D-optimal 迭代轨迹数：{counts['n_trajectories']}。",
        f"- D-optimal surrogate 拟合次数：{counts['n_dopt_fits']}。",
        f"- 随机 surrogate 拟合次数：{counts['n_random_fits']}。",
        "",
        "## 最后一轮总体结论",
        "",
        f"最后一轮全部重复上的平均 MSE gain 为 `{mean_final_gain:.2f}%`，正收益比例为 `{100.0 * overall_positive:.1f}%`。这里的 gain 定义为 `(random median MSE - D-optimal MSE) / random median MSE`，因此正值表示 D-optimal 优于同预算随机升级。",
        "",
        "## 最后一轮分组统计",
        "",
        "| device | case | p | init frac | final n | reps | pilot shape | D-opt MSE | random MSE | gain mean ±95%CI | positive rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final.itertuples(index=False):
        reps = int(row.mse_gain_vs_random_median_pct_count)
        gain = row.mse_gain_vs_random_median_pct_mean
        ci = row.mse_gain_vs_random_median_pct_ci95
        lines.append(
            f"| {row.device} | {row.case} | {row.p} | {row.initial_fraction:g} | "
            f"{row.selected_n} | {reps} | "
            f"{row.pilot_shape_fullpr_vs_nn_mean:.3e} | "
            f"{row.dopt_mse_vs_nn_mean:.3e} | "
            f"{row.random_mse_median_mean:.3e} | "
            f"{gain:.2f}% ± {ci:.2f}% | "
            f"{100.0 * row.positive_rate:.1f}% |"
        )

    lines.extend([
        "",
        "## 论文表述建议",
        "",
        "1. 对 p=20，初始设计集通常已经超过二阶 FullPR 的特征数，因此 D-optimal 的优势主要表现为小幅但稳定的改进。",
        "2. 对 p=50，初始设计集低于 FullPR 特征数，早期迭代可能出现欠定不稳定；随着升级后的数据集不断被用于下一轮 D-optimal，误差会在选点数超过特征数后明显收敛。",
        "3. 强非线性场景用于说明方法边界：当 NN oracle 的局部行为不能由二阶 FullPR 充分表达时，D-optimal 仍可能改善采样覆盖，但不应被解释为充分逼近 NN 的保证。",
        "4. 因此论文中的核心结论应写成：pilot 距离和迭代式 D-optimal 升级提供了一个可观测、可复现的数据选择诊断流程；它在高维中可以稳定优于随机升级，但收益幅度取决于 FullPR 特征空间、初始样本比例和迭代预算。",
        "",
        "## 输出文件",
        "",
        f"- `{out_prefix}_raw.csv`：所有重复与迭代点的原始结果。",
        f"- `{out_prefix}_aggregate.csv`：均值、标准差、SEM、95% CI 和正收益比例。",
        f"- `{out_prefix}_final_summary.csv`：最后一轮分组汇总。",
        f"- `{out_prefix}_cpu_gain_mean_ci.png` / `{out_prefix}_cuda_gain_mean_ci.png`：gain 均值曲线与 95% CI。",
        f"- `{out_prefix}_cpu_mse_mean_ci.png` / `{out_prefix}_cuda_mse_mean_ci.png`：MSE 均值曲线与 95% CI。",
        f"- `{out_prefix}_final_gain_boxplot.png`：最后一轮 gain 分布箱线图。",
    ])
    Path(f"{out_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    args.devices = [str(d) for d in args.devices]
    counts = summarize_random_count(args)
    print("Paper-scale high-dimensional iterative D-optimal experiment")
    print(counts)
    all_rows = []
    out_raw = Path(f"{args.out_prefix}_raw.csv")
    if out_raw.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_raw} exists. Use --overwrite or choose another --out-prefix.")

    total_oracles = counts["n_oracles"]
    done_oracles = 0
    for device_name in args.devices:
        get_device(device_name)
        for p in args.p_values:
            for case in args.cases:
                for data_seed in args.data_seeds:
                    for init_seed in args.init_seeds:
                        done_oracles += 1
                        t0 = time.perf_counter()
                        print(
                            f"[{done_oracles}/{total_oracles}] "
                            f"device={device_name} case={case} p={p} "
                            f"data_seed={data_seed} init_seed={init_seed}",
                            flush=True,
                        )
                        shared = run_oracle(case, p, data_seed, init_seed,
                                            device_name, args)
                        for frac in args.initial_fractions:
                            rows = run_design_path(
                                case, p, device_name, data_seed, init_seed,
                                frac, shared, args)
                            all_rows.extend(rows)
                        df_partial = pd.DataFrame(all_rows)
                        df_partial.to_csv(out_raw, index=False,
                                          encoding="utf-8-sig")
                        print(
                            f"  oracle done in {time.perf_counter() - t0:.1f}s; "
                            f"rows={len(all_rows)}",
                            flush=True,
                        )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_raw, index=False, encoding="utf-8-sig")
    agg = aggregate_results(df)
    agg.to_csv(f"{args.out_prefix}_aggregate.csv", index=False,
               encoding="utf-8-sig")
    final_summary = agg[agg["iteration"] == args.iterations].copy()
    final_summary.to_csv(f"{args.out_prefix}_final_summary.csv", index=False,
                         encoding="utf-8-sig")
    plot_mean_curves(agg, args.out_prefix)
    plot_final_boxplots(df, args.out_prefix)
    write_report(df, agg, args, counts, args.out_prefix)
    print(f"Saved {args.out_prefix}_raw.csv")
    print(f"Saved {args.out_prefix}_aggregate.csv")
    print(f"Saved {args.out_prefix}_final_summary.csv")
    print(f"Saved {args.out_prefix}_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--p-values", type=int, nargs="*", default=[20, 50])
    parser.add_argument("--cases", nargs="*", default=CASE_ORDER,
                        choices=CASE_ORDER)
    parser.add_argument("--data-seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--init-seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--initial-fractions", type=float, nargs="*",
                        default=[0.05, 0.1])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden", type=int, nargs="*", default=[128, 64, 32])
    parser.add_argument("--activation", default="tanh",
                        choices=["softplus", "tanh", "sigmoid", "relu"])
    parser.add_argument("--devices", nargs="*", default=["cpu", "cuda"],
                        choices=["cpu", "cuda"])
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--include-special", action="store_true", default=True)
    parser.add_argument("--no-special", dest="include_special", action="store_false")
    parser.add_argument("--min-initial", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise-sd", type=float, default=0.03)
    parser.add_argument("--ridge-alpha", type=float, default=1e-5)
    parser.add_argument("--dopt-ridge", type=float, default=1e-2)
    parser.add_argument("--dopt-chunk-size", type=int, default=512)
    parser.add_argument("--shape-points", type=int, default=1000)
    parser.add_argument("--out-prefix", default="paper_highdim_iter_dopt")
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())
