"""
High-dimensional iterative D-optimal transfer experiment.

This script is the paper-scale version of the pilot-D-optimal idea:

1. Generate an original dataset with n=10000 and p in {10, 20, 50, 100, 200}.
2. Train a full NN oracle on the original training split.
3. Start from a small uniform pilot/design set.
4. Fit a FullPR surrogate to the NN oracle on the current design set.
5. Upgrade the design set with a batch D-optimal rule.
6. Refit the surrogate and repeat.
7. Compare each upgraded design against random additions with the same budget.

For high p, full cubic polynomial features are too large for an exact greedy
D-optimal inverse update. The high-dimensional experiment therefore uses a
degree-2 FullPR feature space and a scalable batch leverage approximation to
D-optimal selection. The approximation is conditional on the currently selected
dataset at each iteration, so the upgraded dataset is used to choose the next
upgrade.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.linalg import solve_triangular
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from measure_morala import (
    ConfigurableNet,
    scale_minmax,
    surface_shape_distance,
    train_nn,
)


CASE_LABELS = {
    "highdim_poly2": "Quadratic",
    "highdim_smooth": "Smooth",
    "highdim_strong": "Strong nonlinear",
}

DEFAULT_P_VALUES = [10, 20, 50, 100, 200]


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(name)


def make_response_params(p, data_seed):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(np.zeros((1, p)))
    rng = np.random.default_rng(data_seed + 10_000)
    beta = rng.normal(size=(poly.n_output_features_, 1))
    beta /= np.sqrt(poly.n_output_features_)
    return {"beta": beta}


def highdim_response(X, case, params, rng, noise_sd):
    p = X.shape[1]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    F = poly.fit_transform(X)
    y = F @ params["beta"]
    if case == "highdim_smooth":
        y = y.copy()
        y[:, 0] += 0.45 * np.sin(2.5 * X[:, 0] + 0.4)
        if p >= 3:
            y[:, 0] += 0.30 * np.cos(1.5 * X[:, 1] * X[:, 2])
        if p >= 5:
            y[:, 0] += 0.25 * np.tanh(X[:, 3] + 0.5 * X[:, 4])
    elif case == "highdim_strong":
        y = y.copy()
        y[:, 0] += 0.70 * np.sin(4.0 * X[:, 0] + 0.7)
        if p >= 4:
            y[:, 0] += 0.45 * np.cos(3.0 * X[:, 1] * X[:, 2])
            y[:, 0] += 0.35 * np.sin(2.0 * X[:, 2] + X[:, 3])
        if p >= 8:
            y[:, 0] += 0.30 * np.tanh(X[:, 4] * X[:, 5] - X[:, 6] + 0.5 * X[:, 7])
    y += rng.normal(0.0, noise_sd, size=y.shape)
    return y


def make_highdim_dataset(case, n, p, data_seed, noise_sd):
    rng_x = np.random.default_rng(data_seed)
    X = rng_x.normal(0.0, 1.0, size=(n, p))
    X_train_raw, X_test_raw = train_test_split(
        X, test_size=0.25, random_state=42)
    X_train, X_test = scale_minmax(X_train_raw, X_test_raw, (-1.0, 1.0))

    params = make_response_params(p, data_seed)
    rng_y_train = np.random.default_rng(data_seed + 20_000)
    rng_y_test = np.random.default_rng(data_seed + 30_000)
    y_train_raw = highdim_response(X_train, case, params, rng_y_train, noise_sd)
    y_test_raw = highdim_response(X_test, case, params, rng_y_test, noise_sd)
    y_train, y_test = scale_minmax(y_train_raw, y_test_raw, (-1.0, 1.0))
    return X_train, X_test, y_train, y_test


def build_fullpr_features(X_train, X_test, degree, include_special):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    F_train = poly.fit_transform(X_train)
    F_test = poly.transform(X_test)
    if include_special:
        special_train = np.column_stack([
            fn(X_train[:, i])
            for i in range(X_train.shape[1])
            for fn in (np.sin, lambda z: np.exp(np.clip(z, -30, 30)))
        ])
        special_test = np.column_stack([
            fn(X_test[:, i])
            for i in range(X_test.shape[1])
            for fn in (np.sin, lambda z: np.exp(np.clip(z, -30, 30)))
        ])
        F_train = np.column_stack([F_train, special_train])
        F_test = np.column_stack([F_test, special_test])

    scaler = StandardScaler().fit(F_train)
    F_train = scaler.transform(F_train).astype(np.float64, copy=False)
    F_test = scaler.transform(F_test).astype(np.float64, copy=False)
    return F_train, F_test


def design_matrix_for_dopt(F):
    return np.column_stack([np.ones(len(F)), F]).astype(np.float64, copy=False)


def train_oracle_nn(X_train, y_train, X_test, y_test, hidden, activation, epochs,
                    seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = ConfigurableNet(X_train.shape[1], hidden, activation).to(device)
    train_nn(
        model,
        torch.tensor(X_train, dtype=torch.float32, device=device),
        torch.tensor(y_train, dtype=torch.float32, device=device),
        X_eval=torch.tensor(X_test, dtype=torch.float32, device=device),
        y_eval=torch.tensor(y_test, dtype=torch.float32, device=device),
        epochs=epochs,
        optimizer_name="rprop",
        capture_every=max(1, epochs // 5),
        verbose=False,
    )
    model.eval()
    return model


def nn_predict(model, X, device, batch_size=8192):
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outs).reshape(-1, 1)


def fit_surrogate(F_train, y_train, F_eval, alpha):
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr",
                  tol=1e-7, max_iter=2000)
    model.fit(F_train, y_train)
    return model.predict(F_eval).reshape(-1, 1), model


def limited_shape_distance(X, pred_a, pred_b, rng, max_points):
    if max_points and len(X) > max_points:
        idx = rng.choice(len(X), size=max_points, replace=False)
        return surface_shape_distance(X[idx], pred_a[idx], pred_b[idx])
    return surface_shape_distance(X, pred_a, pred_b)


def batch_dopt_select(Z, selected_mask, batch_size, ridge, chunk_size):
    remaining = np.flatnonzero(~selected_mask)
    if len(remaining) == 0:
        return np.array([], dtype=int)
    batch_size = min(batch_size, len(remaining))
    Z_selected = Z[selected_mask]
    d = Z.shape[1]

    if d > len(Z_selected):
        return batch_dopt_select_dual(
            Z, Z_selected, remaining, batch_size, ridge, chunk_size)

    info = Z_selected.T @ Z_selected
    info.flat[::d + 1] += ridge

    jitter = 0.0
    while True:
        try:
            L = np.linalg.cholesky(info)
            break
        except np.linalg.LinAlgError:
            jitter = ridge if jitter == 0.0 else jitter * 10.0
            info.flat[::d + 1] += jitter

    scores = np.empty(len(remaining), dtype=np.float64)
    for start in range(0, len(remaining), chunk_size):
        stop = min(start + chunk_size, len(remaining))
        idx = remaining[start:stop]
        solved = solve_triangular(
            L, Z[idx].T, lower=True, check_finite=False, overwrite_b=False)
        scores[start:stop] = np.sum(solved * solved, axis=0)

    top = np.argpartition(scores, -batch_size)[-batch_size:]
    top = top[np.argsort(scores[top])[::-1]]
    return remaining[top]


def batch_dopt_select_dual(Z, Z_selected, remaining, batch_size, ridge, chunk_size):
    """Compute leverage scores through the selected-sample Gram matrix.

    This is equivalent to z^T (Z_s^T Z_s + ridge I)^-1 z by Woodbury, and is
    much cheaper when p is high and the selected design has fewer rows than
    design parameters.
    """
    s = len(Z_selected)
    sample_gram = Z_selected @ Z_selected.T
    sample_gram /= ridge
    sample_gram.flat[::s + 1] += 1.0

    jitter = 0.0
    while True:
        try:
            L = np.linalg.cholesky(sample_gram)
            break
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
            sample_gram.flat[::s + 1] += jitter

    scores = np.empty(len(remaining), dtype=np.float64)
    inv_ridge = 1.0 / ridge
    inv_ridge_sq = inv_ridge * inv_ridge
    for start in range(0, len(remaining), chunk_size):
        stop = min(start + chunk_size, len(remaining))
        idx = remaining[start:stop]
        Z_chunk = Z[idx]
        norm2 = np.einsum("ij,ij->i", Z_chunk, Z_chunk, optimize=True)
        cross = Z_selected @ Z_chunk.T
        solved = solve_triangular(
            L, cross, lower=True, check_finite=False, overwrite_b=False)
        scores[start:stop] = norm2 * inv_ridge - np.sum(
            solved * solved, axis=0) * inv_ridge_sq

    scores = np.maximum(scores, 0.0)
    top = np.argpartition(scores, -batch_size)[-batch_size:]
    top = top[np.argsort(scores[top])[::-1]]
    return remaining[top]


def random_baseline(F_train, y_oracle_train, F_eval, pred_oracle_eval, X_eval,
                    initial_idx, selected_n, repeats, alpha, rng, shape_points):
    n_train = len(F_train)
    initial_idx = np.asarray(initial_idx, dtype=int)
    initial_set = set(initial_idx.tolist())
    rest = np.array([i for i in range(n_train) if i not in initial_set], dtype=int)
    mse_values = []
    shape_values = []
    extra_n = max(0, selected_n - len(initial_idx))

    for _ in range(repeats):
        if extra_n == 0:
            ridx = rng.choice(n_train, size=selected_n, replace=False)
        else:
            extra = rng.choice(rest, size=extra_n, replace=False)
            ridx = np.concatenate([initial_idx, extra])
        pred, _ = fit_surrogate(F_train[ridx], y_oracle_train[ridx], F_eval, alpha)
        mse_values.append(mean_squared_error(pred_oracle_eval, pred))
        shape_values.append(
            limited_shape_distance(X_eval, pred_oracle_eval, pred, rng, shape_points)
        )

    arr_mse = np.asarray(mse_values)
    arr_shape = np.asarray(shape_values)
    return {
        "random_mse_median": float(np.median(arr_mse)),
        "random_mse_q25": float(np.quantile(arr_mse, 0.25)),
        "random_mse_q75": float(np.quantile(arr_mse, 0.75)),
        "random_shape_median": float(np.median(arr_shape)),
        "random_shape_q25": float(np.quantile(arr_shape, 0.25)),
        "random_shape_q75": float(np.quantile(arr_shape, 0.75)),
    }


def pilot_distance(F_train, y_oracle_train, X_train, initial_idx, alpha, rng,
                   shape_points):
    initial_idx = np.asarray(initial_idx, dtype=int)
    train_idx, val_idx = train_test_split(
        initial_idx, test_size=0.25, random_state=2026)
    pred, _ = fit_surrogate(F_train[train_idx], y_oracle_train[train_idx],
                            F_train[val_idx], alpha)
    return {
        "pilot_mse_fullpr_vs_nn": mean_squared_error(y_oracle_train[val_idx], pred),
        "pilot_shape_fullpr_vs_nn": limited_shape_distance(
            X_train[val_idx], y_oracle_train[val_idx], pred, rng, shape_points),
    }


def run_one(case, p, args, device, rng):
    t0 = time.perf_counter()
    hidden = tuple(args.hidden)
    X_train, X_test, y_train, y_test = make_highdim_dataset(
        case, args.n, p, args.data_seed, args.noise_sd)

    model = train_oracle_nn(
        X_train, y_train, X_test, y_test, hidden, args.activation,
        args.epochs, args.seed, device)
    pred_oracle_train = nn_predict(model, X_train, device)
    pred_oracle_eval = nn_predict(model, X_test, device)
    oracle_mse_vs_y = mean_squared_error(y_test, pred_oracle_eval)

    F_train, F_eval = build_fullpr_features(
        X_train, X_test, args.degree, args.include_special)
    Z = design_matrix_for_dopt(F_train)
    n_features = F_train.shape[1]
    n_design_params = Z.shape[1]

    initial_n = max(int(round(len(X_train) * args.initial_fraction)),
                    args.min_initial)
    initial_n = min(initial_n, len(X_train))
    initial_idx = rng.choice(len(X_train), size=initial_n, replace=False)
    selected_mask = np.zeros(len(X_train), dtype=bool)
    selected_mask[initial_idx] = True

    pilot = pilot_distance(F_train, pred_oracle_train, X_train, initial_idx,
                           args.ridge_alpha, rng, args.shape_points)

    rows = []
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
            "p": p,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
            "n_original": args.n,
            "n_train_pool": len(X_train),
            "n_eval": len(X_test),
            "hidden": ",".join(str(x) for x in hidden),
            "activation": args.activation,
            "epochs": args.epochs,
            "degree": args.degree,
            "include_special": args.include_special,
            "n_fullpr_features": n_features,
            "n_design_params": n_design_params,
            "initial_fraction": args.initial_fraction,
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
            "elapsed_sec_case_p": time.perf_counter() - t0,
        })

        if iteration == args.iterations:
            break
        add_idx = batch_dopt_select(
            Z, selected_mask, args.batch_size, args.dopt_ridge,
            args.dopt_chunk_size)
        selected_mask[add_idx] = True
        print(
            f"  {case}, p={p}, iter={iteration + 1}: "
            f"selected {selected_mask.sum()} / {len(X_train)}")

    return rows


def plot_results(df, out_prefix):
    for metric, ylabel, suffix in [
        ("dopt_mse_vs_nn", "MSE vs NN oracle", "mse_curve"),
        ("mse_gain_vs_random_median_pct", "Gain vs random median MSE (%)", "gain_curve"),
    ]:
        fig, axes = plt.subplots(1, len(sorted(df["p"].unique())),
                                 figsize=(6.2 * len(sorted(df["p"].unique())), 4.8),
                                 squeeze=False)
        for ax, p in zip(axes[0], sorted(df["p"].unique())):
            subp = df[df["p"] == p]
            for case, sub in subp.groupby("case"):
                sub = sub.sort_values("iteration")
                label = CASE_LABELS.get(case, case)
                ax.plot(sub["selected_n"], sub[metric], marker="o", label=label)
                if metric == "dopt_mse_vs_nn":
                    ax.plot(sub["selected_n"], sub["random_mse_median"],
                            linestyle="--", alpha=0.7, label=f"{label} random")
            if metric.endswith("pct"):
                ax.axhline(0, color="#555555", linewidth=1)
            ax.set_title(f"p={p}")
            ax.set_xlabel("Selected training points")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=":", alpha=0.65)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_{suffix}.png", dpi=200)
        plt.close(fig)


def write_report(df, out_prefix):
    final = df.sort_values("iteration").groupby(["case", "p", "device"]).tail(1)
    pos = int((final["mse_gain_vs_random_median_pct"] > 0).sum())
    total = len(final)
    p_list = sorted(df["p"].unique())
    p_desc = ", ".join(str(x) for x in p_list)
    feature_rows = (
        final[["p", "n_fullpr_features", "n_design_params", "initial_n"]]
        .drop_duplicates()
        .sort_values("p")
    )
    feature_note = "；".join(
        f"p={int(row.p)} 时 FullPR 特征数为 {int(row.n_fullpr_features)}、"
        f"D-opt 参数数为 {int(row.n_design_params)}、初始样本为 {int(row.initial_n)}"
        for row in feature_rows.itertuples(index=False)
    )
    lines = [
        "# 高维迭代 D-optimal 实验报告",
        "",
        "## 实验流程",
        "",
        f"本实验把原始数据集大小固定为 10000，并测试输入维度 p={p_desc}。先在原始训练集上训练 NN，作为新的目标模型；随后从训练池中均匀抽取初始设计集，用 FullPR surrogate 拟合 NN 输出。每一轮根据当前已经升级后的设计集计算 D-optimal leverage，选择下一批样本加入设计集，再重新拟合 FullPR surrogate。因此这里的 D-optimal 是迭代式的，而不是一次性选点。",
        "",
        "## 设置",
        "",
        f"- 设备：{', '.join(sorted(df['device'].astype(str).unique()))}",
        f"- 原始数据集大小：{int(df['n_original'].iloc[0])}",
        f"- 维度：{', '.join(str(x) for x in sorted(df['p'].unique()))}",
        f"- 训练池/评估集：{int(df['n_train_pool'].iloc[0])}/{int(df['n_eval'].iloc[0])}",
        f"- NN hidden：{df['hidden'].iloc[0]}；activation={df['activation'].iloc[0]}；epochs={int(df['epochs'].iloc[0])}",
        f"- FullPR：degree={int(df['degree'].iloc[0])}，include_special={bool(df['include_special'].iloc[0])}",
        f"- 初始 uniform 设计比例：{df['initial_fraction'].iloc[0]:g}",
        f"- 每轮 D-optimal 增加样本数：{int(df['batch_size'].iloc[0])}",
        f"- 随机基准重复次数：{int(df.attrs.get('random_repeats', 0))}",
        "",
        "## 最后一轮结果",
        "",
        "| case | p | FullPR特征数 | 初始n | 最终n | pilot shape | D-opt MSE | random中位MSE | gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final.sort_values(["p", "case"]).itertuples(index=False):
        lines.append(
            f"| {row.case} | {row.p} | {row.n_fullpr_features} | "
            f"{row.initial_n} | {row.selected_n} | "
            f"{row.pilot_shape_fullpr_vs_nn:.3e} | "
            f"{row.dopt_mse_vs_nn:.3e} | {row.random_mse_median:.3e} | "
            f"{row.mse_gain_vs_random_median_pct:.1f}% |"
        )

    lines.extend([
        "",
        "## 结论",
        "",
        f"最后一轮共有 {pos}/{total} 个组合中 D-optimal 优于同预算随机升级中位数。这个表明：在高维情况下，D-optimal 仍然可以作为升级数据集的可执行策略，但效果依赖于 FullPR 特征空间是否足以表达 NN oracle 的局部行为，以及初始 uniform 设计是否已经覆盖了关键方向。",
        "",
        f"需要强调的是，高维时如果继续使用三阶 FullPR，特征数会迅速膨胀，原始逐点贪心 D-optimal 会变成矩阵计算瓶颈。本实验使用二阶 FullPR 加特殊项，并采用 batch leverage 近似 D-optimal；当 D-opt 参数数超过当前已选样本数时，代码会用等价的样本空间 Woodbury 形式计算 leverage，避免直接分解巨大的参数空间信息矩阵。当前特征规模为：{feature_note}。",
        "",
        "## 输出文件",
        "",
        f"- `{out_prefix}_results.csv`：逐轮完整结果。",
        f"- `{out_prefix}_summary.csv`：最后一轮汇总。",
        f"- `{out_prefix}_mse_curve.png`：迭代误差曲线。",
        f"- `{out_prefix}_gain_curve.png`：相对随机基准收益曲线。",
    ])
    Path(f"{out_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    device = get_device(args.device)
    rng = np.random.default_rng(args.seed)
    all_rows = []
    for p in args.p_values:
        for case in args.cases:
            print(f"Running case={case}, p={p}, device={device}")
            rows = run_one(case, p, args, device, rng)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.attrs["random_repeats"] = args.random_repeats
    df.to_csv(f"{args.out_prefix}_results.csv", index=False, encoding="utf-8-sig")
    summary = df.sort_values("iteration").groupby(["case", "p", "device"]).tail(1)
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False, encoding="utf-8-sig")
    plot_results(df, args.out_prefix)
    write_report(df, args.out_prefix)
    print(df.to_string(index=False))
    print(f"Saved {args.out_prefix}_results.csv")
    print(f"Saved {args.out_prefix}_summary.csv")
    print(f"Saved {args.out_prefix}_mse_curve.png")
    print(f"Saved {args.out_prefix}_gain_curve.png")
    print(f"Saved {args.out_prefix}_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--p-values", type=int, nargs="*", default=DEFAULT_P_VALUES)
    parser.add_argument("--cases", nargs="*", default=["highdim_poly2", "highdim_smooth"],
                        choices=["highdim_poly2", "highdim_smooth", "highdim_strong"])
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--include-special", action="store_true", default=True)
    parser.add_argument("--no-special", dest="include_special", action="store_false")
    parser.add_argument("--initial-fraction", type=float, default=0.1)
    parser.add_argument("--min-initial", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--hidden", type=int, nargs="*", default=[128, 64, 32])
    parser.add_argument("--activation", default="tanh",
                        choices=["softplus", "tanh", "sigmoid", "relu"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--noise-sd", type=float, default=0.03)
    parser.add_argument("--ridge-alpha", type=float, default=1e-5)
    parser.add_argument("--dopt-ridge", type=float, default=1e-2)
    parser.add_argument("--dopt-chunk-size", type=int, default=512)
    parser.add_argument("--shape-points", type=int, default=1000)
    parser.add_argument("--out-prefix", default="highdim_iter_dopt")
    run(parser.parse_args())
