"""
Pilot-distance gated D-optimal transfer experiment.

Practical question:
  In real use we do not know the true NN-FullPR distance. Can a small uniform
  pilot sample estimate that distance, and can that estimate predict whether
  D-optimal sampling from the FullPR feature space helps distill the NN?

Workflow:
  For each dataset and each pilot fraction:
    1. Uniformly sample a small pilot subset from a larger dataset.
    2. Train a pilot NN and pilot FullPR on the pilot subset.
    3. Estimate NN-FullPR distance on pilot validation points.
    4. Treat the pilot NN as the "new model" oracle.
    5. Use FullPR-feature D-optimal design to query the NN.
    6. Fit a FullPR surrogate to those NN outputs.
    7. Compare D-optimal against random designs with the same budget.

If small pilot distance predicts positive D-optimal gain, then the pilot
distance can be used as a practical pre-check for transferring old polynomial
design technology to NN models.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from d_optimal_transfer_experiment import (
    _feature_matrix,
    fit_surrogate,
    greedy_d_optimal,
    summarize_random,
)
from measure_morala import (
    ConfigurableNet,
    custom_features_full,
    generate_dataset,
    register_hooks,
    remove_hooks,
    reset_capture,
    surface_shape_distance,
    train_nn,
)


CONFIGS = [
    ("paper_poly3", "tanh"),
    ("paper_poly4", "tanh"),
    ("smooth_nonlinear", "softplus"),
    ("smooth_nonlinear_rand", "tanh"),
]


def fit_fullpr_predict(X_train, y_train, X_eval, max_order, include_special):
    F_tr = custom_features_full(X_train, max_order=max_order,
                                include_special=include_special)
    F_ev = custom_features_full(X_eval, max_order=max_order,
                                include_special=include_special)
    lr = LinearRegression().fit(F_tr, y_train)
    return lr.predict(F_ev), lr


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(name)


def nn_predict(model, X, device, batch_size=8192):
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outs).reshape(-1, 1)


def train_model(X_train, y_train, X_eval, y_eval, activation, hidden, epochs, seed,
                device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    reset_capture()
    model = ConfigurableNet(X_train.shape[1], hidden, activation).to(device)
    hooks = register_hooks(model)
    train_nn(model,
             torch.tensor(X_train, dtype=torch.float32, device=device),
             torch.tensor(y_train, dtype=torch.float32, device=device),
             X_eval=torch.tensor(X_eval, dtype=torch.float32, device=device),
             y_eval=torch.tensor(y_eval, dtype=torch.float32, device=device),
             epochs=epochs,
             optimizer_name="rprop",
             capture_every=max(1, epochs // 4),
             verbose=False)
    remove_hooks(hooks)
    model.eval()
    return model


def make_full_scaled_dataset(case, n_full, data_seed):
    X_tr, X_te, y_tr, y_te = generate_dataset(case, n=n_full, p=3, rng=data_seed,
                                              scaling=(-1.0, 1.0))
    X = np.vstack([X_tr.numpy(), X_te.numpy()])
    y = np.vstack([y_tr.numpy(), y_te.numpy()])
    return X, y


def run_one(case, activation, pilot_fraction, args, rng):
    device = get_device(args.device)
    hidden = tuple(args.hidden)
    max_order = len(hidden)
    include_special = True
    X_all, y_all = make_full_scaled_dataset(case, args.n_full, args.data_seed)
    n_total = len(X_all)
    pilot_n = max(int(round(n_total * pilot_fraction)), args.min_pilot)
    pilot_n = min(pilot_n, n_total)

    pilot_idx = rng.choice(n_total, size=pilot_n, replace=False)
    rest_idx = np.setdiff1d(np.arange(n_total), pilot_idx, assume_unique=False)
    if len(rest_idx) < args.eval_points:
        eval_idx = rng.choice(n_total, size=args.eval_points, replace=False)
    else:
        eval_idx = rng.choice(rest_idx, size=args.eval_points, replace=False)
    cand_idx = rest_idx if len(rest_idx) >= args.candidates else np.arange(n_total)
    if len(cand_idx) > args.candidates:
        cand_idx = rng.choice(cand_idx, size=args.candidates, replace=False)

    X_pilot = X_all[pilot_idx]
    y_pilot = y_all[pilot_idx]
    X_train, X_val, y_train, y_val = train_test_split(
        X_pilot, y_pilot, test_size=0.25, random_state=args.seed)

    F_tmp = _feature_matrix(X_all[: min(100, len(X_all))], max_order,
                            include_special)
    n_params = F_tmp.shape[1]
    if len(X_train) <= n_params:
        raise ValueError(
            f"pilot train size {len(X_train)} <= parameter count {n_params}; "
            f"increase n_full or min_pilot")

    model = train_model(X_train, y_train, X_val, y_val, activation, hidden,
                        args.epochs, args.seed, device)
    pred_nn_val = nn_predict(model, X_val, device)
    pred_fpr_val, _ = fit_fullpr_predict(X_train, y_train, X_val, max_order,
                                         include_special)

    pilot = {
        "case": case,
        "activation": activation,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "hidden": ",".join(map(str, hidden)),
        "pilot_fraction": pilot_fraction,
        "pilot_n": pilot_n,
        "pilot_train_n": len(X_train),
        "pilot_val_n": len(X_val),
        "n_parameters": n_params,
        "pilot_nn_mse_vs_y": mean_squared_error(y_val, pred_nn_val),
        "pilot_fpr_mse_vs_y": mean_squared_error(y_val, pred_fpr_val),
        "pilot_fpr_mse_vs_nn": mean_squared_error(pred_nn_val, pred_fpr_val),
        "pilot_shape_nn_fpr": surface_shape_distance(X_val, pred_nn_val,
                                                     pred_fpr_val),
    }

    X_cand = X_all[cand_idx]
    X_eval = X_all[eval_idx]
    pred_nn_eval = nn_predict(model, X_eval, device)
    F_cand = _feature_matrix(X_cand, max_order, include_special)
    budgets = args.budgets or [n_params, 2 * n_params]
    budgets = [int(b) for b in budgets if int(b) <= len(X_cand)]

    rows = []
    for budget in budgets:
        d_idx = greedy_d_optimal(F_cand, budget)
        pred_d, _ = fit_surrogate(X_cand[d_idx], nn_predict(model, X_cand[d_idx], device),
                                  X_eval, max_order, include_special)
        d_mse = mean_squared_error(pred_nn_eval, pred_d)
        d_shape = surface_shape_distance(X_eval, pred_nn_eval, pred_d)

        rand_mse = []
        rand_shape = []
        for _ in range(args.random_repeats):
            ridx = rng.choice(len(X_cand), size=budget, replace=False)
            pred_r, _ = fit_surrogate(X_cand[ridx], nn_predict(model, X_cand[ridx], device),
                                      X_eval, max_order, include_special)
            rand_mse.append(mean_squared_error(pred_nn_eval, pred_r))
            rand_shape.append(surface_shape_distance(X_eval, pred_nn_eval, pred_r))

        ms = summarize_random(rand_mse)
        ss = summarize_random(rand_shape)
        rows.append({
            **pilot,
            "budget": budget,
            "dopt_mse_vs_nn": d_mse,
            "dopt_shape_vs_nn": d_shape,
            "random_mse_median": ms["median"],
            "random_mse_q25": ms["q25"],
            "random_mse_q75": ms["q75"],
            "random_shape_median": ss["median"],
            "random_shape_q25": ss["q25"],
            "random_shape_q75": ss["q75"],
            "mse_gain_vs_random_median_pct": 100.0 * (ms["median"] - d_mse) / ms["median"],
            "shape_gain_vs_random_median_pct": 100.0 * (ss["median"] - d_shape) / ss["median"],
        })
    return rows


def plot_outputs(df, out_prefix):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {0.1: "o", 0.01: "s", 0.001: "^"}
    for case, sub in df.groupby("case"):
        for frac, sub2 in sub.groupby("pilot_fraction"):
            s = sub2[sub2["budget"] == sub2["n_parameters"] * 2]
            if len(s) == 0:
                s = sub2
            ax.scatter(s["pilot_shape_nn_fpr"],
                       s["mse_gain_vs_random_median_pct"],
                       marker=markers.get(float(frac), "o"),
                       s=80,
                       label=f"{case}, pilot={frac:g}")
            for _, r in s.iterrows():
                ax.annotate(case.replace("smooth_", "s_"),
                            (r["pilot_shape_nn_fpr"],
                             r["mse_gain_vs_random_median_pct"]),
                            fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("pilot-estimated shape(NN, FullPR)")
    ax.set_ylabel("D-optimal MSE gain vs random median (%)")
    device_label = ", ".join(sorted(df["device"].astype(str).unique())) if "device" in df else ""
    ax.set_title(f"Can pilot distance predict D-optimal transfer benefit? {device_label}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_distance_vs_gain.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    focus = df[df["budget"] == df["n_parameters"] * 2].copy()
    focus["label"] = focus["case"] + "\n" + focus["pilot_fraction"].map(lambda x: f"{x:g}")
    ax.bar(np.arange(len(focus)), focus["mse_gain_vs_random_median_pct"],
           color=["#2a9d8f" if v >= 0 else "#e76f51"
                  for v in focus["mse_gain_vs_random_median_pct"]])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(focus)))
    ax.set_xticklabels(focus["label"], rotation=35, ha="right")
    ax.set_ylabel("D-optimal MSE gain vs random median (%)")
    ax.set_title("Pilot-gated D-optimal result at 2x parameter budget")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_gain_by_dataset.png", dpi=150)
    plt.close(fig)


def write_report(df, out_prefix):
    focus = df[df["budget"] == df["n_parameters"] * 2].copy()
    corr = focus[["pilot_shape_nn_fpr", "mse_gain_vs_random_median_pct"]].corr(
        method="spearman").iloc[0, 1]
    lines = [
        "# Pilot 距离门控的 D-optimal 迁移实验",
        "",
        "实际问题：真实应用中我们并不知道 NN 和 FullPR 的距离。能否先用很小的均匀 pilot 子集训练初始 NN/FullPR，估计两者距离，再判断 D-optimal 是否值得用？",
        "",
        "默认设置：",
        "",
        f"- 设备：{', '.join(sorted(df['device'].astype(str).unique()))}",
        f"- 数据集：{', '.join(sorted(df['case'].unique()))}",
        f"- pilot 比例：{', '.join(str(x) for x in sorted(df['pilot_fraction'].unique(), reverse=True))}",
        f"- 每个 case/fraction/budget 的随机基准重复次数：{int(df.attrs.get('random_repeats', 0))}",
        "- 汇总重点：2 倍参数数目的查询预算",
        "",
        f"在 2 倍参数预算下，pilot shape distance 与 D-optimal MSE gain 的 Spearman 相关为 `{corr:.3f}`。",
        "",
        "注意：本次结果不是“pilot 距离越小，D-optimal 收益越大”。相反，相关为正，说明在这批设置里，距离更大的非线性数据集反而更容易体现 D-optimal 相对随机的优势；而距离很小的多项式数据集里，随机设计已经足够强，D-optimal 在 2 倍参数预算下不一定占优。",
        "",
        "| 数据集 | pilot 比例 | pilot shape | D-opt 相对随机中位数 MSE 改善 |",
        "|---|---:|---:|---:|",
    ]
    for _, r in focus.sort_values(["case", "pilot_fraction"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {r['case']} | {r['pilot_fraction']:.3g} | "
            f"{r['pilot_shape_nn_fpr']:.3e} | "
            f"{r['mse_gain_vs_random_median_pct']:.1f}% |")
    lines.extend([
        "",
        "解释：pilot 距离仍然有价值，但它不能单独作为“距离小就用 D-optimal”的充分条件。更准确的结论是：pilot 流程能提前暴露 NN-FullPR 的局部关系和随机设计的稳定性；是否迁移 D-optimal，还应结合候选域、预算和一个小规模随机基准。这个结果修正了上一版理想化实验：老技术可以迁移，但实际应用中需要 pilot 诊断，而不能只靠理论距离假设。",
    ])
    with open(f"{out_prefix}_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(args):
    rng = np.random.default_rng(args.seed)
    all_rows = []
    for case, activation in CONFIGS:
        for frac in args.pilot_fractions:
            print(f"Running case={case}, activation={activation}, pilot={frac:g}")
            rows = run_one(case, activation, frac, args, rng)
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.attrs["random_repeats"] = args.random_repeats
    df.to_csv(f"{args.out_prefix}_results.csv", index=False, encoding="utf-8-sig")
    focus = df[df["budget"] == df["n_parameters"] * 2].copy()
    focus.to_csv(f"{args.out_prefix}_summary.csv", index=False, encoding="utf-8-sig")
    plot_outputs(df, args.out_prefix)
    write_report(df, args.out_prefix)
    print(df.to_string(index=False))
    print(f"\nSaved: {args.out_prefix}_results.csv")
    print(f"Saved: {args.out_prefix}_summary.csv")
    print(f"Saved: {args.out_prefix}_distance_vs_gain.png")
    print(f"Saved: {args.out_prefix}_gain_by_dataset.png")
    print(f"Saved: {args.out_prefix}_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-full", type=int, default=5000)
    ap.add_argument("--pilot-fractions", type=float, nargs="*",
                    default=[0.1, 0.01])
    ap.add_argument("--min-pilot", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--hidden", type=int, nargs="*", default=[16, 8, 4])
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--candidates", type=int, default=3000)
    ap.add_argument("--eval-points", type=int, default=1000)
    ap.add_argument("--random-repeats", type=int, default=30)
    ap.add_argument("--budgets", type=int, nargs="*", default=None)
    ap.add_argument("--out-prefix", default="pilot_dopt")
    run(ap.parse_args())
