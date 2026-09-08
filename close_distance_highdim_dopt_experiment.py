"""
High-dimensional close-distance D-optimal transfer experiment.

This is the high-dimensional version of the close-distance D-optimal test. It
aligns with the other paper-scale experiments by using p in {5, 10, 20, 50},
high-dimensional synthetic cases, a degree-2 FullPR feature space with special
terms, and in-domain candidate/evaluation regions.

The target claim is conditional:

    If a trained NN oracle is close to a FullPR surrogate on the same data
    region where candidates are selected and evaluated, then FullPR-based
    D-optimal designs should query the NN more efficiently than random designs.
"""

from __future__ import annotations

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
from scipy.linalg import qr
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

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
    train_oracle_nn,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "close_distance_highdim_dopt"
FIG_DIR = ROOT / "figures" / "close_distance_highdim_dopt"
NOTES_DIR = ROOT / "reports" / "notes"

CASE_ORDER = ["highdim_poly2", "highdim_smooth", "highdim_strong"]
CASE_LABELS = {
    **CASE_LABELS,
    "highdim_poly2": "Quadratic",
    "highdim_smooth": "Smooth",
    "highdim_strong": "Strong nonlinear",
}
CASE_COLORS = {
    "highdim_poly2": "#4C78A8",
    "highdim_smooth": "#54A24B",
    "highdim_strong": "#B279A2",
}
STRATEGY_COLORS = {
    "D-optimal": "#D55E00",
    "Latin hypercube": "#009E73",
    "Random median": "#6C6C6C",
}


def stable_seed(*values) -> int:
    seed = 1729
    for value in values:
        seed = (seed * 1_000_003 + int(value) + 97) % (2**32 - 1)
    return int(seed)


def sem(values) -> float:
    s = pd.Series(values).dropna()
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=1) / math.sqrt(len(s)))


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def candidate_indices(n_train, max_candidates, seed):
    if max_candidates <= 0 or max_candidates >= n_train:
        return np.arange(n_train, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_train, size=max_candidates, replace=False))


def dopt_select_qr_then_leverage(Z, budget, ridge, chunk_size, batch_size):
    """Select rows by QR-pivot seed plus leverage-score D-optimal upgrades."""
    n, d = Z.shape
    if budget > n:
        raise ValueError("budget cannot exceed candidate pool size")

    seed_n = min(budget, d)
    # QR with column pivoting on Z.T selects informative rows of Z. For k>d,
    # use those d rows as a full-rank seed, then add leverage-score batches.
    _q, _r, piv = qr(Z.T, mode="economic", pivoting=True)
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[piv[:seed_n]] = True

    while selected_mask.sum() < budget:
        add_n = min(batch_size, budget - int(selected_mask.sum()))
        add_idx = batch_dopt_select(Z, selected_mask, add_n, ridge, chunk_size)
        if len(add_idx) == 0:
            break
        selected_mask[add_idx] = True

    return np.flatnonzero(selected_mask)[:budget]


def latin_candidate_select(X_cand, k, seed):
    sampler = qmc.LatinHypercube(d=X_cand.shape[1], seed=seed)
    targets_unit = sampler.random(k)
    lo = X_cand.min(axis=0)
    hi = X_cand.max(axis=0)
    targets = qmc.scale(targets_unit, lo, hi)

    n_neighbors = min(max(10, k // 4), len(X_cand))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(X_cand)
    _dist, neigh = nbrs.kneighbors(targets, return_distance=True)

    picked = []
    used = set()
    for row in neigh:
        for pos in row:
            pos = int(pos)
            if pos not in used:
                picked.append(pos)
                used.add(pos)
                break
        if len(picked) == k:
            break

    if len(picked) < k:
        rng = np.random.default_rng(seed + 19)
        free = np.array([i for i in range(len(X_cand)) if i not in used], dtype=int)
        fill = rng.choice(free, size=k - len(picked), replace=False)
        picked.extend(int(i) for i in fill)
    return np.asarray(picked, dtype=int)


def one_design_comparison(args, shared, base, budget):
    X_cand = shared["X_cand"]
    F_cand = shared["F_cand"]
    F_eval = shared["F_eval"]
    Z_cand = shared["Z_cand"]
    pred_nn_cand = shared["pred_nn_cand"]
    pred_nn_eval = shared["pred_nn_eval"]
    X_eval = shared["X_eval"]

    d_idx = dopt_select_qr_then_leverage(
        Z_cand,
        budget,
        args.dopt_ridge,
        args.dopt_chunk_size,
        args.dopt_batch_size,
    )
    pred_dopt, _ = fit_surrogate(
        F_cand[d_idx], pred_nn_cand[d_idx], F_eval, args.ridge_alpha
    )
    dopt_mse = float(mean_squared_error(pred_nn_eval, pred_dopt))
    dopt_shape = limited_shape_distance(
        X_eval, pred_nn_eval, pred_dopt, shared["rng"], args.shape_points
    )

    latin_idx = latin_candidate_select(
        X_cand, budget, stable_seed(base["p"], base["data_seed"], base["init_seed"], budget, 811)
    )
    pred_latin, _ = fit_surrogate(
        F_cand[latin_idx], pred_nn_cand[latin_idx], F_eval, args.ridge_alpha
    )
    latin_mse = float(mean_squared_error(pred_nn_eval, pred_latin))
    latin_shape = limited_shape_distance(
        X_eval, pred_nn_eval, pred_latin, shared["rng"], args.shape_points
    )

    rand_mse = []
    rand_shape = []
    rng = np.random.default_rng(
        stable_seed(base["p"], base["data_seed"], base["init_seed"], budget, 911)
    )
    for _ in range(args.random_repeats):
        ridx = rng.choice(len(X_cand), size=budget, replace=False)
        pred_random, _ = fit_surrogate(
            F_cand[ridx], pred_nn_cand[ridx], F_eval, args.ridge_alpha
        )
        rand_mse.append(float(mean_squared_error(pred_nn_eval, pred_random)))
        rand_shape.append(
            limited_shape_distance(
                X_eval, pred_nn_eval, pred_random, rng, args.shape_points
            )
        )

    mse_s = summarize(rand_mse)
    shape_s = summarize(rand_shape)
    return {
        **base,
        "status": "used_close",
        "candidate_domain": "training_data_region",
        "n_candidates": len(X_cand),
        "budget": budget,
        "budget_multiplier": budget / base["n_design_params"],
        "dopt_mse_vs_nn": dopt_mse,
        "dopt_shape_vs_nn": dopt_shape,
        "latin_mse_vs_nn": latin_mse,
        "latin_shape_vs_nn": latin_shape,
        "random_mse_median": mse_s["median"],
        "random_mse_q25": mse_s["q25"],
        "random_mse_q75": mse_s["q75"],
        "random_mse_mean": mse_s["mean"],
        "random_shape_median": shape_s["median"],
        "random_shape_q25": shape_s["q25"],
        "random_shape_q75": shape_s["q75"],
        "random_shape_mean": shape_s["mean"],
        "dopt_gain_vs_random_median_pct": (
            100.0 * (mse_s["median"] - dopt_mse) / mse_s["median"]
        ),
        "latin_gain_vs_random_median_pct": (
            100.0 * (mse_s["median"] - latin_mse) / mse_s["median"]
        ),
        "dopt_shape_gain_vs_random_median_pct": (
            100.0 * (shape_s["median"] - dopt_shape) / shape_s["median"]
        ),
        "latin_shape_gain_vs_random_median_pct": (
            100.0 * (shape_s["median"] - latin_shape) / shape_s["median"]
        ),
    }


def run_one(args, case, p, data_seed, init_seed, device):
    t0 = time.perf_counter()
    rng = np.random.default_rng(stable_seed(p, data_seed, init_seed, 101))
    hidden = tuple(args.hidden)

    X_train, X_eval, y_train, y_eval = make_highdim_dataset(
        case, args.n, p, data_seed, args.noise_sd
    )
    model = train_oracle_nn(
        X_train,
        y_train,
        X_eval,
        y_eval,
        hidden,
        args.activation,
        args.epochs,
        init_seed,
        device,
    )
    pred_nn_train = nn_predict(model, X_train, device)
    pred_nn_eval = nn_predict(model, X_eval, device)
    oracle_mse_vs_y = float(mean_squared_error(y_eval, pred_nn_eval))

    F_train, F_eval = build_fullpr_features(
        X_train, X_eval, args.degree, args.include_special
    )
    pred_full_fpr, _ = fit_surrogate(
        F_train, pred_nn_train, F_eval, args.ridge_alpha
    )
    fpr_mimic_mse = float(mean_squared_error(pred_nn_eval, pred_full_fpr))
    fpr_mse_vs_y = float(mean_squared_error(y_eval, pred_full_fpr))
    shape_nn_fpr = limited_shape_distance(
        X_eval, pred_nn_eval, pred_full_fpr, rng, args.shape_points
    )
    abs_rmse_gap = abs(math.sqrt(oracle_mse_vs_y) - math.sqrt(fpr_mse_vs_y))

    Z_train = design_matrix_for_dopt(F_train)
    n_features = F_train.shape[1]
    n_design_params = Z_train.shape[1]
    is_close = shape_nn_fpr <= args.max_shape_distance
    base = {
        "case": case,
        "case_label": CASE_LABELS.get(case, case),
        "p": p,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "data_seed": data_seed,
        "init_seed": init_seed,
        "n": args.n,
        "n_train": len(X_train),
        "n_eval": len(X_eval),
        "hidden": ",".join(str(x) for x in hidden),
        "activation": args.activation,
        "epochs": args.epochs,
        "degree": args.degree,
        "include_special": args.include_special,
        "n_fullpr_features": n_features,
        "n_design_params": n_design_params,
        "oracle_mse_vs_y": oracle_mse_vs_y,
        "full_fpr_mse_vs_y": fpr_mse_vs_y,
        "full_fpr_mse_vs_nn": fpr_mimic_mse,
        "shape_nn_fpr": shape_nn_fpr,
        "abs_rmse_gap": abs_rmse_gap,
        "close_threshold": args.max_shape_distance,
        "is_close": is_close,
    }

    if not is_close and not args.keep_nonclose:
        return [{**base, "budget": np.nan, "status": "skipped_nonclose"}]

    cand_idx = candidate_indices(
        len(X_train), args.candidates, stable_seed(p, data_seed, init_seed, 701)
    )
    X_cand = X_train[cand_idx]
    F_cand = F_train[cand_idx]
    Z_cand = Z_train[cand_idx]
    pred_nn_cand = pred_nn_train[cand_idx]
    shared = {
        "X_cand": X_cand,
        "F_cand": F_cand,
        "F_eval": F_eval,
        "Z_cand": Z_cand,
        "pred_nn_cand": pred_nn_cand,
        "pred_nn_eval": pred_nn_eval,
        "X_eval": X_eval,
        "rng": rng,
    }

    if args.budgets:
        budgets = [int(x) for x in args.budgets]
    else:
        budgets = [int(round(n_design_params * mult)) for mult in args.budget_multipliers]
    budgets = sorted({b for b in budgets if 2 <= b <= len(X_cand)})
    if not budgets:
        return [{**base, "budget": np.nan, "status": "skipped_no_valid_budget"}]

    rows = []
    for budget in budgets:
        row = one_design_comparison(args, shared, base, budget)
        row["elapsed_sec_run"] = time.perf_counter() - t0
        rows.append(row)
    return rows


def aggregate_results(raw):
    used = raw[raw["status"] == "used_close"].copy()
    if used.empty:
        return pd.DataFrame()
    group_cols = [
        "p",
        "case",
        "case_label",
        "budget",
        "budget_multiplier",
        "n_design_params",
        "n_fullpr_features",
    ]
    metrics = [
        "shape_nn_fpr",
        "abs_rmse_gap",
        "full_fpr_mse_vs_nn",
        "dopt_mse_vs_nn",
        "latin_mse_vs_nn",
        "random_mse_median",
        "dopt_gain_vs_random_median_pct",
        "latin_gain_vs_random_median_pct",
        "dopt_shape_gain_vs_random_median_pct",
        "latin_shape_gain_vs_random_median_pct",
    ]
    agg = used.groupby(group_cols)[metrics].agg(["mean", "std", "count"]).reset_index()
    agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    for metric in metrics:
        agg[f"{metric}_se"] = agg[f"{metric}_std"].fillna(0.0) / np.sqrt(
            agg[f"{metric}_count"].clip(lower=1)
        )
        agg[f"{metric}_ci95"] = 1.96 * agg[f"{metric}_se"]
    return agg


def plot_close_filter(raw, out_prefix):
    base = raw.drop_duplicates(["p", "case", "data_seed", "init_seed"]).copy()
    p_values = sorted(base["p"].dropna().unique())
    ncols = 2
    nrows = int(math.ceil(len(p_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 4.4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, p_values):
        part = base[base["p"] == p]
        skipped = part[part["status"] != "used_close"]
        if not skipped.empty:
            ax.scatter(
                skipped["shape_nn_fpr"],
                skipped["abs_rmse_gap"],
                s=48,
                color="#B8B8B8",
                alpha=0.82,
                label="Skipped",
            )
        for case in CASE_ORDER:
            sub = part[(part["case"] == case) & (part["status"] == "used_close")]
            if sub.empty:
                continue
            ax.scatter(
                sub["shape_nn_fpr"],
                sub["abs_rmse_gap"],
                s=56,
                color=CASE_COLORS.get(case, "#333333"),
                alpha=0.86,
                label=CASE_LABELS.get(case, case),
            )
        ax.axvline(base["close_threshold"].dropna().iloc[0], color="#333333", linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"p={int(p)}")
        ax.set_xlabel("NN-FPR shape distance")
        ax.set_ylabel("NN/FPR RMSE gap")
        ax.grid(True, alpha=0.25)
    for ax in axes[len(p_values):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Close-distance filter across high-dimensional settings", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / f"{out_prefix}_close_filter.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_gain_by_p(agg, out_prefix):
    if agg.empty:
        return None
    stats = (
        agg.groupby(["p", "budget_multiplier"], as_index=False)
        .agg(
            dopt_gain=("dopt_gain_vs_random_median_pct_mean", "mean"),
            dopt_se=("dopt_gain_vs_random_median_pct_mean", sem),
            latin_gain=("latin_gain_vs_random_median_pct_mean", "mean"),
            latin_se=("latin_gain_vs_random_median_pct_mean", sem),
        )
    )
    p_values = sorted(stats["p"].unique())
    ncols = 2
    nrows = int(math.ceil(len(p_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 4.2 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, p_values):
        sub = stats[stats["p"] == p].sort_values("budget_multiplier")
        x = sub["budget_multiplier"].to_numpy(dtype=float)
        for label, mean_col, se_col in [
            ("D-optimal", "dopt_gain", "dopt_se"),
            ("Latin hypercube", "latin_gain", "latin_se"),
        ]:
            y = sub[mean_col].to_numpy(dtype=float)
            ci = 1.96 * sub[se_col].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2, color=STRATEGY_COLORS[label], label=label)
            ax.fill_between(x, y - ci, y + ci, color=STRATEGY_COLORS[label], alpha=0.13, linewidth=0)
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_title(f"p={int(p)}")
        ax.set_xlabel("Budget / number of FullPR parameters")
        ax.set_ylabel("MSE gain vs random median (%)")
        ax.grid(True, alpha=0.25)
    for ax in axes[len(p_values):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("High-dimensional close-distance D-optimal transfer gains", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / f"{out_prefix}_gain_by_p.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_gain_heatmap(agg, out_prefix, focus_multiplier):
    if agg.empty:
        return None
    focus = agg[np.isclose(agg["budget_multiplier"], focus_multiplier)].copy()
    if focus.empty:
        return None
    table = (
        focus.pivot_table(
            index="case_label",
            columns="p",
            values="dopt_gain_vs_random_median_pct_mean",
            aggfunc="mean",
        )
        .reindex([CASE_LABELS[c] for c in CASE_ORDER])
    )
    vals = table.to_numpy(dtype=float)
    vmax = max(1.0, np.nanmax(np.abs(vals)))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    im = ax.imshow(vals, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([f"p={int(c)}" for c in table.columns])
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = vals[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9)
    ax.set_title(f"D-optimal gain at {focus_multiplier:g}x parameter budget")
    fig.colorbar(im, ax=ax, label="MSE gain vs random median (%)")
    fig.tight_layout()
    out = FIG_DIR / f"{out_prefix}_dopt_gain_heatmap_{focus_multiplier:g}x.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_report(raw, agg, figure_paths, out_prefix):
    base = raw.drop_duplicates(["p", "case", "data_seed", "init_seed"]).copy()
    used = base[base["status"] == "used_close"]
    skipped = base[base["status"] != "used_close"]
    lines = [
        "# 高维近距离 D-optimal 迁移实验报告",
        "",
        "## 实验目的",
        "",
        (
            "本实验把近距离 D-optimal 迁移测试扩展到高维设置，和其他实验的 "
            "`p=5,10,20,50` 保持一致。核心问题是：当 NN oracle 能被二阶 "
            "FullPR surrogate 在同一数据区域内较好复刻时，FullPR-based "
            "D-optimal 选点是否比同预算随机采样更高效。"
        ),
        "",
        "## 设置",
        "",
        f"- p values：{', '.join(str(int(x)) for x in sorted(raw['p'].dropna().unique()))}。",
        f"- cases：{', '.join(raw['case'].dropna().unique())}。",
        f"- n：{int(raw['n'].dropna().iloc[0])}；训练/评估：{int(raw['n_train'].dropna().iloc[0])}/{int(raw['n_eval'].dropna().iloc[0])}。",
        f"- NN：hidden={raw['hidden'].dropna().iloc[0]}，activation={raw['activation'].dropna().iloc[0]}，epochs={int(raw['epochs'].dropna().iloc[0])}。",
        f"- FullPR：degree={int(raw['degree'].dropna().iloc[0])}，include_special={bool(raw['include_special'].dropna().iloc[0])}。",
        f"- close filter：`shape_nn_fpr <= {raw['close_threshold'].dropna().iloc[0]:.3e}`。",
        f"- base runs：{len(base)}；通过 close filter：{len(used)}；跳过：{len(skipped)}。",
        "",
    ]
    if not used.empty:
        lines.extend([
            "## 近距离筛选摘要",
            "",
            "| p | case | used runs | median shape | median FPR mimic MSE | median RMSE gap | design params |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ])
        summary = (
            raw[raw["status"] == "used_close"]
            .drop_duplicates(["p", "case", "data_seed", "init_seed"])
            .groupby(["p", "case", "case_label"], as_index=False)
            .agg(
                used_runs=("shape_nn_fpr", "size"),
                median_shape=("shape_nn_fpr", "median"),
                median_mimic_mse=("full_fpr_mse_vs_nn", "median"),
                median_gap=("abs_rmse_gap", "median"),
                design_params=("n_design_params", "first"),
            )
            .sort_values(["p", "case"])
        )
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {int(row.p)} | {row.case_label} | {int(row.used_runs)} | "
                f"{row.median_shape:.3e} | {row.median_mimic_mse:.3e} | "
                f"{row.median_gap:.3e} | {int(row.design_params)} |"
            )
        lines.append("")
    if not agg.empty:
        lines.extend([
            "## D-optimal gain 摘要",
            "",
            "| p | case | budget x params | D-opt gain mean +/-95%CI | Latin gain mean +/-95%CI | reps |",
            "|---:|---|---:|---:|---:|---:|",
        ])
        for row in agg.sort_values(["p", "case", "budget_multiplier"]).itertuples(index=False):
            lines.append(
                f"| {int(row.p)} | {row.case_label} | {row.budget_multiplier:.1f} | "
                f"{row.dopt_gain_vs_random_median_pct_mean:.2f}% +/- "
                f"{row.dopt_gain_vs_random_median_pct_ci95:.2f}% | "
                f"{row.latin_gain_vs_random_median_pct_mean:.2f}% +/- "
                f"{row.latin_gain_vs_random_median_pct_ci95:.2f}% | "
                f"{int(row.dopt_gain_vs_random_median_pct_count)} |"
            )
        lines.append("")
    lines.extend([
        "## 图",
        "",
    ])
    for path in figure_paths:
        if path is not None:
            lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend([
        "",
        "## 解读",
        "",
        (
            "这版实验比低维 p=3 sanity check 更适合放进主线，因为维度、数据生成机制、"
            "FullPR 特征阶数和候选区域都与其他高维实验对齐。需要注意的是，D-optimal "
            "优势仍应按 close filter 后的子集解释：它检验的是条件命题，而不是声称 "
            "D-optimal 在所有高维场景下都优于随机。"
        ),
    ])
    out = NOTES_DIR / f"{out_prefix}_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run(args):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / f"{args.out_prefix}_raw.csv"
    if raw_path.exists() and not args.overwrite:
        raise FileExistsError(f"{raw_path} exists; use --overwrite or change --out-prefix")

    device = get_device(args.device)
    rows = []
    total = len(args.p_values) * len(args.cases) * len(args.data_seeds) * len(args.init_seeds)
    done = 0
    for p in args.p_values:
        for case in args.cases:
            for data_seed in args.data_seeds:
                for init_seed in args.init_seeds:
                    done += 1
                    print(
                        f"[{done}/{total}] p={p} case={case} "
                        f"data_seed={data_seed} init_seed={init_seed}",
                        flush=True,
                    )
                    rows.extend(run_one(args, case, p, data_seed, init_seed, device))
                    pd.DataFrame(rows).to_csv(raw_path, index=False, encoding="utf-8-sig")

    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    agg = aggregate_results(raw)
    agg_path = DATA_DIR / f"{args.out_prefix}_aggregate.csv"
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")

    figure_paths = [
        plot_close_filter(raw, args.out_prefix),
        plot_gain_by_p(agg, args.out_prefix),
        plot_gain_heatmap(agg, args.out_prefix, args.focus_multiplier),
    ]
    report_path = write_report(raw, agg, figure_paths, args.out_prefix)

    print(f"Saved {raw_path}")
    print(f"Saved {agg_path}")
    for path in figure_paths:
        if path is not None:
            print(f"Saved {path}")
    print(f"Saved {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-dimensional close-distance D-optimal transfer experiment."
    )
    parser.add_argument("--out-prefix", default="close_distance_highdim_dopt")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--p-values", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=CASE_ORDER,
        choices=CASE_ORDER,
    )
    parser.add_argument("--data-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--noise-sd", type=float, default=0.03)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--include-special", action="store_true", default=True)
    parser.add_argument("--no-special", dest="include_special", action="store_false")
    parser.add_argument("--ridge-alpha", type=float, default=1e-5)
    parser.add_argument("--max-shape-distance", type=float, default=5e-3)
    parser.add_argument("--keep-nonclose", action="store_true")
    parser.add_argument(
        "--candidates",
        type=int,
        default=7500,
        help="Maximum in-domain candidate rows sampled from the training pool; <=0 uses all.",
    )
    parser.add_argument("--budget-multipliers", type=float, nargs="+", default=[2, 3, 4])
    parser.add_argument("--budgets", type=int, nargs="*", default=None)
    parser.add_argument("--focus-multiplier", type=float, default=3.0)
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--shape-points", type=int, default=1000)
    parser.add_argument("--dopt-ridge", type=float, default=1e-2)
    parser.add_argument("--dopt-chunk-size", type=int, default=1024)
    parser.add_argument("--dopt-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
