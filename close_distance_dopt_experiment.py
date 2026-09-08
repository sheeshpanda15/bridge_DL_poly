"""
Close-distance D-optimal transfer experiment.

This script tests the specific theoretical claim:

    When an NN and the user's FullPR feature basis are close on the same
    in-domain candidate/evaluation region, FullPR-based D-optimal designs
    should sample the NN oracle more efficiently than random designs.

The experiment deliberately keeps candidate points inside the data region. That
avoids the failure mode where NN-FPR closeness is measured on the data
distribution, but D-optimal selection is allowed to chase high-leverage points
in a wider extrapolation domain.
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
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

from measure_morala import (
    ConfigurableNet,
    custom_features_full,
    fit_full_pr,
    generate_dataset,
    surface_shape_distance,
    train_nn,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "close_distance_dopt"
FIG_DIR = ROOT / "figures" / "close_distance_dopt"
NOTES_DIR = ROOT / "reports" / "notes"

CASE_LABELS = {
    "paper_poly3": "Poly3",
    "paper_poly4": "Poly4",
    "smooth_nonlinear": "Smooth",
    "smooth_nonlinear_rand": "Smooth-rand",
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


def nn_predict(model, X_np):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X_np, dtype=torch.float32))
    return out.cpu().numpy().reshape(-1, 1)


def feature_matrix(X, max_order, include_special):
    features = custom_features_full(
        X, max_order=max_order, include_special=include_special
    )
    return np.column_stack([np.ones(len(X)), features])


def standardized_design(F):
    Z = np.asarray(F, dtype=np.float64).copy()
    if Z.shape[1] <= 1:
        return Z
    mu = Z[:, 1:].mean(axis=0)
    sd = Z[:, 1:].std(axis=0)
    sd[sd == 0] = 1.0
    Z[:, 1:] = (Z[:, 1:] - mu) / sd
    return Z


def greedy_d_optimal(F, k, ridge=1e-3):
    """Greedy max-det row selection using sequential leverage scores."""
    Z = standardized_design(F)
    n, p = Z.shape
    if k > n:
        raise ValueError("budget cannot exceed candidate pool size")
    info_inv = np.eye(p, dtype=np.float64) / ridge
    available = np.ones(n, dtype=bool)
    selected = []
    for _ in range(k):
        scores = np.einsum("ij,jk,ik->i", Z, info_inv, Z, optimize=True)
        scores[~available] = -np.inf
        idx = int(np.argmax(scores))
        selected.append(idx)
        available[idx] = False
        f = Z[idx:idx + 1].T
        denom = float((1.0 + f.T @ info_inv @ f).item())
        info_inv = info_inv - (info_inv @ f @ f.T @ info_inv) / denom
    return np.asarray(selected, dtype=int)


def latin_candidate_select(X_cand, k, seed):
    sampler = qmc.LatinHypercube(d=X_cand.shape[1], seed=seed)
    unit_targets = sampler.random(k)
    lo = X_cand.min(axis=0)
    hi = X_cand.max(axis=0)
    targets = qmc.scale(unit_targets, lo, hi)

    n_neighbors = min(max(10, k // 4), len(X_cand))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(X_cand)
    _, neigh = nbrs.kneighbors(targets, return_distance=True)

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
        free = np.array([i for i in range(len(X_cand)) if i not in used])
        fill = rng.choice(free, size=k - len(picked), replace=False)
        picked.extend(int(i) for i in fill)
    return np.asarray(picked, dtype=int)


def fit_surrogate(X_design, y_design, X_eval, max_order, include_special, ridge_alpha):
    F_design = custom_features_full(
        X_design, max_order=max_order, include_special=include_special
    )
    F_eval = custom_features_full(
        X_eval, max_order=max_order, include_special=include_special
    )
    if ridge_alpha > 0:
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    else:
        model = LinearRegression()
    model.fit(F_design, y_design)
    return model.predict(F_eval).reshape(-1, 1)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def sem(values):
    s = pd.Series(values).dropna()
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=1) / math.sqrt(len(s)))


def pick_candidate_pool(X_train, max_candidates, seed):
    if max_candidates is None or max_candidates <= 0 or len(X_train) <= max_candidates:
        return X_train
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=max_candidates, replace=False)
    return X_train[np.sort(idx)]


def run_one(args, case, data_seed, init_seed):
    t0 = time.perf_counter()
    hidden = tuple(args.hidden)
    max_order = args.max_order if args.max_order is not None else len(hidden)
    include_special = not args.no_special

    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    X_train_t, X_eval_t, y_train_t, y_eval_t = generate_dataset(
        case, n=args.n, p=args.p, rng=data_seed, scaling=(-1.0, 1.0)
    )
    X_train = X_train_t.numpy()
    X_eval = X_eval_t.numpy()
    y_train = y_train_t.numpy()
    y_eval = y_eval_t.numpy()

    model = ConfigurableNet(args.p, hidden, args.activation)
    train_nn(
        model,
        X_train_t,
        y_train_t,
        X_eval=X_eval_t,
        y_eval=y_eval_t,
        epochs=args.epochs,
        optimizer_name=args.optimizer,
        capture_every=max(1, args.epochs // 5),
        verbose=False,
    )

    pred_nn_eval = nn_predict(model, X_eval)
    pred_fpr_eval, _ = fit_full_pr(
        X_train, y_train, X_eval, max_order=max_order,
        include_special=include_special
    )
    nn_mse_vs_y = float(mean_squared_error(y_eval, pred_nn_eval))
    fpr_mse_vs_y = float(mean_squared_error(y_eval, pred_fpr_eval))
    fpr_mse_vs_nn = float(mean_squared_error(pred_nn_eval, pred_fpr_eval))
    shape_nn_fpr = float(surface_shape_distance(X_eval, pred_nn_eval, pred_fpr_eval))
    abs_rmse_gap = abs(math.sqrt(nn_mse_vs_y) - math.sqrt(fpr_mse_vs_y))
    is_close = shape_nn_fpr <= args.max_shape_distance

    base = {
        "case": case,
        "case_label": CASE_LABELS.get(case, case),
        "p": args.p,
        "activation": args.activation,
        "hidden": ",".join(str(x) for x in hidden),
        "max_order": max_order,
        "include_special": include_special,
        "data_seed": data_seed,
        "init_seed": init_seed,
        "n": args.n,
        "n_train": len(X_train),
        "n_eval": len(X_eval),
        "nn_mse_vs_y": nn_mse_vs_y,
        "fpr_mse_vs_y": fpr_mse_vs_y,
        "fpr_mse_vs_nn": fpr_mse_vs_nn,
        "shape_nn_fpr": shape_nn_fpr,
        "abs_rmse_gap": abs_rmse_gap,
        "close_threshold": args.max_shape_distance,
        "is_close": is_close,
    }

    if not is_close and not args.keep_nonclose:
        return [{**base, "budget": np.nan, "status": "skipped_nonclose"}]

    X_cand = pick_candidate_pool(
        X_train, args.candidates, stable_seed(data_seed, init_seed, 701)
    )
    pred_nn_cand = nn_predict(model, X_cand)
    F_cand = feature_matrix(X_cand, max_order, include_special)
    n_parameters = F_cand.shape[1]

    if args.budgets:
        budgets = [int(b) for b in args.budgets]
    else:
        budgets = [n_parameters * m for m in args.budget_multipliers]
    budgets = sorted({b for b in budgets if 2 <= b <= len(X_cand)})
    if not budgets:
        raise ValueError("No valid budgets after clipping to candidate pool size")

    rows = []
    dopt_cache = {}
    for budget in budgets:
        d_idx = dopt_cache.get(budget)
        if d_idx is None:
            d_idx = greedy_d_optimal(F_cand, budget, ridge=args.dopt_ridge)
            dopt_cache[budget] = d_idx
        pred_d = fit_surrogate(
            X_cand[d_idx], pred_nn_cand[d_idx], X_eval, max_order,
            include_special, args.surrogate_ridge
        )
        dopt_mse = float(mean_squared_error(pred_nn_eval, pred_d))
        dopt_shape = float(surface_shape_distance(X_eval, pred_nn_eval, pred_d))

        latin_idx = latin_candidate_select(
            X_cand, budget, stable_seed(data_seed, init_seed, budget, 811)
        )
        pred_l = fit_surrogate(
            X_cand[latin_idx], pred_nn_cand[latin_idx], X_eval, max_order,
            include_special, args.surrogate_ridge
        )
        latin_mse = float(mean_squared_error(pred_nn_eval, pred_l))
        latin_shape = float(surface_shape_distance(X_eval, pred_nn_eval, pred_l))

        rand_mse = []
        rand_shape = []
        rng = np.random.default_rng(stable_seed(data_seed, init_seed, budget, 911))
        for _ in range(args.random_repeats):
            r_idx = rng.choice(len(X_cand), size=budget, replace=False)
            pred_r = fit_surrogate(
                X_cand[r_idx], pred_nn_cand[r_idx], X_eval, max_order,
                include_special, args.surrogate_ridge
            )
            rand_mse.append(float(mean_squared_error(pred_nn_eval, pred_r)))
            rand_shape.append(float(surface_shape_distance(X_eval, pred_nn_eval, pred_r)))
        mse_s = summarize(rand_mse)
        shape_s = summarize(rand_shape)

        rows.append({
            **base,
            "status": "used_close" if is_close else "used_nonclose",
            "candidate_domain": "training_data_region",
            "n_candidates": len(X_cand),
            "budget": budget,
            "budget_multiplier": budget / n_parameters,
            "n_parameters": n_parameters,
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
            "elapsed_sec_run": time.perf_counter() - t0,
        })
    return rows


def aggregate_results(df):
    used = df[df["status"].isin(["used_close", "used_nonclose"])].copy()
    if used.empty:
        return pd.DataFrame()
    group_cols = ["case", "case_label", "budget", "budget_multiplier", "n_parameters"]
    metrics = [
        "shape_nn_fpr",
        "abs_rmse_gap",
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


def plot_gain_by_budget(agg, out_prefix):
    if agg.empty:
        return None
    cases = list(agg["case_label"].drop_duplicates())
    fig, axes = plt.subplots(
        1, len(cases), figsize=(5.1 * len(cases), 4.3), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, case_label in zip(axes, cases):
        sub = agg[agg["case_label"] == case_label].sort_values("budget_multiplier")
        x = sub["budget_multiplier"].to_numpy(dtype=float)
        for label, metric, color in [
            ("D-optimal", "dopt_gain_vs_random_median_pct", STRATEGY_COLORS["D-optimal"]),
            ("Latin hypercube", "latin_gain_vs_random_median_pct", STRATEGY_COLORS["Latin hypercube"]),
        ]:
            y = sub[f"{metric}_mean"].to_numpy(dtype=float)
            ci = sub[f"{metric}_ci95"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=label)
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.13, linewidth=0)
        ax.axhline(0, color="#444444", linewidth=1.0)
        ax.set_title(case_label)
        ax.set_xlabel("Budget / number of FullPR parameters")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("MSE gain vs random median (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("D-optimal transfer gains after filtering to NN-FPR-close runs", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{out_prefix}_gain_by_budget.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gain_by_budget_stable(agg, out_prefix):
    stable = agg[agg["budget_multiplier"] >= 2.0].copy()
    if stable.empty:
        return None
    cases = list(stable["case_label"].drop_duplicates())
    fig, axes = plt.subplots(
        1, len(cases), figsize=(5.1 * len(cases), 4.3), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, case_label in zip(axes, cases):
        sub = stable[stable["case_label"] == case_label].sort_values("budget_multiplier")
        x = sub["budget_multiplier"].to_numpy(dtype=float)
        for label, metric, color in [
            ("D-optimal", "dopt_gain_vs_random_median_pct", STRATEGY_COLORS["D-optimal"]),
            ("Latin hypercube", "latin_gain_vs_random_median_pct", STRATEGY_COLORS["Latin hypercube"]),
        ]:
            y = sub[f"{metric}_mean"].to_numpy(dtype=float)
            ci = sub[f"{metric}_ci95"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=label)
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.13, linewidth=0)
        ax.axhline(0, color="#444444", linewidth=1.0)
        ax.set_title(case_label)
        ax.set_xlabel("Budget / number of FullPR parameters")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("MSE gain vs random median (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Stable-budget D-optimal transfer gains after close-distance filtering", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{out_prefix}_gain_by_budget_stable.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mse_by_budget(agg, out_prefix):
    if agg.empty:
        return None
    cases = list(agg["case_label"].drop_duplicates())
    fig, axes = plt.subplots(
        1, len(cases), figsize=(5.1 * len(cases), 4.3), sharey=False
    )
    axes = np.atleast_1d(axes)
    for ax, case_label in zip(axes, cases):
        sub = agg[agg["case_label"] == case_label].sort_values("budget_multiplier")
        x = sub["budget_multiplier"].to_numpy(dtype=float)
        ax.plot(
            x,
            sub["dopt_mse_vs_nn_mean"],
            marker="o",
            linewidth=2.0,
            color=STRATEGY_COLORS["D-optimal"],
            label="D-optimal",
        )
        ax.plot(
            x,
            sub["latin_mse_vs_nn_mean"],
            marker="^",
            linewidth=2.0,
            color=STRATEGY_COLORS["Latin hypercube"],
            label="Latin hypercube",
        )
        ax.plot(
            x,
            sub["random_mse_median_mean"],
            marker="s",
            linestyle="--",
            linewidth=2.0,
            color=STRATEGY_COLORS["Random median"],
            label="Random median",
        )
        ax.set_yscale("log")
        ax.set_title(case_label)
        ax.set_xlabel("Budget / number of FullPR parameters")
        ax.set_ylabel("Surrogate MSE vs NN oracle")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Surrogate accuracy on the same in-domain evaluation region", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{out_prefix}_mse_by_budget.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_close_filter(df, out_prefix):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    used = df[df["status"].isin(["used_close", "used_nonclose"])]
    skipped = df[df["status"] == "skipped_nonclose"]
    if not skipped.empty:
        ax.scatter(
            skipped["shape_nn_fpr"],
            skipped["abs_rmse_gap"],
            s=55,
            color="#B7B7B7",
            label="Skipped: not close",
        )
    if not used.empty:
        for case_label, sub in used.groupby("case_label"):
            ax.scatter(
                sub["shape_nn_fpr"],
                sub["abs_rmse_gap"],
                s=62,
                alpha=0.82,
                label=f"Used: {case_label}",
            )
    ax.axvline(df["close_threshold"].dropna().iloc[0], color="#333333", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NN-FPR shape distance on held-out data")
    ax.set_ylabel("Absolute RMSE gap between NN and FPR")
    ax.set_title("Close-distance filter for the D-optimal transfer test")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / f"{out_prefix}_close_filter.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(df, agg, figure_paths, out_prefix):
    used = df[df["status"].isin(["used_close", "used_nonclose"])].copy()
    skipped = df[df["status"] == "skipped_nonclose"].copy()
    lines = [
        "# Close-distance D-optimal transfer experiment",
        "",
        "## Question",
        "",
        (
            "When NN and FullPR are close on the same in-domain region used for "
            "candidate selection and evaluation, does FullPR-based D-optimal "
            "design approximate the NN oracle better than random sampling?"
        ),
        "",
        "## Close filter",
        "",
        f"- Close threshold: `shape_nn_fpr <= {df['close_threshold'].dropna().iloc[0]:.3e}`.",
        f"- Used rows: `{len(used)}` budget-level rows.",
        f"- Skipped non-close base runs: `{skipped[['case', 'data_seed', 'init_seed']].drop_duplicates().shape[0]}`.",
        "",
    ]
    if not used.empty:
        base_runs = used.drop_duplicates(["case", "data_seed", "init_seed"])
        lines.extend([
            f"- Median NN-FPR shape distance among used base runs: `{base_runs['shape_nn_fpr'].median():.3e}`.",
            f"- Median NN/FPR RMSE gap among used base runs: `{base_runs['abs_rmse_gap'].median():.3e}`.",
            "",
        ])
    if not agg.empty:
        lines.extend([
            "## Aggregate gains",
            "",
            "| Case | Budget x params | D-opt gain mean +/-95%CI | Latin gain mean +/-95%CI | Reps |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in agg.sort_values(["case_label", "budget_multiplier"]).itertuples(index=False):
            lines.append(
                f"| {row.case_label} | {row.budget_multiplier:.1f} | "
                f"{row.dopt_gain_vs_random_median_pct_mean:.2f}% +/- "
                f"{row.dopt_gain_vs_random_median_pct_ci95:.2f}% | "
                f"{row.latin_gain_vs_random_median_pct_mean:.2f}% +/- "
                f"{row.latin_gain_vs_random_median_pct_ci95:.2f}% | "
                f"{int(row.dopt_gain_vs_random_median_pct_count)} |"
            )
        lines.append("")
    lines.extend([
        "## Figures",
        "",
    ])
    for path in figure_paths:
        if path is not None:
            lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend([
        "",
        "## Interpretation",
        "",
        (
            "A positive D-optimal gain means that the surrogate fitted from "
            "D-optimal NN queries has lower MSE against the NN oracle than the "
            "same-budget random-design median. Because candidates are drawn "
            "from the training-data region and evaluation is held-out in the "
            "same distribution, this experiment directly targets the "
            "close-distance transfer claim."
        ),
    ])
    path = NOTES_DIR / f"{out_prefix}_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(args):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / f"{args.out_prefix}_raw.csv"
    if raw_path.exists() and not args.overwrite:
        raise FileExistsError(f"{raw_path} exists; use --overwrite or change --out-prefix")

    rows = []
    for case in args.cases:
        for data_seed in args.data_seeds:
            for init_seed in args.init_seeds:
                print(
                    f"Running case={case}, data_seed={data_seed}, init_seed={init_seed}",
                    flush=True,
                )
                rows.extend(run_one(args, case, data_seed, init_seed))

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    agg = aggregate_results(df)
    agg_path = DATA_DIR / f"{args.out_prefix}_aggregate.csv"
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")
    figure_paths = [
        plot_close_filter(df, args.out_prefix),
        plot_gain_by_budget(agg, args.out_prefix),
        plot_gain_by_budget_stable(agg, args.out_prefix),
        plot_mse_by_budget(agg, args.out_prefix),
    ]
    report = write_report(df, agg, figure_paths, args.out_prefix)

    print(f"Saved {raw_path}")
    print(f"Saved {agg_path}")
    for path in figure_paths:
        if path is not None:
            print(f"Saved {path}")
    print(f"Saved {report}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test D-optimal design after filtering to NN-FPR-close regimes."
    )
    parser.add_argument("--out-prefix", default="close_distance_dopt")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["paper_poly3", "paper_poly4"],
        choices=list(CASE_LABELS),
    )
    parser.add_argument("--data-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--hidden", type=int, nargs="+", default=[16, 8, 4])
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--optimizer", choices=["rprop", "adam"], default="rprop")
    parser.add_argument("--max-order", type=int, default=None)
    parser.add_argument("--no-special", action="store_true")
    parser.add_argument("--max-shape-distance", type=float, default=5e-4)
    parser.add_argument(
        "--keep-nonclose",
        action="store_true",
        help="Run design comparisons even if a base NN-FPR fit fails the close filter.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=750,
        help="Maximum candidate rows sampled from the training-data region; <=0 uses all.",
    )
    parser.add_argument("--budgets", type=int, nargs="*", default=None)
    parser.add_argument("--budget-multipliers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--dopt-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--surrogate-ridge",
        type=float,
        default=0.0,
        help="Use OLS when 0; use Ridge(alpha=value) when positive.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
