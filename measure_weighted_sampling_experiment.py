"""
Measure-weighted batch sampling experiment.

Workflow:
1. Split a generated dataset into train/test.
2. Uniformly sample a small pilot set from the training split.
3. Train an initial single-hidden-layer NN on the pilot set.
4. Measure the distance between that NN and two PR views:
   - FullPR fitted on the same pilot data.
   - Taylor-PR derived from the trained NN weights.
5. Normalize the combined NN-PR distance to [0, 1].
6. Use the complementary closeness weight to decide the next-batch mix:
   a small distance gives a large D-optimal fraction, while a large distance
   gives a larger random fraction.
7. Compare the measure-weighted policy with all-random, all-D-optimal, and
   Latin-hypercube candidate selection under the same batch budget.
"""

import argparse
import copy
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from iterative_highdim_dopt_experiment import (
    CASE_LABELS,
    batch_dopt_select,
    build_fullpr_features,
    design_matrix_for_dopt,
    fit_surrogate,
    get_device,
    make_highdim_dataset,
    nn_predict,
    train_oracle_nn,
)
from measure_morala import (
    eval_taylor_polynomial,
    nn_to_taylor_polynomial,
    scale_minmax,
    surface_shape_distance,
)


BASE_CASES = set(CASE_LABELS)
CASE_LABELS.update(
    {
        "highdim_local": "Local interior",
        "highdim_highfreq": "High-frequency",
    }
)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use NN-PR distance as a weight for mixed random/D-optimal sampling."
    )
    parser.add_argument("--out-prefix", default="measure_weighted_sampling")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["highdim_poly2", "highdim_smooth", "highdim_strong"],
    )
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--init-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--noise-sd", type=float, default=0.03)
    parser.add_argument("--initial-fraction", type=float, default=0.05)
    parser.add_argument("--initial-val-fraction", type=float, default=0.25)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--hidden", nargs="+", type=int, default=[64])
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--init-epochs", type=int, default=None)
    parser.add_argument("--pr-degree", type=int, default=2)
    parser.add_argument("--no-special-pr", action="store_true")
    parser.add_argument("--taylor-order", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1e-4)
    parser.add_argument("--dopt-ridge", type=float, default=1e-6)
    parser.add_argument("--dopt-chunk-size", type=int, default=2048)
    parser.add_argument("--shape-points", type=int, default=256)
    parser.add_argument(
        "--distance-combine",
        choices=["min", "mean", "max"],
        default="min",
        help="How to combine FullPR and Taylor-PR distances before normalization.",
    )
    parser.add_argument("--distance-low-q", type=float, default=0.10)
    parser.add_argument("--distance-high-q", type=float, default=0.90)
    parser.add_argument(
        "--min-dopt-weight",
        type=float,
        default=0.20,
        help="Lower bound for the D-optimal fraction in the measure-weighted policy.",
    )
    parser.add_argument(
        "--max-dopt-weight",
        type=float,
        default=0.90,
        help="Upper bound for the D-optimal fraction in the measure-weighted policy.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def stable_seed(*values):
    seed = 1729
    for value in values:
        seed = (seed * 1_000_003 + int(value) + 97) % (2**32 - 1)
    return int(seed)


def uniform_initial_indices(n_train, fraction, data_seed, init_seed):
    initial_n = max(32, int(round(n_train * fraction)))
    initial_n = min(initial_n, n_train)
    rng = np.random.default_rng(stable_seed(data_seed, init_seed, 11))
    return np.sort(rng.choice(n_train, size=initial_n, replace=False))


def custom_transfer_response(X, case, data_seed, rng, noise_sd):
    p = X.shape[1]
    y = 0.08 * X[:, 0].copy()
    if p >= 4:
        y += 0.06 * X[:, 1] ** 2 - 0.05 * X[:, 2] * X[:, 3]

    if case == "highdim_local":
        center = np.zeros(min(6, p))
        center[:4] = np.array([0.15, -0.20, 0.10, -0.15])
        radius = np.sum((X[:, : len(center)] - center) ** 2, axis=1)
        y += 1.65 * np.exp(-26.0 * radius)
        if p >= 8:
            ridge = X[:, 4] + 0.65 * X[:, 5] - 0.35 * X[:, 6] * X[:, 7]
            y += 0.35 * np.exp(-14.0 * ridge**2)
        y += 0.18 * np.sin(5.0 * X[:, 0] * X[:, 1])
    elif case == "highdim_highfreq":
        y += 0.55 * np.sin(8.0 * X[:, 0] + 0.35 * data_seed)
        if p >= 4:
            y += 0.45 * np.sin(9.0 * X[:, 1] * X[:, 2])
            y += 0.30 * np.cos(7.0 * X[:, 3] - 0.2)
        if p >= 8:
            y += 0.25 * np.sin(5.0 * (X[:, 4] + X[:, 5] - X[:, 6] * X[:, 7]))
    else:
        raise ValueError(f"Unknown custom transfer case: {case}")

    y = y.reshape(-1, 1)
    y += rng.normal(0.0, noise_sd, size=y.shape)
    return y


def make_dataset(case, n, p, data_seed, noise_sd):
    if case in BASE_CASES:
        return make_highdim_dataset(case, n, p, data_seed, noise_sd)

    rng_x = np.random.default_rng(data_seed)
    X = rng_x.normal(0.0, 1.0, size=(n, p))
    X_train_raw, X_test_raw = train_test_split(
        X, test_size=0.25, random_state=42
    )
    X_train, X_test = scale_minmax(X_train_raw, X_test_raw, (-1.0, 1.0))
    rng_y_train = np.random.default_rng(data_seed + 40_000)
    rng_y_test = np.random.default_rng(data_seed + 50_000)
    y_train_raw = custom_transfer_response(
        X_train, case, data_seed, rng_y_train, noise_sd
    )
    y_test_raw = custom_transfer_response(
        X_test, case, data_seed, rng_y_test, noise_sd
    )
    y_train, y_test = scale_minmax(y_train_raw, y_test_raw, (-1.0, 1.0))
    return X_train, X_test, y_train, y_test


def split_initial_indices(initial_idx, val_fraction, data_seed, init_seed):
    idx = np.asarray(initial_idx, dtype=int).copy()
    rng = np.random.default_rng(stable_seed(data_seed, init_seed, 23))
    rng.shuffle(idx)
    val_n = int(round(len(idx) * val_fraction))
    val_n = max(16, min(val_n, len(idx) - 16))
    valid_idx = np.sort(idx[:val_n])
    fit_idx = np.sort(idx[val_n:])
    return fit_idx, valid_idx


def limited_shape_distance(X, pred_a, pred_b, rng, max_points):
    if max_points and len(X) > max_points:
        idx = rng.choice(len(X), size=max_points, replace=False)
        return float(surface_shape_distance(X[idx], pred_a[idx], pred_b[idx]))
    return float(surface_shape_distance(X, pred_a, pred_b))


def combine_distances(d_fpr, d_tpr, mode):
    vals = np.asarray([d_fpr, d_tpr], dtype=float)
    if mode == "min":
        return float(np.nanmin(vals))
    if mode == "mean":
        return float(np.nanmean(vals))
    if mode == "max":
        return float(np.nanmax(vals))
    raise ValueError(f"Unknown distance combine mode: {mode}")


def compute_initial_distance_record(args, case, data_seed, init_seed, device):
    X_train, X_test, y_train, y_test = make_dataset(
        case, args.n, args.p, data_seed, args.noise_sd
    )
    initial_idx = uniform_initial_indices(
        len(X_train), args.initial_fraction, data_seed, init_seed
    )
    fit_idx, valid_idx = split_initial_indices(
        initial_idx, args.initial_val_fraction, data_seed, init_seed
    )

    init_epochs = args.init_epochs or args.epochs
    init_model = train_oracle_nn(
        X_train[fit_idx],
        y_train[fit_idx],
        X_train[valid_idx],
        y_train[valid_idx],
        args.hidden,
        args.activation,
        init_epochs,
        stable_seed(data_seed, init_seed, 101),
        device,
    )
    pred_nn = nn_predict(init_model, X_train[valid_idx], device)
    pred_nn_fit = nn_predict(init_model, X_train[fit_idx], device)

    F_fit, F_valid = build_fullpr_features(
        X_train[fit_idx],
        X_train[valid_idx],
        args.pr_degree,
        not args.no_special_pr,
    )
    pred_fpr, _ = fit_surrogate(F_fit, pred_nn_fit, F_valid, args.ridge_alpha)

    model_cpu = copy.deepcopy(init_model).cpu()
    beta0, betas = nn_to_taylor_polynomial(
        model_cpu, args.activation, args.taylor_order
    )
    pred_tpr = eval_taylor_polynomial(beta0, betas, X_train[valid_idx])

    dist_rng = np.random.default_rng(stable_seed(data_seed, init_seed, 313))
    d_fpr = limited_shape_distance(
        X_train[valid_idx], pred_nn, pred_fpr, dist_rng, args.shape_points
    )
    d_tpr = limited_shape_distance(
        X_train[valid_idx], pred_nn, pred_tpr, dist_rng, args.shape_points
    )
    d_raw = combine_distances(d_fpr, d_tpr, args.distance_combine)

    return {
        "case": case,
        "case_label": CASE_LABELS.get(case, case),
        "p": args.p,
        "n_total": args.n,
        "train_n": len(X_train),
        "test_n": len(X_test),
        "data_seed": data_seed,
        "init_seed": init_seed,
        "initial_n": len(initial_idx),
        "initial_fit_n": len(fit_idx),
        "initial_valid_n": len(valid_idx),
        "fpr_distance": d_fpr,
        "tpr_distance": d_tpr,
        "combined_distance": d_raw,
        "initial_nn_valid_mse": float(mean_squared_error(y_train[valid_idx], pred_nn)),
        "initial_fpr_valid_mse": float(mean_squared_error(y_train[valid_idx], pred_fpr)),
        "initial_tpr_valid_mse": float(mean_squared_error(y_train[valid_idx], pred_tpr)),
        "initial_fpr_mimic_mse": float(mean_squared_error(pred_nn, pred_fpr)),
        "initial_tpr_mimic_mse": float(mean_squared_error(pred_nn, pred_tpr)),
    }


def attach_distance_weights(distance_df, low_q, high_q, min_dopt_weight, max_dopt_weight):
    df = distance_df.copy()
    if not (0.0 <= min_dopt_weight <= max_dopt_weight <= 1.0):
        raise ValueError("Require 0 <= min_dopt_weight <= max_dopt_weight <= 1")
    vals = df["combined_distance"].to_numpy(dtype=float)
    lo = float(np.quantile(vals, low_q))
    hi = float(np.quantile(vals, high_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        df["distance_norm"] = 0.5
    else:
        df["distance_norm"] = np.clip((df["combined_distance"] - lo) / (hi - lo), 0, 1)
    df["raw_dopt_weight"] = 1.0 - df["distance_norm"]
    df["dopt_weight"] = min_dopt_weight + (
        max_dopt_weight - min_dopt_weight
    ) * df["raw_dopt_weight"]
    df["normalization_low"] = lo
    df["normalization_high"] = hi
    return df


def latin_candidate_select(X_train, selected_mask, batch_size, rng):
    remaining = np.flatnonzero(~selected_mask)
    if len(remaining) == 0:
        return np.array([], dtype=int)
    batch_size = min(batch_size, len(remaining))

    sampler = qmc.LatinHypercube(
        d=X_train.shape[1], seed=int(rng.integers(0, 2**32 - 1))
    )
    unit_targets = sampler.random(batch_size)
    lo = X_train[remaining].min(axis=0)
    hi = X_train[remaining].max(axis=0)
    targets = qmc.scale(unit_targets, lo, hi)

    n_neighbors = min(max(10, batch_size // 4), len(remaining))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nbrs.fit(X_train[remaining])
    _, neigh = nbrs.kneighbors(targets, return_distance=True)

    picked_pos = []
    used = set()
    for row in neigh:
        chosen = None
        for pos in row:
            pos = int(pos)
            if pos not in used:
                chosen = pos
                break
        if chosen is not None:
            used.add(chosen)
            picked_pos.append(chosen)
        if len(picked_pos) == batch_size:
            break

    if len(picked_pos) < batch_size:
        all_pos = np.arange(len(remaining))
        free = np.array([pos for pos in all_pos if int(pos) not in used], dtype=int)
        fill = rng.choice(free, size=batch_size - len(picked_pos), replace=False)
        picked_pos.extend([int(pos) for pos in fill])

    return remaining[np.asarray(picked_pos, dtype=int)]


def select_next_batch(args, strategy, X_train, Z, selected_mask, dopt_weight, rng):
    remaining = np.flatnonzero(~selected_mask)
    if len(remaining) == 0:
        return np.array([], dtype=int), 0, 0, 0
    batch_size = min(args.batch_size, len(remaining))

    if strategy == "random":
        idx = rng.choice(remaining, size=batch_size, replace=False)
        return np.asarray(idx, dtype=int), 0, batch_size, 0

    if strategy == "dopt":
        idx = batch_dopt_select(
            Z, selected_mask, batch_size, args.dopt_ridge, args.dopt_chunk_size
        )
        return np.asarray(idx, dtype=int), len(idx), 0, 0

    if strategy == "latin":
        idx = latin_candidate_select(X_train, selected_mask, batch_size, rng)
        return np.asarray(idx, dtype=int), 0, 0, len(idx)

    if strategy == "measure_weighted":
        n_dopt = int(round(batch_size * float(dopt_weight)))
        n_dopt = max(0, min(batch_size, n_dopt))
        parts = []
        actual_dopt = 0
        if n_dopt > 0:
            dopt_idx = batch_dopt_select(
                Z, selected_mask, n_dopt, args.dopt_ridge, args.dopt_chunk_size
            )
            if len(dopt_idx) > 0:
                parts.append(dopt_idx)
                actual_dopt = len(dopt_idx)

        tmp_mask = selected_mask.copy()
        if parts:
            tmp_mask[np.concatenate(parts)] = True
        random_need = batch_size - actual_dopt
        actual_random = 0
        if random_need > 0:
            remain_after_dopt = np.flatnonzero(~tmp_mask)
            random_need = min(random_need, len(remain_after_dopt))
            if random_need > 0:
                random_idx = rng.choice(
                    remain_after_dopt, size=random_need, replace=False
                )
                parts.append(np.asarray(random_idx, dtype=int))
                actual_random = random_need

        if not parts:
            return np.array([], dtype=int), 0, 0, 0
        return np.concatenate(parts), actual_dopt, actual_random, 0

    raise ValueError(f"Unknown strategy: {strategy}")


def run_strategy_path(args, dist_row, strategy, device):
    case = dist_row["case"]
    data_seed = int(dist_row["data_seed"])
    init_seed = int(dist_row["init_seed"])
    dopt_weight = float(dist_row["dopt_weight"])
    X_train, X_test, y_train, y_test = make_dataset(
        case, args.n, args.p, data_seed, args.noise_sd
    )
    F_train, _ = build_fullpr_features(
        X_train, X_test, args.pr_degree, not args.no_special_pr
    )
    Z = design_matrix_for_dopt(F_train)

    initial_idx = uniform_initial_indices(
        len(X_train), args.initial_fraction, data_seed, init_seed
    )
    selected_mask = np.zeros(len(X_train), dtype=bool)
    selected_mask[initial_idx] = True
    rng = np.random.default_rng(
        stable_seed(data_seed, init_seed, list(STRATEGY_LABELS).index(strategy), 997)
    )

    rows = []
    previous_dopt = 0
    previous_random = 0
    previous_latin = 0
    for round_id in range(args.rounds + 1):
        selected_idx = np.flatnonzero(selected_mask)
        train_seed = stable_seed(
            data_seed,
            init_seed,
            round_id,
            list(STRATEGY_LABELS).index(strategy),
            601,
        )
        model = train_oracle_nn(
            X_train[selected_idx],
            y_train[selected_idx],
            X_test,
            y_test,
            args.hidden,
            args.activation,
            args.epochs,
            train_seed,
            device,
        )
        pred_test = nn_predict(model, X_test, device)
        test_mse = float(mean_squared_error(y_test, pred_test))

        rows.append(
            {
                "case": case,
                "case_label": CASE_LABELS.get(case, case),
                "p": args.p,
                "n_total": args.n,
                "train_n": len(X_train),
                "test_n": len(X_test),
                "data_seed": data_seed,
                "init_seed": init_seed,
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "round": round_id,
                "selected_n": len(selected_idx),
                "test_mse": test_mse,
                "fpr_distance": float(dist_row["fpr_distance"]),
                "tpr_distance": float(dist_row["tpr_distance"]),
                "combined_distance": float(dist_row["combined_distance"]),
                "distance_norm": float(dist_row["distance_norm"]),
                "dopt_weight": dopt_weight,
                "batch_dopt_added": previous_dopt,
                "batch_random_added": previous_random,
                "batch_latin_added": previous_latin,
            }
        )

        if round_id == args.rounds:
            break

        batch_idx, previous_dopt, previous_random, previous_latin = select_next_batch(
            args, strategy, X_train, Z, selected_mask, dopt_weight, rng
        )
        selected_mask[batch_idx] = True

    return rows


def summarize_results(raw_df, rounds):
    final = raw_df[raw_df["round"] == rounds].copy()
    random_final = final[final["strategy"] == "random"][
        ["case", "data_seed", "init_seed", "test_mse"]
    ].rename(columns={"test_mse": "random_final_mse"})
    final = final.merge(random_final, on=["case", "data_seed", "init_seed"], how="left")
    final["gain_vs_random_pct"] = (
        (final["random_final_mse"] - final["test_mse"]) / final["random_final_mse"] * 100.0
    )

    summary = (
        final.groupby(["strategy", "strategy_label"], as_index=False)
        .agg(
            final_mse_mean=("test_mse", "mean"),
            final_mse_std=("test_mse", "std"),
            gain_vs_random_mean=("gain_vs_random_pct", "mean"),
            gain_vs_random_std=("gain_vs_random_pct", "std"),
            positive_gain_rate=("gain_vs_random_pct", lambda x: float((x > 0).mean())),
            runs=("test_mse", "size"),
        )
        .sort_values("final_mse_mean")
    )
    summary["final_mse_se"] = summary["final_mse_std"] / np.sqrt(summary["runs"])
    summary["gain_vs_random_se"] = summary["gain_vs_random_std"] / np.sqrt(summary["runs"])
    return final, summary


def plot_mse_curves(raw_df, out_prefix):
    cases = list(raw_df["case"].drop_duplicates())
    fig, axes = plt.subplots(
        1, len(cases), figsize=(5.2 * len(cases), 4.0), sharey=False
    )
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        sub = raw_df[raw_df["case"] == case]
        stats = (
            sub.groupby(["strategy", "strategy_label", "round"], as_index=False)
            .agg(mean=("test_mse", "mean"), std=("test_mse", "std"), n=("test_mse", "size"))
        )
        stats["se"] = stats["std"].fillna(0.0) / np.sqrt(stats["n"])
        for strategy in STRATEGY_LABELS:
            one = stats[stats["strategy"] == strategy].sort_values("round")
            if one.empty:
                continue
            x = one["round"].to_numpy()
            y = one["mean"].to_numpy()
            se = one["se"].to_numpy()
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
                y - 1.96 * se,
                y + 1.96 * se,
                color=STRATEGY_COLORS[strategy],
                alpha=0.14,
                linewidth=0,
            )
        ax.set_title(CASE_LABELS.get(case, case))
        ax.set_xlabel("Batch round")
        ax.set_ylabel("Test MSE")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    path = Path(f"{out_prefix}_mse_curve.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_final_mse(final_df, out_prefix):
    stats = (
        final_df.groupby(["strategy", "strategy_label"], as_index=False)
        .agg(mean=("test_mse", "mean"), std=("test_mse", "std"), n=("test_mse", "size"))
    )
    stats["se"] = stats["std"].fillna(0.0) / np.sqrt(stats["n"])
    stats = stats.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = [STRATEGY_COLORS[s] for s in stats["strategy"]]
    ax.bar(stats["strategy_label"], stats["mean"], color=colors, alpha=0.88)
    ax.errorbar(
        stats["strategy_label"],
        stats["mean"],
        yerr=1.96 * stats["se"],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.0,
    )
    ax.set_ylabel("Final test MSE")
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    path = Path(f"{out_prefix}_final_mse.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_weight_gain(final_df, out_prefix):
    weighted = final_df[final_df["strategy"] == "measure_weighted"].copy()
    if weighted.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for case, sub in weighted.groupby("case"):
        ax.scatter(
            sub["dopt_weight"],
            sub["gain_vs_random_pct"],
            s=64,
            alpha=0.9,
            label=CASE_LABELS.get(case, case),
        )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.55)
    ax.set_xlabel("D-optimal fraction from NN-PR closeness")
    ax.set_ylabel("Final gain vs random (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(f"{out_prefix}_weight_vs_gain.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def simple_markdown_table(df):
    columns = [str(col) for col in df.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[col]).replace("|", "\\|") for col in df.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(args, distance_df, final_df, summary_df, plot_paths, out_prefix, elapsed):
    best = summary_df.iloc[0]
    weighted = summary_df[summary_df["strategy"] == "measure_weighted"]
    weighted_text = ""
    if not weighted.empty:
        row = weighted.iloc[0]
        weighted_text = (
            f"- Measure-weighted final MSE: {row['final_mse_mean']:.6f}; "
            f"mean gain vs random: {row['gain_vs_random_mean']:.3f}% "
            f"(positive-rate {row['positive_gain_rate'] * 100:.1f}%)."
        )

    summary_show = summary_df.copy()
    for col in [
        "final_mse_mean",
        "final_mse_std",
        "gain_vs_random_mean",
        "gain_vs_random_std",
        "positive_gain_rate",
    ]:
        summary_show[col] = summary_show[col].map(lambda x: f"{x:.6g}")

    dist_show = distance_df[
        [
            "case_label",
            "data_seed",
            "init_seed",
            "fpr_distance",
            "tpr_distance",
            "combined_distance",
            "distance_norm",
            "dopt_weight",
        ]
    ].copy()
    for col in [
        "fpr_distance",
        "tpr_distance",
        "combined_distance",
        "distance_norm",
        "dopt_weight",
    ]:
        dist_show[col] = dist_show[col].map(lambda x: f"{x:.6g}")

    lines = [
        "# Measure-weighted sampling experiment",
        "",
        "## Design",
        "",
        (
            "For every dataset, the training split is first reduced by uniform "
            f"sampling to {args.initial_fraction:.3g} of the training data. "
            "A single-hidden-layer tanh NN is trained on that pilot subset."
        ),
        "",
        (
            "The pilot NN is compared with FullPR and Taylor-PR on a held-out "
            "part of the pilot subset. FullPR is fitted to mimic the pilot NN "
            "predictions, so the distance is a model-to-model transfer measure. "
            "The combined distance is normalized to [0, 1] across all pilot runs. "
            "The actual D-optimal batch fraction is a bounded closeness score, "
            "`min_dopt + (max_dopt - min_dopt) * (1 - distance_norm)`, so a closer "
            "NN-PR relation gives more D-optimal points and a farther relation "
            "still keeps a small random/D-optimal mixture."
        ),
        "",
        "The controls use the same initial subset and the same batch size:",
        "",
        "- Random: every new batch is uniform random.",
        "- D-optimal: every new batch is selected by approximate D-optimal leverage.",
        "- Latin hypercube: every new batch follows a Latin-hypercube coverage target over the candidate pool.",
        "- Measure-weighted: a fixed-size mixed batch using the learned distance weight.",
        "",
        "## Configuration",
        "",
        f"- n={args.n}, p={args.p}, train/test split=75/25.",
        f"- Cases: {', '.join(args.cases)}.",
        f"- Data seeds: {args.data_seeds}; initial seeds: {args.init_seeds}.",
        f"- Initial fraction={args.initial_fraction}, batch size={args.batch_size}, rounds={args.rounds}.",
        f"- Measure-weighted D-opt bounds=[{args.min_dopt_weight}, {args.max_dopt_weight}].",
        f"- NN hidden={args.hidden}, activation={args.activation}, epochs={args.epochs}.",
        f"- PR degree={args.pr_degree}, Taylor order={args.taylor_order}.",
        "",
        "## Pilot distances and sampling weights",
        "",
        simple_markdown_table(dist_show),
        "",
        "## Final comparison",
        "",
        simple_markdown_table(
            summary_show[
                [
                    "strategy_label",
                    "final_mse_mean",
                    "final_mse_std",
                    "gain_vs_random_mean",
                    "gain_vs_random_std",
                    "positive_gain_rate",
                    "runs",
                ]
            ]
        ),
        "",
        "## Key result",
        "",
        f"- Best mean final MSE: {best['strategy_label']} ({best['final_mse_mean']:.6f}).",
        weighted_text,
        "",
        "## Figures",
        "",
    ]
    for path in plot_paths:
        if path is not None:
            lines.append(f"- {path}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This experiment tests the measure as a policy variable rather "
                "than only as a post-hoc diagnostic. If the measure-weighted curve "
                "beats or matches the best control while avoiding the failure cases "
                "of all-D-optimal selection, it supports the claim that the distance "
                "measure can decide when old PR/D-optimal technology is transferable "
                "to a new NN model."
            ),
            "",
            f"Runtime: {elapsed / 60:.2f} minutes.",
            "",
        ]
    )
    path = Path(f"{out_prefix}_report.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    args = parse_args()
    started = time.time()
    device = get_device(args.device)
    print(f"[INFO] device={device}")
    print(
        f"[INFO] n={args.n}, p={args.p}, cases={args.cases}, "
        f"data_seeds={args.data_seeds}, init_seeds={args.init_seeds}"
    )

    distance_records = []
    for case in args.cases:
        for data_seed in args.data_seeds:
            for init_seed in args.init_seeds:
                print(
                    f"[DIST] case={case} data_seed={data_seed} init_seed={init_seed}"
                )
                record = compute_initial_distance_record(
                    args, case, data_seed, init_seed, device
                )
                distance_records.append(record)

    distance_df = attach_distance_weights(
        pd.DataFrame(distance_records),
        args.distance_low_q,
        args.distance_high_q,
        args.min_dopt_weight,
        args.max_dopt_weight,
    )
    distance_path = Path(f"{args.out_prefix}_distances.csv")
    distance_df.to_csv(distance_path, index=False)
    print(f"[INFO] wrote {distance_path}")

    all_rows = []
    strategies = ["measure_weighted", "random", "dopt", "latin"]
    for _, dist_row in distance_df.iterrows():
        for strategy in strategies:
            print(
                "[RUN] "
                f"case={dist_row['case']} data_seed={int(dist_row['data_seed'])} "
                f"init_seed={int(dist_row['init_seed'])} strategy={strategy} "
                f"dopt_weight={float(dist_row['dopt_weight']):.3f}"
            )
            rows = run_strategy_path(args, dist_row, strategy, device)
            all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    raw_path = Path(f"{args.out_prefix}_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"[INFO] wrote {raw_path}")

    final_df, summary_df = summarize_results(raw_df, args.rounds)
    final_path = Path(f"{args.out_prefix}_final.csv")
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    final_df.to_csv(final_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] wrote {final_path}")
    print(f"[INFO] wrote {summary_path}")

    plot_paths = []
    if not args.skip_plots:
        plot_paths.append(plot_mse_curves(raw_df, args.out_prefix))
        plot_paths.append(plot_final_mse(final_df, args.out_prefix))
        plot_paths.append(plot_weight_gain(final_df, args.out_prefix))
        for path in plot_paths:
            if path is not None:
                print(f"[INFO] wrote {path}")

    elapsed = time.time() - started
    report_path = write_report(
        args, distance_df, final_df, summary_df, plot_paths, args.out_prefix, elapsed
    )
    print(f"[INFO] wrote {report_path}")
    print(f"[DONE] elapsed={elapsed / 60:.2f} min")


if __name__ == "__main__":
    main()
