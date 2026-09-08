"""
High-dimensional close-distance D/A/I-optimal transfer experiment.

This extends close_distance_highdim_dopt_experiment.py in two directions:

1. Compare D-optimal, A-optimal, and I-optimal design criteria against the
   same-budget random median baseline. Latin hypercube is kept as a geometric
   coverage baseline.
2. Sweep NN activation functions, including an identity linear-network control
   that should be easier for a low-degree FullPR surrogate to mimic.
3. Add error-regularized D/A/I criteria. These use the full-data FullPR mimic
   residual as a simulation-only proxy for NN regions that are hard to mimic.

The experiment keeps the same p values, close-distance filter, budget
definition, candidate region, and output conventions as the previous
close-distance high-dimensional D-optimal experiment. The greedy optimal design
step is sequential, with one point added and the information state updated after
each addition. It can also run a broader case library spanning the
iterative-D-optimal, measure-weighted, and GPU geometry data-generation families
used elsewhere in this project.
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
from scipy.linalg import solve_triangular
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from iterative_highdim_dopt_experiment import (
    CASE_LABELS as BASE_CASE_LABELS,
    build_fullpr_features,
    design_matrix_for_dopt,
    fit_surrogate,
    get_device,
    limited_shape_distance,
    make_highdim_dataset,
    nn_predict,
    train_oracle_nn,
)
from measure_morala import scale_minmax
from measure_morala_gpu import (
    make_highdim_polynomial,
    make_highdim_nonlinear,
)
from measure_weighted_sampling_experiment import custom_transfer_response


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "close_distance_highdim_optimality"
FIG_DIR = ROOT / "figures" / "close_distance_highdim_optimality"
NOTES_DIR = ROOT / "reports" / "notes"

ITERATIVE_CASES = ["highdim_poly2", "highdim_smooth", "highdim_strong"]
MEASURE_WEIGHTED_CASES = ["highdim_local", "highdim_highfreq"]
GEOMETRY_CASES = ["highdim_poly3", "highdim_poly4", "highdim_nonlinear"]
CASE_ORDER = ITERATIVE_CASES + MEASURE_WEIGHTED_CASES + GEOMETRY_CASES
CASE_LABELS = {
    **BASE_CASE_LABELS,
    "highdim_poly2": "Quadratic",
    "highdim_smooth": "Smooth",
    "highdim_strong": "Strong nonlinear",
    "highdim_local": "Local interior",
    "highdim_highfreq": "High-frequency",
    "highdim_poly3": "Sparse cubic",
    "highdim_poly4": "Sparse quartic",
    "highdim_nonlinear": "Random nonlinear",
}
CASE_COLORS = {
    "highdim_poly2": "#4C78A8",
    "highdim_smooth": "#54A24B",
    "highdim_strong": "#B279A2",
    "highdim_local": "#72B7B2",
    "highdim_highfreq": "#E45756",
    "highdim_poly3": "#F58518",
    "highdim_poly4": "#9467BD",
    "highdim_nonlinear": "#8C564B",
}
DATASET_FAMILY = {
    **{case: "iterative_dopt" for case in ITERATIVE_CASES},
    **{case: "measure_weighted" for case in MEASURE_WEIGHTED_CASES},
    **{case: "gpu_geometry" for case in GEOMETRY_CASES},
}
CRITERION_ORDER = [
    "dopt",
    "aopt",
    "iopt",
    "dopt_err",
    "aopt_err",
    "iopt_err",
    "latin",
]
CRITERION_LABELS = {
    "dopt": "D-optimal",
    "aopt": "A-optimal",
    "iopt": "I-optimal",
    "dopt_err": "D-optimal + error reg.",
    "aopt_err": "A-optimal + error reg.",
    "iopt_err": "I-optimal + error reg.",
    "latin": "Latin hypercube",
}
CRITERION_COLORS = {
    "dopt": "#D55E00",
    "aopt": "#0072B2",
    "iopt": "#CC79A7",
    "dopt_err": "#8B2E00",
    "aopt_err": "#00537E",
    "iopt_err": "#8E4C7B",
    "latin": "#009E73",
}
ACTIVATION_ALIASES = {
    "identical": "identity",
    "linear": "identity",
}


def canonical_activation(name: str) -> str:
    return ACTIVATION_ALIASES.get(name.lower(), name.lower())


def stable_seed(*values) -> int:
    seed = 1729
    for value in values:
        if isinstance(value, str):
            items = value.encode("utf-8")
        else:
            items = str(int(value)).encode("ascii")
        for item in items:
            seed = (seed * 1_000_003 + int(item) + 97) % (2**32 - 1)
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


def normalize_criteria(criteria):
    normalized = []
    for criterion in criteria:
        key = criterion.lower().replace("-", "").replace("_", "")
        if key in {"d", "dopt", "doptimal"}:
            criterion = "dopt"
        elif key in {"a", "aopt", "aoptimal"}:
            criterion = "aopt"
        elif key in {"i", "iopt", "ioptimal"}:
            criterion = "iopt"
        elif key in {"dopterr", "dopterror", "doptreg", "doptimalerr", "doptimalerror"}:
            criterion = "dopt_err"
        elif key in {"aopterr", "aopterror", "aoptreg", "aoptimalerr", "aoptimalerror"}:
            criterion = "aopt_err"
        elif key in {"iopterr", "iopterror", "ioptreg", "ioptimalerr", "ioptimalerror"}:
            criterion = "iopt_err"
        elif key in {"latin", "lhs", "latinhypercube"}:
            criterion = "latin"
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        if criterion not in normalized:
            normalized.append(criterion)
    return normalized


def base_optimal_criterion(criterion):
    if criterion.endswith("_err"):
        return criterion.removesuffix("_err")
    return criterion


def is_error_regularized_criterion(criterion):
    return criterion.endswith("_err")


def method_signature(args):
    return ";".join(
        [
            "close-distance-optimality-v2",
            "selection=sequential_ridge_rank1",
            f"hidden={','.join(str(x) for x in args.hidden)}",
            f"degree={args.degree}",
            f"include_special={bool(args.include_special)}",
            f"criteria={','.join(args.criteria)}",
            f"ridge_alpha={args.ridge_alpha:g}",
            f"opt_ridge={args.opt_ridge:g}",
            f"max_shape_distance={args.max_shape_distance:g}",
            f"error_reg_strength={args.error_regularizer_strength:g}",
            f"error_reg_power={args.error_regularizer_power:g}",
            f"error_reg_cap={args.error_regularizer_cap:g}",
        ]
    )


def infer_criteria_from_columns(df):
    found = []
    for criterion in CRITERION_ORDER:
        if f"{criterion}_gain_vs_random_median_pct" in df.columns:
            found.append(criterion)
    return found


def candidate_indices(n_train, max_candidates, seed):
    if max_candidates <= 0 or max_candidates >= n_train:
        return np.arange(n_train, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_train, size=max_candidates, replace=False))


def make_geometry_nonlinear_dataset(n, p, data_seed, noise_sd):
    if p >= 6:
        X, y = make_highdim_nonlinear(
            n=n, p=p, noise_sd=noise_sd, rng=data_seed, randomize=True
        )
        return X, y

    g = np.random.default_rng(data_seed)
    X = g.normal(0, 1, size=(n, p))
    omega = g.uniform(1.5, 3.5)
    phi = g.uniform(0, 2 * np.pi)
    amp = g.uniform(1.0, 2.5)
    y = amp * np.sin(omega * X[:, 0] + phi)
    if p >= 2:
        y += X[:, 1] ** 2
    if p >= 3:
        y += X[:, 1] * X[:, 2]
    if p >= 4:
        y += X[:, 3]
    if p >= 5:
        y += 0.5 * X[:, 4]
    y += g.normal(0, noise_sd, size=n)
    return X, y


def make_experiment_dataset(case, n, p, data_seed, noise_sd):
    if case in ITERATIVE_CASES:
        return make_highdim_dataset(case, n, p, data_seed, noise_sd)

    if case in MEASURE_WEIGHTED_CASES:
        rng_x = np.random.default_rng(data_seed)
        X = rng_x.normal(0.0, 1.0, size=(n, p))
        X_train_raw, X_eval_raw = train_test_split(
            X, test_size=0.25, random_state=42
        )
        X_train, X_eval = scale_minmax(X_train_raw, X_eval_raw, (-1.0, 1.0))
        rng_y_train = np.random.default_rng(data_seed + 40_000)
        rng_y_eval = np.random.default_rng(data_seed + 50_000)
        y_train_raw = custom_transfer_response(
            X_train, case, data_seed, rng_y_train, noise_sd
        )
        y_eval_raw = custom_transfer_response(
            X_eval, case, data_seed, rng_y_eval, noise_sd
        )
        y_train, y_eval = scale_minmax(y_train_raw, y_eval_raw, (-1.0, 1.0))
        return X_train, X_eval, y_train, y_eval

    if case in GEOMETRY_CASES:
        if case == "highdim_poly3":
            X, y = make_highdim_polynomial(
                n=n, p=p, degree=3, noise_sd=noise_sd, rng=data_seed
            )
        elif case == "highdim_poly4":
            X, y = make_highdim_polynomial(
                n=n, p=p, degree=4, noise_sd=noise_sd, rng=data_seed
            )
        elif case == "highdim_nonlinear":
            X, y = make_geometry_nonlinear_dataset(n, p, data_seed, noise_sd)
        else:
            raise ValueError(f"Unknown GPU geometry case: {case}")
        X_train_raw, X_eval_raw, y_train_raw, y_eval_raw = train_test_split(
            X, y.reshape(-1, 1), test_size=0.25, random_state=42
        )
        X_train, X_eval = scale_minmax(X_train_raw, X_eval_raw, (-1.0, 1.0))
        y_train, y_eval = scale_minmax(y_train_raw, y_eval_raw, (-1.0, 1.0))
        return X_train, X_eval, y_train, y_eval

    raise ValueError(f"Unknown data-generation case: {case}")


def info_cholesky(Z_selected, ridge):
    d = Z_selected.shape[1]
    info = Z_selected.T @ Z_selected
    info.flat[:: d + 1] += ridge

    jitter = 0.0
    while True:
        try:
            return np.linalg.cholesky(info)
        except np.linalg.LinAlgError:
            jitter = ridge if jitter == 0.0 else jitter * 10.0
            if jitter <= 0.0:
                jitter = 1e-10
            info.flat[:: d + 1] += jitter


def build_error_regularizer(abs_errors, power, cap):
    errors = np.asarray(abs_errors, dtype=np.float64).reshape(-1)
    if len(errors) == 0 or float(np.nanmax(errors)) < 1e-10:
        return np.zeros_like(errors)
    scale = max(
        float(np.nanmedian(errors)),
        float(np.nanquantile(errors, 0.75)),
        float(np.nanmean(errors)),
        1e-10,
    )
    reg = np.power(np.maximum(errors, 0.0) / scale, power)
    if cap > 0:
        reg = np.minimum(reg, cap)
    return reg.astype(np.float64, copy=False)


def apply_error_regularizer(scores, error_regularizer, strength):
    if error_regularizer is None or strength <= 0:
        return scores
    return scores / (1.0 + strength * error_regularizer)


def optimal_select_sequential(
    Z,
    budget,
    criterion,
    ridge,
    chunk_size,
    batch_size,
    Z_iopt_ref=None,
    error_regularizer=None,
    error_regularizer_strength=0.0,
):
    """Sequential greedy optimal selection.

    The design starts from a ridge prior and adds every point one at a time,
    updating leverage-related state with the Sherman-Morrison rank-one formula.
    The old chunk/batch arguments are accepted only for CLI compatibility; the
    greedy update itself is pointwise from the first selected point.
    """
    _ = chunk_size
    _ = batch_size
    n, d = Z.shape
    if budget > n:
        raise ValueError("budget cannot exceed candidate pool size")
    if ridge <= 0:
        raise ValueError("Sequential ridge-regularized selection requires ridge > 0")
    criterion_base = base_optimal_criterion(criterion)
    if criterion_base not in {"dopt", "aopt", "iopt"}:
        raise ValueError(f"Unknown optimality criterion: {criterion}")
    if criterion_base == "iopt" and Z_iopt_ref is None:
        raise ValueError("I-optimal selection requires Z_iopt_ref")
    if error_regularizer is not None and len(error_regularizer) != n:
        raise ValueError("error_regularizer must have one value per candidate")

    selected_order = []
    remaining = np.arange(n, dtype=int)
    Z_remaining = np.ascontiguousarray(Z[remaining], dtype=np.float64)
    active = np.ones(len(remaining), dtype=bool)

    inv_info = np.eye(d, dtype=np.float64) / ridge
    V = Z_remaining @ inv_info
    leverage = np.einsum("ij,ij->i", V, Z_remaining, optimize=True)
    leverage = np.maximum(leverage, 0.0)

    row_norm = None
    if criterion_base == "aopt":
        row_norm = np.einsum("ij,ij->i", V, V, optimize=True)
        row_norm = np.maximum(row_norm, 0.0)

    V_ref = None
    H_ref = None
    inv_ref_n = None
    if criterion_base == "iopt":
        V_ref = np.ascontiguousarray(Z_iopt_ref @ inv_info, dtype=np.float64)
        H_ref = V_ref @ Z_remaining.T
        inv_ref_n = 1.0 / max(1, len(Z_iopt_ref))

    regularizer = None
    if is_error_regularized_criterion(criterion):
        if error_regularizer is None:
            regularizer = np.zeros(len(remaining), dtype=np.float64)
        else:
            regularizer = np.asarray(error_regularizer[remaining], dtype=np.float64)

    compact_every = 256
    since_compact = 0
    while len(selected_order) < budget:
        denom = np.maximum(1.0 + leverage, 1e-12)
        if criterion_base == "dopt":
            scores = leverage.copy()
        elif criterion_base == "aopt":
            scores = row_norm / denom
        else:
            scores = (
                np.einsum("ij,ij->j", H_ref, H_ref, optimize=True)
                * inv_ref_n
                / denom
            )
        scores = apply_error_regularizer(
            scores, regularizer, error_regularizer_strength
        )
        scores[~active] = -np.inf
        pos = int(np.argmax(scores))
        if not np.isfinite(scores[pos]):
            free = np.flatnonzero(active)
            if len(free) == 0:
                break
            pos = int(free[0])

        selected_order.append(int(remaining[pos]))
        active[pos] = False

        z = Z_remaining[pos].copy()
        v = V[pos].copy()
        lev = max(float(leverage[pos]), 0.0)
        denom_i = max(1.0 + lev, 1e-12)
        g = V @ z

        if criterion_base == "aopt":
            h = V @ v
            v_norm = float(np.dot(v, v))
            row_norm -= 2.0 * (g / denom_i) * h
            row_norm += (g * g / (denom_i * denom_i)) * v_norm
            row_norm = np.maximum(row_norm, 0.0)

        if criterion_base == "iopt":
            u = V_ref @ z
            H_ref -= np.outer(u, g) / denom_i
            V_ref -= np.outer(u / denom_i, v)

        V -= np.outer(g / denom_i, v)
        leverage -= (g * g) / denom_i
        leverage = np.maximum(leverage, 0.0)
        since_compact += 1

        if since_compact >= compact_every and len(selected_order) < budget:
            keep = active
            remaining = remaining[keep]
            Z_remaining = np.ascontiguousarray(Z_remaining[keep])
            V = np.ascontiguousarray(V[keep])
            leverage = leverage[keep]
            if row_norm is not None:
                row_norm = row_norm[keep]
            if H_ref is not None:
                H_ref = np.ascontiguousarray(H_ref[:, keep])
            if regularizer is not None:
                regularizer = regularizer[keep]
            active = np.ones(len(remaining), dtype=bool)
            since_compact = 0

    return np.asarray(selected_order[:budget], dtype=int)


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


def evaluate_surrogate(F_cand, pred_nn_cand, F_eval, pred_nn_eval, X_eval, idx, args, rng):
    pred, _ = fit_surrogate(F_cand[idx], pred_nn_cand[idx], F_eval, args.ridge_alpha)
    mse = float(mean_squared_error(pred_nn_eval, pred))
    shape = limited_shape_distance(X_eval, pred_nn_eval, pred, rng, args.shape_points)
    return mse, shape


def one_design_comparison(args, shared, base, budget):
    X_cand = shared["X_cand"]
    F_cand = shared["F_cand"]
    F_eval = shared["F_eval"]
    Z_cand = shared["Z_cand"]
    pred_nn_cand = shared["pred_nn_cand"]
    pred_nn_eval = shared["pred_nn_eval"]
    X_eval = shared["X_eval"]
    criteria = shared["criteria"]
    selected_paths = shared.get("selected_paths", {})

    row = {
        **base,
        "status": "used_close",
        "candidate_domain": "training_data_region",
        "n_candidates": len(X_cand),
        "budget": budget,
        "budget_multiplier": budget / base["n_design_params"],
        "criteria": ",".join(criteria),
    }

    for criterion in criteria:
        if criterion == "latin":
            idx = latin_candidate_select(
                X_cand,
                budget,
                stable_seed(
                    base["p"],
                    base["data_seed"],
                    base["init_seed"],
                    base["activation"],
                    budget,
                    811,
                ),
            )
        else:
            idx = selected_paths.get(criterion)
            if idx is None:
                idx = optimal_select_sequential(
                    Z_cand,
                    budget,
                    criterion,
                    args.opt_ridge,
                    args.opt_chunk_size,
                    args.opt_batch_size,
                    shared.get("Z_iopt_ref"),
                    shared.get("error_regularizer_cand"),
                    args.error_regularizer_strength,
                )
            else:
                idx = idx[:budget]
        mse, shape = evaluate_surrogate(
            F_cand, pred_nn_cand, F_eval, pred_nn_eval, X_eval, idx, args, shared["rng"]
        )
        row[f"{criterion}_mse_vs_nn"] = mse
        row[f"{criterion}_shape_vs_nn"] = shape

    rand_mse = []
    rand_shape = []
    rng = np.random.default_rng(
        stable_seed(base["p"], base["data_seed"], base["init_seed"], budget, 911)
    )
    for _ in range(args.random_repeats):
        ridx = rng.choice(len(X_cand), size=budget, replace=False)
        mse, shape = evaluate_surrogate(
            F_cand, pred_nn_cand, F_eval, pred_nn_eval, X_eval, ridx, args, rng
        )
        rand_mse.append(mse)
        rand_shape.append(shape)

    mse_s = summarize(rand_mse)
    shape_s = summarize(rand_shape)
    row.update(
        {
            "random_mse_median": mse_s["median"],
            "random_mse_q25": mse_s["q25"],
            "random_mse_q75": mse_s["q75"],
            "random_mse_mean": mse_s["mean"],
            "random_shape_median": shape_s["median"],
            "random_shape_q25": shape_s["q25"],
            "random_shape_q75": shape_s["q75"],
            "random_shape_mean": shape_s["mean"],
        }
    )
    for criterion in criteria:
        row[f"{criterion}_gain_vs_random_median_pct"] = (
            100.0
            * (mse_s["median"] - row[f"{criterion}_mse_vs_nn"])
            / mse_s["median"]
        )
        row[f"{criterion}_shape_gain_vs_random_median_pct"] = (
            100.0
            * (shape_s["median"] - row[f"{criterion}_shape_vs_nn"])
            / shape_s["median"]
        )
    return row


def run_one(args, activation, case, p, data_seed, init_seed, device):
    t0 = time.perf_counter()
    rng = np.random.default_rng(stable_seed(p, data_seed, init_seed, activation, 101))
    hidden = tuple(args.hidden)
    criteria = normalize_criteria(args.criteria)

    X_train, X_eval, y_train, y_eval = make_experiment_dataset(
        case, args.n, p, data_seed, args.noise_sd
    )
    model = train_oracle_nn(
        X_train,
        y_train,
        X_eval,
        y_eval,
        hidden,
        activation,
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
    pred_full_fpr, full_fpr_model = fit_surrogate(
        F_train, pred_nn_train, F_eval, args.ridge_alpha
    )
    pred_full_train = full_fpr_model.predict(F_train).reshape(-1, 1)
    full_fpr_abs_residual_train = np.abs(pred_nn_train - pred_full_train).reshape(-1)
    fpr_mimic_mse = float(mean_squared_error(pred_nn_eval, pred_full_fpr))
    fpr_mse_vs_y = float(mean_squared_error(y_eval, pred_full_fpr))
    shape_nn_fpr = limited_shape_distance(
        X_eval, pred_nn_eval, pred_full_fpr, rng, args.shape_points
    )
    abs_rmse_gap = abs(math.sqrt(oracle_mse_vs_y) - math.sqrt(fpr_mse_vs_y))

    Z_train = design_matrix_for_dopt(F_train)
    Z_eval = design_matrix_for_dopt(F_eval)
    n_features = F_train.shape[1]
    n_design_params = Z_train.shape[1]
    is_close = shape_nn_fpr <= args.max_shape_distance
    base = {
        "case": case,
        "case_label": CASE_LABELS.get(case, case),
        "dataset_family": DATASET_FAMILY.get(case, "unknown"),
        "p": p,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "data_seed": data_seed,
        "init_seed": init_seed,
        "n": args.n,
        "n_train": len(X_train),
        "n_eval": len(X_eval),
        "hidden": ",".join(str(x) for x in hidden),
        "activation": activation,
        "epochs": args.epochs,
        "degree": args.degree,
        "include_special": args.include_special,
        "selection_mode": "sequential_ridge_rank1",
        "method_signature": method_signature(args),
        "error_regularizer_source": "train_abs_nn_minus_full_data_fullpr",
        "error_regularizer_strength": args.error_regularizer_strength,
        "error_regularizer_power": args.error_regularizer_power,
        "error_regularizer_cap": args.error_regularizer_cap,
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
    error_regularizer_cand = build_error_regularizer(
        full_fpr_abs_residual_train[cand_idx],
        args.error_regularizer_power,
        args.error_regularizer_cap,
    )

    Z_iopt_ref = None
    if any(base_optimal_criterion(criterion) == "iopt" for criterion in criteria):
        if args.iopt_reference_size <= 0 or args.iopt_reference_size >= len(Z_eval):
            Z_iopt_ref = Z_eval
        else:
            ref_idx = rng.choice(
                len(Z_eval), size=args.iopt_reference_size, replace=False
            )
            Z_iopt_ref = Z_eval[np.sort(ref_idx)]

    shared = {
        "X_cand": X_cand,
        "F_cand": F_cand,
        "F_eval": F_eval,
        "Z_cand": Z_cand,
        "Z_iopt_ref": Z_iopt_ref,
        "error_regularizer_cand": error_regularizer_cand,
        "pred_nn_cand": pred_nn_cand,
        "pred_nn_eval": pred_nn_eval,
        "X_eval": X_eval,
        "rng": rng,
        "criteria": criteria,
    }

    if args.budgets:
        budgets = [int(x) for x in args.budgets]
    else:
        budgets = [int(round(n_design_params * mult)) for mult in args.budget_multipliers]
    budgets = sorted({b for b in budgets if 2 <= b <= len(X_cand)})
    if not budgets:
        return [{**base, "budget": np.nan, "status": "skipped_no_valid_budget"}]

    max_budget = max(budgets)
    selected_paths = {}
    for criterion in criteria:
        if criterion == "latin":
            continue
        selected_paths[criterion] = optimal_select_sequential(
            Z_cand,
            max_budget,
            criterion,
            args.opt_ridge,
            args.opt_chunk_size,
            args.opt_batch_size,
            Z_iopt_ref,
            error_regularizer_cand,
            args.error_regularizer_strength,
        )
    shared["selected_paths"] = selected_paths

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
    criteria = infer_criteria_from_columns(used)
    group_cols = [
        "activation",
        "dataset_family",
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
        "random_mse_median",
    ]
    for criterion in criteria:
        metrics.extend(
            [
                f"{criterion}_mse_vs_nn",
                f"{criterion}_shape_vs_nn",
                f"{criterion}_gain_vs_random_median_pct",
                f"{criterion}_shape_gain_vs_random_median_pct",
            ]
        )
    metrics = [metric for metric in metrics if metric in used.columns]

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
    base = raw.drop_duplicates(["activation", "p", "case", "data_seed", "init_seed"]).copy()
    p_values = sorted(base["p"].dropna().unique())
    activations = sorted(base["activation"].dropna().unique())
    fig, axes = plt.subplots(
        len(activations),
        len(p_values),
        figsize=(4.1 * len(p_values), 3.6 * len(activations)),
        squeeze=False,
    )
    for i, activation in enumerate(activations):
        for j, p in enumerate(p_values):
            ax = axes[i, j]
            part = base[(base["activation"] == activation) & (base["p"] == p)]
            skipped = part[part["status"] != "used_close"]
            if not skipped.empty:
                ax.scatter(
                    skipped["shape_nn_fpr"],
                    skipped["abs_rmse_gap"],
                    s=38,
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
                    s=44,
                    color=CASE_COLORS.get(case, "#333333"),
                    alpha=0.86,
                    label=CASE_LABELS.get(case, case),
                )
            thresholds = part["close_threshold"].dropna()
            if not thresholds.empty:
                ax.axvline(thresholds.iloc[0], color="#333333", linestyle="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{activation}, p={int(p)}")
            ax.set_xlabel("NN-FPR shape distance")
            if j == 0:
                ax.set_ylabel("NN/FPR RMSE gap")
            ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Close-distance filter by activation and dimension", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / f"{out_prefix}_close_filter.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_criterion_gain_by_p_activation(agg, out_prefix):
    if agg.empty:
        return None
    criteria = [
        criterion
        for criterion in CRITERION_ORDER
        if f"{criterion}_gain_vs_random_median_pct_mean" in agg.columns
    ]
    if not criteria:
        return None

    records = []
    for keys, sub in agg.groupby(["activation", "p", "budget_multiplier"]):
        activation, p, budget_multiplier = keys
        rec = {
            "activation": activation,
            "p": p,
            "budget_multiplier": budget_multiplier,
        }
        for criterion in criteria:
            values = sub[f"{criterion}_gain_vs_random_median_pct_mean"].dropna()
            rec[f"{criterion}_gain"] = float(values.mean()) if len(values) else np.nan
            rec[f"{criterion}_ci95"] = 1.96 * sem(values)
        records.append(rec)
    stats = pd.DataFrame(records)

    p_values = sorted(stats["p"].dropna().unique())
    activations = sorted(stats["activation"].dropna().unique())
    fig, axes = plt.subplots(
        len(activations),
        len(p_values),
        figsize=(4.2 * len(p_values), 3.5 * len(activations)),
        sharey=True,
        squeeze=False,
    )
    for i, activation in enumerate(activations):
        for j, p in enumerate(p_values):
            ax = axes[i, j]
            sub = stats[
                (stats["activation"] == activation) & (stats["p"] == p)
            ].sort_values("budget_multiplier")
            x = sub["budget_multiplier"].to_numpy(dtype=float)
            for criterion in criteria:
                y = sub[f"{criterion}_gain"].to_numpy(dtype=float)
                ci = sub[f"{criterion}_ci95"].to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    color=CRITERION_COLORS[criterion],
                    label=CRITERION_LABELS[criterion],
                )
                ax.fill_between(
                    x,
                    y - ci,
                    y + ci,
                    color=CRITERION_COLORS[criterion],
                    alpha=0.10,
                    linewidth=0,
                )
            ax.axhline(0, color="#444444", linewidth=1)
            ax.set_title(f"{activation}, p={int(p)}")
            ax.set_xlabel("Budget / number of FullPR parameters")
            if j == 0:
                ax.set_ylabel("MSE gain vs random median (%)")
            ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("D/A/I-optimal transfer gains by activation", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / f"{out_prefix}_criterion_gain_by_p_activation.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_focus_heatmap(agg, out_prefix, focus_multiplier):
    if agg.empty:
        return None
    focus = agg[np.isclose(agg["budget_multiplier"], focus_multiplier)].copy()
    if focus.empty:
        return None
    criteria = [
        criterion
        for criterion in CRITERION_ORDER
        if f"{criterion}_gain_vs_random_median_pct_mean" in focus.columns
    ]
    if not criteria:
        return None

    activations = sorted(focus["activation"].dropna().unique())
    p_values = sorted(focus["p"].dropna().unique())
    row_index = pd.MultiIndex.from_tuples(
        [
            (activation, CASE_LABELS[case])
            for activation in activations
            for case in CASE_ORDER
        ],
        names=["activation", "case_label"],
    )

    tables = []
    vmax = 1.0
    for criterion in criteria:
        table = (
            focus.pivot_table(
                index=["activation", "case_label"],
                columns="p",
                values=f"{criterion}_gain_vs_random_median_pct_mean",
                aggfunc="mean",
            )
            .reindex(row_index)
            .reindex(columns=p_values)
        )
        tables.append((criterion, table))
        vals = table.to_numpy(dtype=float)
        if np.isfinite(vals).any():
            vmax = max(vmax, float(np.nanmax(np.abs(vals))))

    fig, axes = plt.subplots(
        1,
        len(criteria),
        figsize=(3.0 * len(criteria) + 3.5, 0.48 * len(row_index) + 2.8),
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    im = None
    ylabels = [f"{act}: {case}" for act, case in row_index]
    for ax, (criterion, table) in zip(axes, tables):
        vals = table.to_numpy(dtype=float)
        im = ax.imshow(vals, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(CRITERION_LABELS[criterion])
        ax.set_xticks(np.arange(len(p_values)))
        ax.set_xticklabels([f"p={int(p)}" for p in p_values], rotation=0)
        ax.set_yticks(np.arange(len(row_index)))
        ax.set_yticklabels(ylabels)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                val = vals[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), label="MSE gain vs random median (%)")
    fig.suptitle(f"Optimality gain at {focus_multiplier:g}x parameter budget")
    fig.tight_layout()
    out = FIG_DIR / f"{out_prefix}_criteria_heatmap_{focus_multiplier:g}x.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_report(raw, agg, figure_paths, out_prefix, focus_multiplier):
    base = raw.drop_duplicates(["activation", "p", "case", "data_seed", "init_seed"]).copy()
    used_base = base[base["status"] == "used_close"]
    skipped = base[base["status"] != "used_close"]
    used_rows = raw[raw["status"] == "used_close"].copy()
    criteria = infer_criteria_from_columns(raw)

    lines = [
        "# 高维近距离 D/A/I-optimal 迁移与 activation 对照报告",
        "",
        "## 实验目的",
        "",
        (
            "本实验在高维 close-distance D-optimal 测试基础上加入 A-optimal "
            "和 I-optimal 选点准则，并加入 error-regularized D/A/I 作为一组"
            "额外对照。NN 使用双隐藏层，使网络深度与二阶 FullPR 对齐；FullPR "
            "默认只使用二阶多项式项，不再加入逐变量 sin/exp 特征。核心问题是："
            "当 NN oracle 与二阶 FullPR 在同一数据区域内距离较近时，不同实验"
            "设计准则是否比同预算随机采样更高效，以及显式惩罚难拟合区域能否"
            "改善 classical optimal design 的稳定性。"
        ),
        "",
        "## 设置",
        "",
        f"- p values：{', '.join(str(int(x)) for x in sorted(raw['p'].dropna().unique()))}。",
        f"- activations：{', '.join(raw['activation'].dropna().unique())}。",
        f"- data families：{', '.join(raw['dataset_family'].dropna().unique())}。",
        f"- cases：{', '.join(raw['case'].dropna().unique())}。",
        f"- criteria：{', '.join(CRITERION_LABELS[c] for c in criteria)}。",
        f"- n：{int(raw['n'].dropna().iloc[0])}；训练/评估：{int(raw['n_train'].dropna().iloc[0])}/{int(raw['n_eval'].dropna().iloc[0])}。",
        f"- NN hidden：{raw['hidden'].dropna().iloc[0]}；epochs={int(raw['epochs'].dropna().iloc[0])}。",
        f"- FullPR：degree={int(raw['degree'].dropna().iloc[0])}，include_special={bool(raw['include_special'].dropna().iloc[0])}。",
        f"- optimal selection：{raw['selection_mode'].dropna().iloc[0]}。",
        f"- error regularizer：source={raw['error_regularizer_source'].dropna().iloc[0]}，strength={raw['error_regularizer_strength'].dropna().iloc[0]:.3g}，power={raw['error_regularizer_power'].dropna().iloc[0]:.3g}，cap={raw['error_regularizer_cap'].dropna().iloc[0]:.3g}。",
        f"- close filter：`shape_nn_fpr <= {raw['close_threshold'].dropna().iloc[0]:.3e}`。",
        f"- base runs：{len(base)}；通过 close filter：{len(used_base)}；跳过：{len(skipped)}。",
        "",
        "## 准则含义",
        "",
        "- D-optimal：增大信息矩阵行列式，减少参数置信椭球体积。",
        "- A-optimal：减少信息矩阵逆矩阵的 trace，偏向降低平均参数方差。",
        "- I-optimal：减少评估区域的平均预测方差，更直接面向目标区域预测稳定性。",
        "- D/A/I + error reg.：把训练域中 `|NN - full-data FullPR|` 作为难拟合代理，对高误差候选点施加分母惩罚。",
        "- Latin hypercube：几何覆盖基线，不使用 FullPR 信息矩阵。",
        "",
    ]

    if not used_base.empty:
        lines.extend(
            [
                "## 近距离筛选摘要",
                "",
                "| activation | family | p | case | used runs | median shape | median FPR mimic MSE | median RMSE gap | design params |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        summary = (
            used_base.groupby(
                ["activation", "dataset_family", "p", "case", "case_label"],
                as_index=False,
            )
            .agg(
                used_runs=("shape_nn_fpr", "size"),
                median_shape=("shape_nn_fpr", "median"),
                median_mimic_mse=("full_fpr_mse_vs_nn", "median"),
                median_gap=("abs_rmse_gap", "median"),
                design_params=("n_design_params", "first"),
            )
            .sort_values(["activation", "dataset_family", "p", "case"])
        )
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.activation} | {row.dataset_family} | {int(row.p)} | "
                f"{row.case_label} | "
                f"{int(row.used_runs)} | {row.median_shape:.3e} | "
                f"{row.median_mimic_mse:.3e} | {row.median_gap:.3e} | "
                f"{int(row.design_params)} |"
            )
        lines.append("")

    if not used_rows.empty and criteria:
        lines.extend(
            [
                "## 总体 gain 摘要",
                "",
                "| activation | p | criterion | mean gain | median gain | win rate | reps |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for activation in sorted(used_rows["activation"].unique()):
            for p in sorted(used_rows["p"].unique()):
                sub = used_rows[(used_rows["activation"] == activation) & (used_rows["p"] == p)]
                if sub.empty:
                    continue
                for criterion in criteria:
                    col = f"{criterion}_gain_vs_random_median_pct"
                    vals = sub[col].dropna()
                    if vals.empty:
                        continue
                    lines.append(
                        f"| {activation} | {int(p)} | {CRITERION_LABELS[criterion]} | "
                        f"{vals.mean():.2f}% | {vals.median():.2f}% | "
                        f"{(vals > 0).mean() * 100:.1f}% | {len(vals)} |"
                    )
        lines.append("")

    if not agg.empty and criteria:
        focus = agg[np.isclose(agg["budget_multiplier"], focus_multiplier)]
        if focus.empty:
            focus = agg[np.isclose(agg["budget_multiplier"], agg["budget_multiplier"].dropna().iloc[0])]
        lines.extend(
            [
                "## 代表预算下的 case-level gain",
                "",
                "| activation | p | case | budget x params | "
                + " | ".join(CRITERION_LABELS[c] for c in criteria)
                + " |",
                "|---|---:|---|---:|"
                + "|".join(["---:"] * len(criteria))
                + "|",
            ]
        )
        for row in focus.sort_values(["activation", "p", "case"]).itertuples(index=False):
            vals = []
            for criterion in criteria:
                vals.append(
                    f"{getattr(row, f'{criterion}_gain_vs_random_median_pct_mean'):.2f}%"
                )
            lines.append(
                f"| {row.activation} | {int(row.p)} | {row.case_label} | "
                f"{row.budget_multiplier:.1f} | " + " | ".join(vals) + " |"
            )
        lines.append("")

    lines.extend(["## 图", ""])
    for path in figure_paths:
        if path is not None:
            lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "## 解读",
            "",
            (
                "这组实验应和上一轮 D-opt-only 结果配合阅读。tanh 条件测试非线性 NN "
                "oracle 下的设计准则稳定性；identity 条件是线性网络对照，用来判断失败"
                "是否来自 NN 非线性与 FullPR basis 的不匹配。若 identity 下 D/A/I-optimal "
                "显著更稳定，而 tanh 下不稳定，则说明主要限制来自 oracle 非线性或 surrogate "
                "欠设定；若两者都不稳定，则更可能是候选区域、预算或 optimality 准则本身"
                "与目标 MSE 不匹配。"
            ),
        ]
    )

    out = NOTES_DIR / f"{out_prefix}_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run(args):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    args.activations = [canonical_activation(x) for x in args.activations]
    args.criteria = normalize_criteria(args.criteria)
    expected_signature = method_signature(args)

    raw_path = DATA_DIR / f"{args.out_prefix}_raw.csv"
    rows = []
    completed = set()
    if raw_path.exists():
        if args.overwrite:
            pass
        elif args.resume:
            old = pd.read_csv(raw_path)
            if "method_signature" not in old.columns:
                raise ValueError(
                    f"{raw_path} was produced by an older method without "
                    "method_signature; use --overwrite or a new --out-prefix."
                )
            old = old[old["method_signature"] == expected_signature].copy()
            old = old[
                old["activation"].isin(args.activations)
                & pd.to_numeric(old["p"], errors="coerce").isin(args.p_values)
                & old["case"].isin(args.cases)
                & pd.to_numeric(old["data_seed"], errors="coerce").isin(args.data_seeds)
                & pd.to_numeric(old["init_seed"], errors="coerce").isin(args.init_seeds)
            ].copy()
            rows = old.to_dict("records")
            key_cols = [
                "activation",
                "p",
                "case",
                "data_seed",
                "init_seed",
                "method_signature",
            ]
            if all(col in old.columns for col in key_cols):
                completed = {
                    tuple(rec[col] for col in key_cols)
                    for rec in rows
                    if all(col in rec and pd.notna(rec[col]) for col in key_cols)
                }
            print(f"Resuming from {raw_path}: {len(completed)} base runs found")
        else:
            raise FileExistsError(
                f"{raw_path} exists; use --overwrite, --resume, or change --out-prefix"
            )

    device = get_device(args.device)
    total = (
        len(args.activations)
        * len(args.p_values)
        * len(args.cases)
        * len(args.data_seeds)
        * len(args.init_seeds)
    )
    done = 0
    for activation in args.activations:
        for p in args.p_values:
            for case in args.cases:
                for data_seed in args.data_seeds:
                    for init_seed in args.init_seeds:
                        done += 1
                        key = (
                            activation,
                            p,
                            case,
                            data_seed,
                            init_seed,
                            expected_signature,
                        )
                        if key in completed:
                            continue
                        print(
                            f"[{done}/{total}] activation={activation} p={p} "
                            f"case={case} data_seed={data_seed} init_seed={init_seed}",
                            flush=True,
                        )
                        rows.extend(
                            run_one(args, activation, case, p, data_seed, init_seed, device)
                        )
                        pd.DataFrame(rows).to_csv(
                            raw_path, index=False, encoding="utf-8-sig"
                        )

    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    agg = aggregate_results(raw)
    agg_path = DATA_DIR / f"{args.out_prefix}_aggregate.csv"
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")

    figure_paths = [
        plot_close_filter(raw, args.out_prefix),
        plot_criterion_gain_by_p_activation(agg, args.out_prefix),
        plot_focus_heatmap(agg, args.out_prefix, args.focus_multiplier),
    ]
    report_path = write_report(
        raw, agg, figure_paths, args.out_prefix, args.focus_multiplier
    )

    print(f"Saved {raw_path}")
    print(f"Saved {agg_path}")
    for path in figure_paths:
        if path is not None:
            print(f"Saved {path}")
    print(f"Saved {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-dimensional close-distance D/A/I-optimal transfer experiment."
    )
    parser.add_argument("--out-prefix", default="close_distance_highdim_optimality_v2")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--p-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=CASE_ORDER,
        choices=CASE_ORDER,
    )
    parser.add_argument("--data-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["softplus", "tanh", "sigmoid", "relu", "identity"],
        help="NN activations to compare. 'identical' and 'linear' alias to identity.",
    )
    parser.add_argument(
        "--criteria",
        nargs="+",
        default=["dopt", "aopt", "iopt", "dopt_err", "aopt_err", "iopt_err", "latin"],
        help=(
            "Design criteria: dopt, aopt, iopt, dopt_err, aopt_err, "
            "iopt_err, latin."
        ),
    )
    parser.add_argument("--noise-sd", type=float, default=0.03)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--include-special", action="store_true", default=False)
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
    parser.add_argument("--opt-ridge", type=float, default=1e-2)
    parser.add_argument("--opt-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--opt-batch-size",
        type=int,
        default=1,
        help="Deprecated compatibility option; optimal design is now pointwise.",
    )
    parser.add_argument("--error-regularizer-strength", type=float, default=1.0)
    parser.add_argument("--error-regularizer-power", type=float, default=1.0)
    parser.add_argument("--error-regularizer-cap", type=float, default=10.0)
    parser.add_argument(
        "--iopt-reference-size",
        type=int,
        default=512,
        help="Evaluation-reference rows for I-optimal scores; <=0 uses all eval rows.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
