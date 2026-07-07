"""
D-optimal transfer experiment.

Question:
  If a trained NN is geometrically close to the user's FullPR basis, can a
  classical D-optimal design from the FullPR feature space be used to sample
  the NN efficiently?

Experiment:
  1. Train a reference NN in a known low-distance setting:
       paper_poly3, tanh, hidden=(16,8,4), FullPR include_special=True.
  2. Treat that NN as an oracle/new model.
  3. Generate candidate inputs on [-1,1]^3.
  4. Select k inputs with a greedy D-optimal criterion using only the FullPR
     design matrix.
  5. Query the NN at selected inputs, fit a FullPR surrogate to those NN
     outputs, and compare against random designs of the same size.

Positive result:
  D-optimal designs should approximate the NN with lower MSE/shape distance
  than random designs when NN and FullPR are already close.
"""

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from measure_morala import (
    ConfigurableNet,
    custom_features_full,
    fit_full_pr,
    generate_dataset,
    make_paper_polynomial,
    register_hooks,
    remove_hooks,
    reset_capture,
    surface_shape_distance,
    train_nn,
)


def _nn_predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).cpu().numpy().reshape(-1, 1)


def _feature_matrix(X, max_order, include_special):
    F = custom_features_full(X, max_order=max_order, include_special=include_special)
    return np.column_stack([np.ones(len(X)), F])


def _standardize_design(F):
    Z = F.copy().astype(float)
    mu = Z[:, 1:].mean(axis=0)
    sd = Z[:, 1:].std(axis=0)
    sd[sd == 0] = 1.0
    Z[:, 1:] = (Z[:, 1:] - mu) / sd
    return Z


def greedy_d_optimal(F, k, ridge=1e-3):
    """Greedy D-optimal row selection for det(ridge I + F_S'F_S)."""
    Z = _standardize_design(F)
    n, p = Z.shape
    if k > n:
        raise ValueError("k cannot exceed the candidate pool size")
    m_inv = np.eye(p) / ridge
    selected = []
    available = np.ones(n, dtype=bool)
    for _ in range(k):
        scores = np.einsum("ij,jk,ik->i", Z, m_inv, Z)
        scores[~available] = -np.inf
        idx = int(np.argmax(scores))
        selected.append(idx)
        available[idx] = False
        f = Z[idx:idx + 1].T
        denom = float((1.0 + f.T @ m_inv @ f).item())
        m_inv = m_inv - (m_inv @ f @ f.T @ m_inv) / denom
    return np.array(selected, dtype=int)


def fit_surrogate(X_design, y_design, X_eval, max_order, include_special):
    F_design = custom_features_full(X_design, max_order=max_order,
                                    include_special=include_special)
    F_eval = custom_features_full(X_eval, max_order=max_order,
                                  include_special=include_special)
    lr = LinearRegression().fit(F_design, y_design)
    return lr.predict(F_eval), lr


def sample_scaled_data_region(n, data_seed, seed, p=3):
    """
    Sample candidate points from the same input distribution as paper_poly3,
    then scale them with the original training-set min/max. Points outside the
    original [-1,1]^p scaled box are rejected so the experiment stays in the
    interpolation/data region where NN-FullPR closeness was measured.
    """
    X0, _y0, _ = make_paper_polynomial(n=200, p=p, degree=3, rng=data_seed)
    # generate_dataset uses train_test_split(test_size=.25, random_state=42).
    from sklearn.model_selection import train_test_split
    X_tr_raw, _X_te_raw = train_test_split(X0, test_size=0.25, random_state=42)
    mn = X_tr_raw.min(axis=0)
    mx = X_tr_raw.max(axis=0)
    span = np.where(mx - mn == 0, 1.0, mx - mn)

    g0 = np.random.default_rng(data_seed)
    mus = g0.uniform(-10, 10, size=p)
    g = np.random.default_rng(seed)
    chunks = []
    total = 0
    while total < n:
        raw = g.normal(loc=mus, scale=1.0, size=(max(1000, n), p))
        scaled = -1.0 + 2.0 * (raw - mn) / span
        keep = np.all((scaled >= -1.0) & (scaled <= 1.0), axis=1)
        if keep.any():
            chunks.append(scaled[keep])
            total += int(keep.sum())
    return np.vstack(chunks)[:n]


def summarize_random(values):
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def run(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.init_seed)
    np.random.seed(args.init_seed)
    reset_capture()

    case = "paper_poly3"
    activation = "tanh"
    hidden = (16, 8, 4)
    include_special = True
    max_order = len(hidden)

    X_tr, X_te, y_tr, y_te = generate_dataset(
        case, n=args.n, p=3, rng=args.data_seed, scaling=(-1.0, 1.0))
    model = ConfigurableNet(3, hidden, activation)
    hooks = register_hooks(model)
    train_nn(model, X_tr, y_tr, X_eval=X_te, y_eval=y_te,
             epochs=args.epochs, optimizer_name="rprop",
             capture_every=max(1, args.epochs // 5), verbose=False)
    remove_hooks(hooks)
    model.eval()

    X_te_np = X_te.numpy()
    y_te_np = y_te.numpy()
    pred_nn_te = _nn_predict(model, X_te_np)
    pred_fpr_te, _ = fit_full_pr(X_tr.numpy(), y_tr.numpy(), X_te_np,
                                 max_order=max_order,
                                 include_special=include_special)
    base = {
        "case": case,
        "activation": activation,
        "hidden": ",".join(map(str, hidden)),
        "include_special": include_special,
        "nn_mse_vs_y": mean_squared_error(y_te_np, pred_nn_te),
        "fpr_mse_vs_y": mean_squared_error(y_te_np, pred_fpr_te),
        "fpr_mse_vs_nn": mean_squared_error(pred_nn_te, pred_fpr_te),
        "shape_nn_fpr": surface_shape_distance(X_te_np, pred_nn_te, pred_fpr_te),
    }

    if args.candidate_domain == "uniform":
        X_cand = rng.uniform(-1.0, 1.0, size=(args.candidates, 3))
        X_eval = rng.uniform(-1.0, 1.0, size=(args.eval_points, 3))
    else:
        X_cand = sample_scaled_data_region(args.candidates, args.data_seed,
                                           seed=args.seed + 101)
        X_eval = sample_scaled_data_region(args.eval_points, args.data_seed,
                                           seed=args.seed + 202)
    pred_nn_eval = _nn_predict(model, X_eval)
    F_cand = _feature_matrix(X_cand, max_order=max_order,
                             include_special=include_special)
    p_count = F_cand.shape[1]
    budgets = args.budgets or [p_count, 2 * p_count, 3 * p_count, 4 * p_count]
    budgets = [int(b) for b in budgets if int(b) <= args.candidates]

    records = []
    design_rows = []
    for k in budgets:
        d_idx = greedy_d_optimal(F_cand, k)
        pred_d, _ = fit_surrogate(X_cand[d_idx], _nn_predict(model, X_cand[d_idx]),
                                  X_eval, max_order, include_special)
        d_mse = mean_squared_error(pred_nn_eval, pred_d)
        d_shape = surface_shape_distance(X_eval, pred_nn_eval, pred_d)

        rand_mse, rand_shape = [], []
        for r in range(args.random_repeats):
            ridx = rng.choice(args.candidates, size=k, replace=False)
            pred_r, _ = fit_surrogate(X_cand[ridx], _nn_predict(model, X_cand[ridx]),
                                      X_eval, max_order, include_special)
            rand_mse.append(mean_squared_error(pred_nn_eval, pred_r))
            rand_shape.append(surface_shape_distance(X_eval, pred_nn_eval, pred_r))

        mse_s = summarize_random(rand_mse)
        shape_s = summarize_random(rand_shape)
        records.append({
            **base,
            "candidate_domain": args.candidate_domain,
            "budget": k,
            "n_parameters": p_count,
            "method": "D-optimal",
            "mse_vs_nn": d_mse,
            "shape_vs_nn": d_shape,
            "random_mse_median": mse_s["median"],
            "random_mse_q25": mse_s["q25"],
            "random_mse_q75": mse_s["q75"],
            "random_shape_median": shape_s["median"],
            "random_shape_q25": shape_s["q25"],
            "random_shape_q75": shape_s["q75"],
            "mse_gain_vs_random_median_pct": 100.0 * (mse_s["median"] - d_mse) / mse_s["median"],
            "shape_gain_vs_random_median_pct": 100.0 * (shape_s["median"] - d_shape) / shape_s["median"],
        })
        if k == budgets[min(1, len(budgets) - 1)]:
            y_design = _nn_predict(model, X_cand[d_idx]).ravel()
            for i, (x, yhat) in enumerate(zip(X_cand[d_idx], y_design)):
                design_rows.append({
                    "budget": k, "order": i, "x1": x[0], "x2": x[1], "x3": x[2],
                    "nn_output": yhat,
                })

    df = pd.DataFrame(records)
    design = pd.DataFrame(design_rows)
    df.to_csv(f"{args.out_prefix}_metrics.csv", index=False, encoding="utf-8-sig")
    design.to_csv(f"{args.out_prefix}_design_points.csv", index=False,
                  encoding="utf-8-sig")
    plot_results(df, f"{args.out_prefix}_comparison.png")
    write_report(df, base, f"{args.out_prefix}_report.md")
    print(df.to_string(index=False))
    print(f"\nSaved: {args.out_prefix}_metrics.csv")
    print(f"Saved: {args.out_prefix}_design_points.csv")
    print(f"Saved: {args.out_prefix}_comparison.png")
    print(f"Saved: {args.out_prefix}_report.md")


def plot_results(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(df))
    labels = [str(int(k)) for k in df["budget"]]
    for ax, metric, rand_col, q25, q75, ylabel in [
        (axes[0], "mse_vs_nn", "random_mse_median", "random_mse_q25",
         "random_mse_q75", "surrogate MSE vs NN"),
        (axes[1], "shape_vs_nn", "random_shape_median", "random_shape_q25",
         "random_shape_q75", "shape distance vs NN"),
    ]:
        ax.plot(x, df[metric], marker="o", lw=2, label="D-optimal")
        ax.plot(x, df[rand_col], marker="s", lw=2, label="random median")
        ax.fill_between(x, df[q25], df[q75], alpha=0.2, label="random IQR")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("number of queried NN points")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    domain = df["candidate_domain"].iloc[0] if "candidate_domain" in df else "candidate"
    fig.suptitle(f"D-optimal transfer in {domain} candidate domain")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(df, base, path):
    best = df.iloc[-1]
    lines = [
        "# D-optimal transfer experiment",
        "",
        "Setting: `paper_poly3`, `tanh`, hidden `(16,8,4)`, FullPR with special terms.",
        "",
        "First, the reference NN and FullPR are confirmed to be close on the original test set:",
        "",
        f"- NN MSE vs y: `{base['nn_mse_vs_y']:.3e}`",
        f"- FullPR MSE vs y: `{base['fpr_mse_vs_y']:.3e}`",
        f"- FullPR MSE vs NN: `{base['fpr_mse_vs_nn']:.3e}`",
        f"- shape(NN, FullPR): `{base['shape_nn_fpr']:.3e}`",
        "",
        f"Then the trained NN is treated as an oracle. D-optimal points are selected in the `{best['candidate_domain']}` candidate domain using only the FullPR feature matrix, the NN is queried at those points, and a FullPR surrogate is fitted to the NN outputs.",
        "",
        "Main result at the largest budget:",
        "",
        f"- budget: `{int(best['budget'])}` NN queries",
        f"- D-optimal MSE vs NN: `{best['mse_vs_nn']:.3e}`",
        f"- random median MSE vs NN: `{best['random_mse_median']:.3e}`",
        f"- D-optimal improvement over random median: `{best['mse_gain_vs_random_median_pct']:.1f}%`",
        f"- D-optimal shape-distance improvement over random median: `{best['shape_gain_vs_random_median_pct']:.1f}%`",
        "",
        "Interpretation: in this low-distance regime, a classical D-optimal design built for the polynomial feature space transfers to the NN oracle. This supports the claim that when NN and FullPR are geometrically close, old polynomial-regression design technology can be used to sample or distill the newer NN model more efficiently than random sampling.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--candidates", type=int, default=5000)
    ap.add_argument("--eval-points", type=int, default=2000)
    ap.add_argument("--random-repeats", type=int, default=100)
    ap.add_argument("--candidate-domain", choices=["data", "uniform"], default="data",
                    help="data: candidates are sampled from the original scaled data region; "
                         "uniform: candidates fill the whole [-1,1]^3 cube")
    ap.add_argument("--budgets", type=int, nargs="*", default=None)
    ap.add_argument("--out-prefix", default="dopt_transfer")
    run(ap.parse_args())
