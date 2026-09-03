"""
measure_morala_gpu.py
─────────────────────────────────────────────────────────────────
大规模 GPU 版本：n = 10000 样本、p = 10/20/50/100/200 五组输入维度的模拟。

与 measure_morala.py（小规模 CPU 版）的关键差异：
  1. 神经网络训练在 GPU 上进行（此规模 GPU 才真正有意义：
     10000×高维输入的数据 + 较宽的网络，矩阵乘法量足够大）；
  2. 用 mini-batch Adam 而非全批量 Rprop（10000 样本全批量 Rprop
     在高维下不稳定，且 mini-batch 更能发挥 GPU 吞吐）；
  3. 几何距离（Procrustes / PCA / Mahalanobis）仍在 CPU 上用
     scipy/sklearn 计算——这部分无法 GPU 化，但相对训练已是小头；
  4. 高维下完整交互多项式会组合爆炸，故 Full-PR 的 max_order
     会同时受 SAFE_MAX_ORDER 和 FULLPR_FEATURE_CAP 保护，避免特征数失控。

用法（Windows）：
  python measure_morala_gpu.py                     # 默认实验组
  python measure_morala_gpu.py --device cuda       # 强制 GPU
  python measure_morala_gpu.py --n 10000 --p-values 10 20 50 100 200 --epochs 300
  python measure_morala_gpu.py --n 10000 --p 20 --epochs 300  # 单组 p 调试
  python measure_morala_gpu.py --repeats 20 --out-prefix gpu_results

先确认 GPU 可用：
  python -c "import torch; print(torch.cuda.is_available())"   # 须为 True
若为 False，请装 CUDA 版 torch：
  pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

import argparse
import math
import os
from itertools import combinations_with_replacement, product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.spatial import procrustes
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 逐层泰勒展开模块（多隐层情形的多项式对照）
try:
    from taylor_expand import expand_nn_to_polynomial
    _HAS_TAYLOR_EXPAND = True
except ImportError:
    _HAS_TAYLOR_EXPAND = False

# 高维下完整交互多项式的安全阶数上限：
#   C(20+order, order)：order=2→231, order=3→1771, order=4→10626。
#   超过 4 阶特征数过大、内存与拟合都吃不消，故封顶 3（max_order=min(隐层数, 3)）。
SAFE_MAX_ORDER = 3
FULLPR_FEATURE_CAP = 30_000
DEFAULT_P_VALUES = [10, 20, 50, 100, 200]


# ─────────────────────────────────────────────
# 设备
# ─────────────────────────────────────────────
def get_device(pref="auto"):
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但 torch.cuda.is_available()=False。"
                               "请安装 CUDA 版 PyTorch。")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_fullpr_feature_count(p, max_order, include_special=True):
    n_poly = math.comb(p + max_order, max_order) - 1
    n_special = 2 * p if include_special else 0
    return n_poly + n_special


def choose_fullpr_order(p, n_hidden, include_special=True):
    requested = min(n_hidden, SAFE_MAX_ORDER)
    for order in range(requested, 0, -1):
        if estimate_fullpr_feature_count(p, order, include_special) <= FULLPR_FEATURE_CAP:
            return order
    return 1


# ─────────────────────────────────────────────
# 激活函数
# ─────────────────────────────────────────────
ACTIVATIONS = {
    "softplus": nn.Softplus, "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid, "relu": nn.ReLU,
}


def activation_derivs_at_zero(act_name, q):
    if act_name == "relu":
        raise ValueError("ReLU 在 0 点不可导，无法泰勒展开。")
    act = ACTIVATIONS[act_name]()
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    g = act(x)
    derivs = [g.item()]
    cur = g
    for _ in range(q):
        (cur,) = torch.autograd.grad(cur.sum(), x, create_graph=True)
        derivs.append(cur.item())
    return derivs


# ─────────────────────────────────────────────
# 数据生成（高维版）
# ─────────────────────────────────────────────
def poly_interaction_features(X, max_order):
    """所有总次数 1..max_order 的交互多项式特征。
    用 combinations_with_replacement 高效枚举（不做 p^order 的笛卡尔积）。"""
    n, p = X.shape
    feats, powers = [], []
    for total in range(1, max_order + 1):
        for combo in combinations_with_replacement(range(p), total):
            exps = np.zeros(p, dtype=int)
            for idx in combo:
                exps[idx] += 1
            feats.append(np.prod(X ** exps, axis=1))
            powers.append(tuple(exps))
    return np.column_stack(feats), powers


def make_highdim_polynomial(n=10000, p=20, degree=3, noise_sd=0.1, rng=0,
                            n_active=8):
    """
    高维多项式真值。为避免高维 degree=3/4 的系数维度爆炸导致信号被稀释，
    只让 n_active 个变量真正参与高阶交互（其余为噪声变量，更贴近真实高维数据）。
      X_i ~ N(mu_i, 1)，mu_i ~ U(-3, 3)（比低维版收窄，控制高阶项幅度）
      Y = 仅由前 n_active 个变量的 degree 阶多项式生成 + 噪声
    """
    g = np.random.default_rng(rng)
    mus = g.uniform(-3, 3, size=p)
    X = g.normal(loc=mus, scale=1.0, size=(n, p))
    Xa = X[:, :n_active]
    design, _ = poly_interaction_features(Xa, max_order=degree)
    betas = g.uniform(-2, 2, size=design.shape[1])
    beta0 = g.uniform(-2, 2)
    y = beta0 + design @ betas + g.normal(0, noise_sd, size=n)
    return X, y


def make_highdim_nonlinear(n=10000, p=20, noise_sd=0.1, rng=0, randomize=True):
    """高维非多项式真值（严格欠设定）：含随机频率 sin + 交互。"""
    g = np.random.default_rng(rng)
    X = g.normal(0, 1, size=(n, p))
    if randomize:
        omega = g.uniform(1.5, 3.5)
        phi = g.uniform(0, 2 * np.pi)
        amp = g.uniform(1.0, 2.5)
        s = amp * np.sin(omega * X[:, 0] + phi)
    else:
        s = np.sin(X[:, 0])
    y = (s + X[:, 1] ** 2 + X[:, 1] * X[:, 2] + X[:, 3]
         + 0.5 * X[:, 4] * X[:, 5] + g.normal(0, noise_sd, size=n))
    return X, y


def scale_minmax(train, test, interval=(-1.0, 1.0)):
    lo, hi = interval
    mn, mx = train.min(axis=0), train.max(axis=0)
    span = np.where(mx - mn == 0, 1.0, mx - mn)
    s = lambda z: lo + (hi - lo) * (z - mn) / span
    return s(train), s(test)


def generate_dataset(case, n, p, rng, scaling=(-1.0, 1.0)):
    if case == "highdim_poly3":
        X, y = make_highdim_polynomial(n=n, p=p, degree=3, rng=rng)
    elif case == "highdim_poly4":
        X, y = make_highdim_polynomial(n=n, p=p, degree=4, rng=rng)
    elif case == "highdim_nonlinear":
        X, y = make_highdim_nonlinear(n=n, p=p, rng=rng, randomize=True)
    else:
        raise ValueError(f"未知数据集：{case}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y.reshape(-1, 1), test_size=0.25, random_state=42)
    X_tr, X_te = scale_minmax(X_tr, X_te, scaling)
    y_tr, y_te = scale_minmax(y_tr, y_te, scaling)
    return X_tr, X_te, y_tr, y_te


# ─────────────────────────────────────────────
# 网络（GPU）
# ─────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, input_size, hidden_layers, act_name):
        super().__init__()
        act = ACTIVATIONS[act_name]
        layers, prev = [], input_size
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_nn_gpu(model, X_tr, y_tr, device, epochs=300, batch_size=512,
                 lr=1e-3, verbose=False):
    """mini-batch Adam 训练（GPU）。返回训练好的模型（在 device 上）。"""
    model = model.to(device)
    X_tr = X_tr.to(device)
    y_tr = y_tr.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    n = X_tr.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.shape[0]
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {epoch+1}/{epochs}  train MSE {tot/n:.6f}")
    return model


# ─────────────────────────────────────────────
# 几何距离（CPU；与小规模版一致的 Procrustes）
# ─────────────────────────────────────────────
def shape_distance(A, B):
    """Procrustes 形状距离（全部点参与对齐）。同形状要求。"""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    if A.shape[0] < 2:
        return float("nan")
    if A.shape[1] != B.shape[1]:
        k = min(A.shape[1], B.shape[1], A.shape[0] - 1)
        if k < 1:
            return float("nan")
        A = PCA(n_components=k).fit_transform(A)
        B = PCA(n_components=k).fit_transform(B)
    if A.shape[1] == 1:
        A = np.hstack([A, np.zeros((A.shape[0], 1))])
        B = np.hstack([B, np.zeros((B.shape[0], 1))])
    if np.allclose(A - A.mean(0), 0) or np.allclose(B - B.mean(0), 0):
        return float("nan")
    _, _, disp = procrustes(A, B)
    return max(float(disp), 0.0)


def surface_shape_distance(X, predA, predB, sample=2000):
    """响应曲面 [X|ŷ] 的形状距离。n 很大时抽样 sample 个点以控成本。"""
    n = X.shape[0]
    if n > sample:
        idx = np.random.default_rng(0).choice(n, sample, replace=False)
        X, predA, predB = X[idx], predA[idx], predB[idx]
    GA = np.hstack([X, predA.reshape(-1, 1)])
    GB = np.hstack([X, predB.reshape(-1, 1)])
    return shape_distance(GA, GB)


def mahalanobis_mean(A, B, ridge=1e-6):
    from scipy.spatial import distance
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    cov = LedoitWolf().fit(np.vstack([A, B])).covariance_
    VI = np.linalg.pinv(np.atleast_2d(cov) + ridge * np.eye(cov.shape[0]))
    return float(distance.mahalanobis(A.mean(0), B.mean(0), VI))


def activation_io_distances(model, X, device, sample=2000):
    """每个激活层 u → g(u) 的 Procrustes 距离（已训练网络）。
    n 大时抽样，且只前向一次用 hook 抓 input/output。"""
    n = X.shape[0]
    if n > sample:
        idx = np.random.default_rng(1).choice(n, sample, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    captured = []
    handles = []

    def mk(name):
        def hook(m, inp, out):
            captured.append((name, inp[0].detach().cpu().numpy(),
                             out.detach().cpu().numpy()))
        return hook

    act_types = (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softplus)
    for name, m in model.named_modules():
        if isinstance(m, act_types):
            handles.append(m.register_forward_hook(mk(name)))
    model.eval()
    with torch.no_grad():
        model(torch.tensor(Xs, dtype=torch.float32, device=device))
    for h in handles:
        h.remove()
    captured.sort(key=lambda t: int(t[0].split(".")[1]))
    return {name: shape_distance(u, gu) for name, u, gu in captured}


# ─────────────────────────────────────────────
# Taylor-PR（仅单隐层）与 Full-PR
# ─────────────────────────────────────────────
def all_multi_indices(p, q):
    for t in range(1, q + 1):
        for combo in combinations_with_replacement(range(p), t):
            m = [0] * p
            for idx in combo:
                m[idx] += 1
            yield tuple(m)


def nn_to_taylor_polynomial(model, act_name, q, p):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linears) != 2:
        raise ValueError("Taylor-PR 仅支持单隐层。")
    W = linears[0].weight.detach().cpu().double().numpy()
    b = linears[0].bias.detach().cpu().double().numpy()
    v = linears[1].weight.detach().cpu().double().numpy().ravel()
    v0 = float(linears[1].bias.detach().cpu().double().numpy()[0])
    g = activation_derivs_at_zero(act_name, q)
    h1 = W.shape[0]
    beta0 = v0 + sum(v[j] * sum(g[k] / math.factorial(k) * b[j] ** k
                                for k in range(q + 1)) for j in range(h1))
    betas = {}
    for m in all_multi_indices(p, q):
        t = sum(m)
        denom = np.prod([math.factorial(mi) for mi in m])
        coef = 0.0
        for j in range(h1):
            wm = np.prod([W[j, i] ** m[i] for i in range(p) if m[i] > 0])
            s = sum(g[k] / math.factorial(k - t) * b[j] ** (k - t)
                    for k in range(t, q + 1))
            coef += v[j] * wm * s
        betas[m] = coef / denom
    return beta0, betas


def eval_taylor_polynomial(beta0, betas, X):
    X = np.asarray(X, np.float64)
    pred = np.full(X.shape[0], beta0)
    for m, beta in betas.items():
        term = np.ones(X.shape[0])
        for i, mi in enumerate(m):
            if mi > 0:
                term = term * X[:, i] ** mi
        pred += beta * term
    return pred.reshape(-1, 1)


def custom_features_full(X, max_order, include_special=True):
    base, _ = poly_interaction_features(X, max_order)
    parts = [base]
    if include_special:
        sp = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            sp += [np.sin(xi), np.exp(np.clip(xi, -30, 30))]
        parts.append(np.column_stack(sp))
    return np.column_stack(parts)


def fit_full_pr(X_tr, y_tr, X_te, max_order, include_special=True):
    F_tr = custom_features_full(X_tr, max_order, include_special)
    F_te = custom_features_full(X_te, max_order, include_special)
    lr = LinearRegression().fit(F_tr, y_tr)
    return lr.predict(F_te), lr


# ─────────────────────────────────────────────
# 单次实验（GPU）
# ─────────────────────────────────────────────
def run_experiment(case, activation, hidden_layers, device,
                   n=10000, p=20, q=3, scaling=(-1.0, 1.0),
                   epochs=300, batch_size=512, data_seed=0, init_seed=0,
                   include_special=True, ltpr_max_p=0, verbose=False):
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    hidden_layers = list(hidden_layers)
    n_hidden = len(hidden_layers)
    max_order = choose_fullpr_order(p, n_hidden, include_special)

    X_tr, X_te, y_tr, y_te = generate_dataset(case, n, p, data_seed, scaling)
    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    Xte_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    model = Net(p, hidden_layers, activation)
    model = train_nn_gpu(model, Xtr_t, ytr_t, device, epochs=epochs,
                         batch_size=batch_size, verbose=verbose)
    model.eval()
    with torch.no_grad():
        pred_nn = model(Xte_t).cpu().numpy()
    nn_mse = mean_squared_error(y_te, pred_nn)

    res = {"case": case, "activation": activation,
           "hidden": ",".join(map(str, hidden_layers)), "n_hidden": n_hidden,
           "q": q, "data_seed": data_seed, "init_seed": init_seed,
           "special": include_special, "n": n, "p": p,
           "max_order": max_order, "NN_mse_vs_y": nn_mse}

    # 每层激活 u → g(u)
    act_io = activation_io_distances(model, X_te, device)
    res["act_io_mean"] = (float(np.nanmean(list(act_io.values())))
                          if act_io else float("nan"))
    # 总值：所有激活层 u→g(u) 距离之和（随深度累积，反映整网注入的非线性总量）
    res["act_io_sum"] = (float(np.nansum(list(act_io.values())))
                         if act_io else float("nan"))

    # Taylor-PR（仅单隐层、光滑激活；高维下 q 受 max_order 约束）
    pred_tpr = None
    if n_hidden == 1 and activation != "relu":
        q_eff = min(q, SAFE_MAX_ORDER)
        beta0, betas = nn_to_taylor_polynomial(model, activation, q_eff, p)
        pred_tpr = eval_taylor_polynomial(beta0, betas, X_te)
        res["TPR_mse_vs_NN"] = mean_squared_error(pred_nn, pred_tpr)
        res["TPR_mse_vs_y"] = mean_squared_error(y_te, pred_tpr)
    else:
        res["TPR_mse_vs_NN"] = float("nan")
        res["TPR_mse_vs_y"] = float("nan")

    # LayerTaylor-PR（逐层泰勒展开，任意层数；多隐层情形的多项式对照）
    # 注意：逐层展开的项数随输入维度 p 与层数指数增长，高维下会组合爆炸。
    # 故仅在 p <= LTPR_MAX_P 时启用，否则跳过（高维下成本不可接受）。
    pred_ltpr = None
    if _HAS_TAYLOR_EXPAND and activation != "relu" and p <= ltpr_max_p:
        try:
            ltpr_poly = expand_nn_to_polynomial(
                model.cpu(), input_dim=p, order=min(q, SAFE_MAX_ORDER),
                expansion_point="data", X_ref=X_tr,
                max_total_degree=max(min(q, SAFE_MAX_ORDER), n_hidden * 2))
            model.to(device)   # 展开需在 CPU 上取权重，完后搬回 device
            pred_ltpr = ltpr_poly.predict(X_te)
            res["LTPR_mse_vs_NN"] = mean_squared_error(pred_nn, pred_ltpr)
            res["LTPR_mse_vs_y"] = mean_squared_error(y_te, pred_ltpr)
            res["LTPR_n_terms"] = ltpr_poly.n_terms
            res["shape_NN_LTPR"] = surface_shape_distance(X_te, pred_nn, pred_ltpr)
        except Exception:
            model.to(device)
            res["LTPR_mse_vs_NN"] = float("nan")
            res["LTPR_mse_vs_y"] = float("nan")
            res["LTPR_n_terms"] = float("nan")
            res["shape_NN_LTPR"] = float("nan")
    else:
        res["LTPR_mse_vs_NN"] = float("nan")
        res["LTPR_mse_vs_y"] = float("nan")
        res["LTPR_n_terms"] = float("nan")
        res["shape_NN_LTPR"] = float("nan")

    # Full-PR
    pred_fpr, fpr_model = fit_full_pr(X_tr, y_tr, X_te, max_order, include_special)
    res["FPR_mse_vs_y"] = mean_squared_error(y_te, pred_fpr)
    res["FPR_mse_vs_NN"] = mean_squared_error(pred_nn, pred_fpr)
    res["FPR_r2_vs_y"] = r2_score(y_te, pred_fpr)

    # 模型间距离（数据区内，抽样）
    res["shape_NN_FPR_in"] = surface_shape_distance(X_te, pred_nn, pred_fpr)
    res["mahal_NN_FPR"] = mahalanobis_mean(pred_nn, pred_fpr)

    # 外推区（2 倍范围，抽样 2000 点）
    lo, hi = scaling
    rng_e = np.random.default_rng(data_seed + 9999)
    X_ext = rng_e.uniform(2 * lo, 2 * hi, size=(2000, p))
    with torch.no_grad():
        pred_nn_ext = model(torch.tensor(X_ext, dtype=torch.float32,
                                         device=device)).cpu().numpy()
    pred_fpr_ext = fpr_model.predict(
        custom_features_full(X_ext, max_order, include_special))
    res["shape_NN_FPR_ext"] = surface_shape_distance(X_ext, pred_nn_ext, pred_fpr_ext)

    # 派生
    if nn_mse > 0 and res["FPR_mse_vs_y"] > 0:
        res["abs_rmse_gap"] = abs(math.sqrt(nn_mse) - math.sqrt(res["FPR_mse_vs_y"]))
        res["NN_FPR_log10"] = math.log10(nn_mse / res["FPR_mse_vs_y"])
    else:
        res["abs_rmse_gap"] = float("nan")
        res["NN_FPR_log10"] = float("nan")
    if nn_mse > 0:
        for short, mse_col in [
            ("FPR", "FPR_mse_vs_y"),
            ("TPR", "TPR_mse_vs_y"),
            ("LTPR", "LTPR_mse_vs_y"),
        ]:
            mv = res.get(mse_col)
            if mv is not None and not math.isnan(mv):
                res[f"{short}_mse_delta_vs_NN"] = mv - nn_mse
                res[f"{short}_improve_vs_NN_pct"] = 100.0 * (nn_mse - mv) / nn_mse
                res[f"{short}_better_than_NN"] = mv < nn_mse
            else:
                res[f"{short}_mse_delta_vs_NN"] = float("nan")
                res[f"{short}_improve_vs_NN_pct"] = float("nan")
                res[f"{short}_better_than_NN"] = float("nan")

    if verbose:
        print(f"  [{case} {activation} h={hidden_layers} sp={include_special}] "
              f"NN_mse={nn_mse:.4f} FPR_mse={res['FPR_mse_vs_y']:.4f} "
              f"act_io={res['act_io_mean']:.4f} shape_in={res['shape_NN_FPR_in']:.4f} "
              f"shape_ext={res['shape_NN_FPR_ext']:.4f}")
    return res


# ─────────────────────────────────────────────
# 实验网格 + 输出
# ─────────────────────────────────────────────
def run_grid(device, n=10000, p=20, epochs=300, batch_size=512,
             repeats=10, data_seed=0, out_prefix="gpu_results",
             resume=True, checkpoint_every=1, ltpr_max_p=0):
    import pandas as pd
    from scipy.stats import spearmanr

    depth_variants = [(64,), (128, 64), (256, 128, 64)]   # 高维下用更宽的网络
    base_configs = (
        [("highdim_poly4", a) for a in ("softplus", "tanh", "sigmoid")]
        + [("highdim_poly3", a) for a in ("softplus", "tanh", "sigmoid")]   # 3阶+3层=命中
        + [("highdim_nonlinear", a) for a in ("softplus", "tanh", "sigmoid")]
    )
    plan = [(c, a, h, sp) for (c, a) in base_configs
            for h in depth_variants for sp in (True, False)]
    total = len(plan) * repeats
    print(f"GPU 大规模模拟：n={n}, p={p}, device={device}")
    print(f"  {len(base_configs)} 配置 × {len(depth_variants)} 深度 × 2 特殊项 "
          f"× {repeats} 重复 = {total} 次")

    raw_path = f"{out_prefix}_raw.csv"
    corr_path = f"{out_prefix}_correlations.csv"
    rows, done = [], 0
    completed = set()
    if resume and os.path.exists(raw_path):
        old = pd.read_csv(raw_path)
        rows = old.to_dict("records")
        key_cols = ["case", "activation", "hidden", "special",
                    "data_seed", "init_seed", "n", "p"]
        if all(c in old.columns for c in key_cols):
            for rec in rows:
                completed.add(tuple(rec[c] for c in key_cols))
            done = len(completed)
            print(f"  续跑：已从 {raw_path} 载入 {done}/{total} 个完成 run")

    def checkpoint():
        if rows:
            pd.DataFrame(rows).to_csv(raw_path, index=False, encoding="utf-8-sig")

    for (case, act, hidden, special) in plan:
        for init_seed in range(repeats):
            key = (case, act, ",".join(map(str, hidden)), special,
                   data_seed, init_seed, n, p)
            if key in completed:
                continue
            r = run_experiment(case, act, hidden, device, n=n, p=p,
                               epochs=epochs, batch_size=batch_size,
                               data_seed=data_seed, init_seed=init_seed,
                               include_special=special, ltpr_max_p=ltpr_max_p,
                               verbose=False)
            r["well_specified"] = (case == "highdim_poly3" and len(hidden) == 3)
            rows.append(r)
            completed.add(key)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  进度 {done}/{total}")
            if checkpoint_every > 0 and (done % checkpoint_every == 0 or done == total):
                checkpoint()

    df = pd.DataFrame(rows)
    checkpoint()

    # 聚合表
    gk = ["case", "activation", "hidden", "special", "well_specified"]
    agg = ["NN_mse_vs_y", "FPR_mse_vs_y", "FPR_mse_vs_NN", "act_io_mean",
           "act_io_sum", "LTPR_mse_vs_NN", "shape_NN_LTPR",
           "shape_NN_FPR_in", "shape_NN_FPR_ext", "NN_FPR_log10",
           "FPR_improve_vs_NN_pct", "TPR_improve_vs_NN_pct",
           "LTPR_improve_vs_NN_pct"]
    g = df.groupby(gk)[agg].agg(["mean", "std"])
    print(f"\n{'='*168}")
    print(f"聚合汇总（每格 {repeats} 次重复 均值±标准差；★=正好命中；"
          f"LTPR=逐层泰勒展开，p>10 时跳过显示—）")
    print(f"{'='*168}")
    print(f"{'case':<20}{'act':<9}{'hid':<11}{'sp':<4}{'命中':<5}"
          f"{'NN_mse':>16}{'FPR_mse':>16}{'LTPR vs NN':>16}{'形NN-LTPR':>16}"
          f"{'act_io':>14}{'shape_in':>16}{'shape_ext':>16}")
    print('-'*168)
    prev = None
    for idx, sub in g.iterrows():
        case, act, hid, sp, ws = idx
        if prev is not None and case != prev:
            print('-'*168)
        prev = case
        def ms(c):
            mv = sub[(c, 'mean')]
            if math.isnan(mv):
                return f"{'—':>14}"
            return f"{mv:.4f}±{sub[(c,'std')]:.4f}"
        print(f"{case:<20}{act:<9}{hid:<11}{'Y' if sp else 'N':<4}"
              f"{'*' if ws else ' ':<5}{ms('NN_mse_vs_y'):>16}{ms('FPR_mse_vs_y'):>16}"
              f"{ms('LTPR_mse_vs_NN'):>16}{ms('shape_NN_LTPR'):>16}"
              f"{ms('act_io_mean'):>14}{ms('shape_NN_FPR_in'):>16}{ms('shape_NN_FPR_ext'):>16}")
    print('='*168)

    # 相关性表
    corr_pairs = [
        ("act_io_sum", "shape_NN_FPR_in", "act-change(sum) -> NN-PR closeness(in)"),
        ("act_io_sum", "shape_NN_FPR_ext", "act-change(sum) -> closeness(extrap)"),
        ("act_io_sum", "abs_rmse_gap", "act-change(sum) -> perf gap"),
        ("shape_NN_FPR_in", "abs_rmse_gap", "NN-FullPR closeness -> perf gap"),
        ("shape_NN_LTPR", "abs_rmse_gap", "NN-LayerTaylorPR closeness -> perf gap"),
        ("shape_NN_LTPR", "shape_NN_FPR_in", "NN-LayerTaylorPR vs NN-FullPR"),
    ]
    recs = []
    for gname, gdf in [("ALL", df)] + [(c, df[df.case == c]) for c in df.case.unique()]:
        for x, y, label in corr_pairs:
            m = gdf[[x, y]].dropna()
            if len(m) >= 5 and m[x].std() > 0 and m[y].std() > 0:
                rho, pv = spearmanr(m[x], m[y])
            else:
                rho, pv = float("nan"), float("nan")
            recs.append({"group": gname, "relation": label, "x": x, "y": y,
                         "spearman_rho": rho, "p_value": pv, "n": len(m)})
    corr_df = pd.DataFrame(recs)
    print(f"\n{'='*92}")
    print("相关性摘要（Spearman ρ）")
    print(f"{'='*92}")
    print(f"{'group':<20}{'relation':<40}{'rho':>9}{'p':>11}{'n':>6}")
    print('-'*92)
    prev = None
    for _, c in corr_df.iterrows():
        if prev is not None and c.group != prev:
            print('-'*92)
        prev = c.group
        sig = "*" if (not math.isnan(c.p_value) and c.p_value < 0.05) else " "
        rs = f"{c.spearman_rho:.3f}" if not math.isnan(c.spearman_rho) else "-"
        ps = f"{c.p_value:.3g}" if not math.isnan(c.p_value) else "-"
        print(f"{c.group:<20}{c.relation:<40}{rs:>8}{sig}{ps:>11}{c.n:>6}")
    print('='*92)

    # 图
    try:
        _plots(df, corr_pairs, out_prefix)
        plotted = True
    except Exception as e:
        print(f"绘图跳过：{e}")
        plotted = False

    # CSV
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
    print(f"\n已导出：{raw_path}（{len(df)} 行）、"
          f"{corr_path}"
          + (f"、{out_prefix}_*.png" if plotted else ""))
    return df, corr_df


def _plots(df, corr_pairs, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p_label = ",".join(str(x) for x in sorted(df["p"].unique()))
    acts = ["softplus", "tanh", "sigmoid"]
    colors = {"softplus": "#2a9d8f", "tanh": "#e76f51", "sigmoid": "#264653"}
    compare_models = [("FPR", "FullPR"), ("TPR", "Taylor-PR"), ("LTPR", "LayerTaylor-PR")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [(axes[0], "act_io_sum", "activation change u->g(u) (sum)"),
                           (axes[1], "shape_NN_FPR_in", "NN-FullPR shape (in)")]:
        data = [df[df.activation == a][col].dropna().values for a in acts]
        bp = ax.boxplot(data, labels=acts, patch_artist=True)
        for patch, a in zip(bp["boxes"], acts):
            patch.set_facecolor(colors[a]); patch.set_alpha(0.6)
        ax.set_title(title); ax.set_ylabel(col); ax.grid(alpha=0.3)
    fig.suptitle(f"High-dim (n=10000,p={p_label}) metrics by activation")
    fig.tight_layout(); fig.savefig(f"{out_prefix}_boxplot.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (x, y, label) in zip(axes.ravel(), corr_pairs):
        for a in acts:
            sub = df[df.activation == a][[x, y]].dropna()
            ax.scatter(sub[x], sub[y], s=16, alpha=0.5, color=colors[a], label=a)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Geometric distance vs closeness/performance (high-dim)")
    fig.tight_layout(); fig.savefig(f"{out_prefix}_scatter.png", dpi=130); plt.close(fig)

    rows = []
    for short, name in compare_models:
        pct_col = f"{short}_improve_vs_NN_pct"
        win_col = f"{short}_better_than_NN"
        if pct_col not in df.columns or win_col not in df.columns:
            continue
        for act in acts:
            sub = df[df.activation == act][[pct_col, win_col]].dropna()
            if len(sub) == 0:
                continue
            rows.append((name, act, float(sub[win_col].mean() * 100.0),
                         float(sub[pct_col].median())))
    if rows:
        models = list(dict.fromkeys(r[0] for r in rows))
        x = np.arange(len(models))
        width = 0.24
        offsets = np.linspace(-width, width, len(acts))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        for i, act in enumerate(acts):
            win_vals, imp_vals = [], []
            for model in models:
                hit = [r for r in rows if r[0] == model and r[1] == act]
                win_vals.append(hit[0][2] if hit else np.nan)
                imp_vals.append(hit[0][3] if hit else np.nan)
            axes[0].bar(x + offsets[i], win_vals, width=width, color=colors[act],
                        alpha=0.75, label=act)
            axes[1].bar(x + offsets[i], imp_vals, width=width, color=colors[act],
                        alpha=0.75, label=act)
        axes[0].axhline(50, color="k", lw=0.8, ls="--", alpha=0.5)
        axes[0].set_ylabel("win rate vs NN (%)")
        axes[0].set_title("How often model MSE < NN MSE", fontsize=10)
        axes[1].axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        axes[1].set_ylabel("median improvement vs NN (%)")
        axes[1].set_title("Positive means better than NN", fontsize=10)
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha="right")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("High-dim models vs NN baseline", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{out_prefix}_nn_baseline.png", dpi=130)
        plt.close(fig)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="高维 GPU 版 Morala 实验 (n=10000, p=10/20/50/100/200)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--p", type=int, default=None,
                    help="Run a single p value; overrides --p-values.")
    ap.add_argument("--p-values", type=int, nargs="*", default=DEFAULT_P_VALUES,
                    help="Input dimensions to run when --p is not set.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="gpu_results")
    ap.add_argument("--no-resume", action="store_true",
                    help="Do not resume from an existing <out-prefix>_raw.csv checkpoint.")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="Write <out-prefix>_raw.csv every N completed runs.")
    ap.add_argument("--ltpr-max-p", type=int, default=0,
                    help="Enable LayerTaylor-PR only when p <= this value. "
                         "Default 0 skips it for high-dimensional comparability.")
    ap.add_argument("--case", default=None,
                    choices=["highdim_poly3", "highdim_poly4", "highdim_nonlinear"])
    ap.add_argument("--activation", default="softplus", choices=list(ACTIVATIONS))
    ap.add_argument("--hidden", default="128,64")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"使用设备：{device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    p_values = [args.p] if args.p is not None else args.p_values
    if args.case is not None:
        # 单配置调试
        hidden = tuple(int(x) for x in args.hidden.split(","))
        for p in p_values:
            print(f"p={p}")
            r = run_experiment(args.case, args.activation, hidden, device,
                               n=args.n, p=p, epochs=args.epochs,
                               batch_size=args.batch_size,
                               ltpr_max_p=args.ltpr_max_p, verbose=True)
            for k, v in r.items():
                print(f"  {k}: {v}")
    else:
        for p in p_values:
            out_prefix = args.out_prefix if len(p_values) == 1 else f"{args.out_prefix}_p{p}"
            run_grid(device, n=args.n, p=p, epochs=args.epochs,
                     batch_size=args.batch_size, repeats=args.repeats,
                     data_seed=args.data_seed, out_prefix=out_prefix,
                     resume=not args.no_resume,
                     checkpoint_every=args.checkpoint_every,
                     ltpr_max_p=args.ltpr_max_p)
