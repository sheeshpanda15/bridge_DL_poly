"""
measure_morala.py
─────────────────────────────────────────────────────────────────
将 measure.py 的实验替换为 Morala et al. (2021, Neural Networks)
"Towards a mathematical framework to inform neural network
modelling via polynomial regression" 风格的模拟实验，并保留
原项目中的几何距离测量（Mahalanobis / Kendall shape distance）。

每个实验中同时构建三个模型并互相比较：
  (1) NN        : 单隐层（或多隐层）前馈神经网络（原始模型）
  (2) Taylor-PR : 按 Morala et al. 式(6)，从 NN 权重经泰勒展开
                  直接导出的多项式回归（仅单隐层时可用）
  (3) Full-PR   : 我原本的"全设定"多项式回归
                  （poly_interaction_features + 特殊特征 + 线性回归），
                  其最高阶 max_order = NN 隐层层数

输出内容：
  - 三个模型的 Test MSE（相对真实 y）
  - 模型两两之间预测的 MSE（论文的核心指标：PR 是否复现 NN）
  - 模型两两之间的 Mahalanobis 距离（预测向量）与
    Kendall 形状距离（响应曲面构形 [X, ŷ]）
  - NN 训练过程中逐 epoch 的 线性层 vs 激活层 几何距离
  - 最后一个采集 epoch 的逐样本成对马氏距离

激活函数接口：--activation {softplus, tanh, sigmoid, relu}
（relu 在 0 点不可导，Taylor-PR 会被自动跳过并提示）

用法示例：
  python measure_morala.py                          # 跑默认的全部实验组
  python measure_morala.py --case paper_poly2 \
      --activation softplus --h1 4 --q 3 --scaling -1,1
  python measure_morala.py --repeats 20             # 重复模拟看 MSE 分布
"""

import argparse
import math
from collections import defaultdict
from itertools import product, combinations_with_replacement

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.spatial import distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from geomstats.geometry.pre_shape import PreShapeSpace


# ─────────────────────────────────────────────
# 激活函数接口
# ─────────────────────────────────────────────
ACTIVATIONS = {
    "softplus": nn.Softplus,
    "tanh":     nn.Tanh,
    "sigmoid":  nn.Sigmoid,
    "relu":     nn.ReLU,     # 仅供 NN 使用；Taylor-PR 不支持（0 点不可导）
}

ACTIVATION_HOOK_TYPES = (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softplus, nn.Identity)


def activation_derivs_at_zero(act_name: str, q: int):
    """
    用自动微分精确计算 g^(n)(0)，n = 0..q。
    对应论文 2.2 节中各激活函数在 0 点的泰勒系数来源。
    """
    if act_name == "relu":
        raise ValueError("ReLU 在 0 点不可导，无法做泰勒展开（参见论文结论：需分段多项式近似）。")
    act = ACTIVATIONS[act_name]()
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    g = act(x)
    derivs = [g.item()]
    cur = g
    for _ in range(q):
        (cur,) = torch.autograd.grad(cur.sum(), x, create_graph=True)
        derivs.append(cur.item())
    return derivs  # derivs[n] = g^(n)(0)


# ─────────────────────────────────────────────
# Hook 系统：按 epoch 采集线性层/激活层输出（沿用原项目）
# ─────────────────────────────────────────────
linear_by_epoch     = defaultdict(lambda: defaultdict(list))
activation_by_epoch = defaultdict(lambda: defaultdict(list))
current_epoch = -1
_capture_this_step = False


def reset_capture():
    global linear_by_epoch, activation_by_epoch, current_epoch, _capture_this_step
    linear_by_epoch     = defaultdict(lambda: defaultdict(list))
    activation_by_epoch = defaultdict(lambda: defaultdict(list))
    current_epoch = -1
    _capture_this_step = False


def to_cpu_detached(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    elif isinstance(x, (list, tuple)):
        return type(x)(to_cpu_detached(t) for t in x)
    elif isinstance(x, dict):
        return {k: to_cpu_detached(v) for k, v in x.items()}
    return x


def make_linear_hook(name):
    def hook(module, inputs, output):
        if _capture_this_step:
            linear_by_epoch[current_epoch][name].append(to_cpu_detached(output))
    return hook


def make_activation_hook(name):
    def hook(module, inputs, output):
        if _capture_this_step:
            activation_by_epoch[current_epoch][name].append(to_cpu_detached(output))
    return hook


def register_hooks(model):
    hooks = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(make_linear_hook(name)))
            elif isinstance(m, ACTIVATION_HOOK_TYPES):
                hooks.append(m.register_forward_hook(make_activation_hook(name)))
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────
# 数据生成：Morala et al. 的模拟框架
# ─────────────────────────────────────────────
def poly_interaction_features(X, max_order=None):
    """（沿用原项目）生成所有总次数 1..max_order 的交互多项式特征"""
    n_samples, n_features = X.shape
    if max_order is None:
        max_order = 1
    features, powers = [], []
    for total_degree in range(1, max_order + 1):
        for exponents in product(range(total_degree + 1), repeat=n_features):
            if sum(exponents) == total_degree:
                features.append(np.prod(X ** exponents, axis=1))
                powers.append(exponents)
    return np.column_stack(features), powers


def make_paper_polynomial(n=200, p=3, degree=2, noise_sd=0.1, rng=0):
    """
    论文 3 节通用框架：
      Xi ~ N(μi, 1)，μi ~ U(-10, 10)；
      Y = 完整 degree 阶多项式（含交互项），β ~ U(-5, 5)；
      再加 N(0, 0.1) 噪声。
    返回未缩放的 (X, y, true_betas)。
    """
    g = np.random.default_rng(rng)
    mus = g.uniform(-10, 10, size=p)
    X = g.normal(loc=mus, scale=1.0, size=(n, p))
    design, powers = poly_interaction_features(X, max_order=degree)
    betas = g.uniform(-5, 5, size=design.shape[1])
    beta0 = g.uniform(-5, 5)
    y = beta0 + design @ betas + g.normal(0, noise_sd, size=n)
    return X, y, (beta0, betas, powers)


def make_smooth_nonlinear_lowdim(n=200, p=3, noise_sd=0.1, rng=0):
    """
    新增实验组：非多项式真值（原项目 smooth_nonlinear 的低维版），
    检验当真实函数不是多项式时，Taylor-PR 对 NN 的局部复现能力。
      y = sin(x1) + x2² + x2·x3 + x3 + ε
    """
    g = np.random.default_rng(rng)
    X = g.normal(0, 1, size=(n, p))
    y = (np.sin(X[:, 0]) + X[:, 1] ** 2 + X[:, 1] * X[:, 2]
         + X[:, 2] + g.normal(0, noise_sd, size=n))
    return X, y, None


def scale_minmax(train, test, interval=(-1.0, 1.0)):
    """按训练集逐列 min-max 缩放到给定区间（论文中对 X 和 y 都缩放）"""
    lo, hi = interval
    mn = train.min(axis=0)
    mx = train.max(axis=0)
    span = np.where(mx - mn == 0, 1.0, mx - mn)

    def _s(z):
        return lo + (hi - lo) * (z - mn) / span

    return _s(train), _s(test)


def generate_dataset(case, n=200, p=3, rng=0, scaling=(-1.0, 1.0)):
    if case == "paper_poly2":
        X, y, _ = make_paper_polynomial(n=n, p=p, degree=2, rng=rng)
    elif case == "paper_poly3":
        X, y, _ = make_paper_polynomial(n=n, p=p, degree=3, rng=rng)
    elif case == "smooth_nonlinear":
        X, y, _ = make_smooth_nonlinear_lowdim(n=n, p=p, rng=rng)
    else:
        raise ValueError(f"未知数据集：{case}")

    # 论文流程：先缩放（X 与 y 都缩放），再 75/25 划分
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y.reshape(-1, 1), test_size=0.25, random_state=42
    )
    X_tr, X_te = scale_minmax(X_tr, X_te, scaling)
    y_tr, y_te = scale_minmax(y_tr, y_te, scaling)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    return to_t(X_tr), to_t(X_te), to_t(y_tr), to_t(y_te)


# ─────────────────────────────────────────────
# 神经网络（可配置隐层与激活函数）
# ─────────────────────────────────────────────
class ConfigurableNet(nn.Module):
    """
    隐层宽度由 hidden_layers 列表给出，激活函数由 act_name 给出，
    输出层为线性（满足论文回归设定）。
    nn.Sequential 命名为 model.0, model.1, ... 与原 hook 系统兼容。
    """
    def __init__(self, input_size, hidden_layers, act_name):
        super().__init__()
        act_cls = ACTIVATIONS[act_name]
        layers, prev = [], input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_nn(model, X_train, y_train, X_eval=None, y_eval=None, epochs=1500,
             optimizer_name="rprop", lr=0.01, capture_every=150, verbose=True):
    """
    全批量训练（模仿论文用 neuralnet 的 Rprop 弹性反向传播）。
    每隔 capture_every 个 epoch：
      - 通过 hook 采集一次各层输出（供几何距离分析）
      - 同步记录 Train MSE（及给定 X_eval 时的 Test MSE）
    返回 history: {epoch: {"train_mse":…, "test_mse":…}}，
    其 epoch 与几何距离的采集 epoch 一一对应，方便对比。
    """
    global current_epoch, _capture_this_step
    if optimizer_name == "rprop":
        opt = optim.Rprop(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        opt = optim.Adam(model.parameters(), lr=1e-3)
    else:
        raise ValueError("optimizer 仅支持 rprop / adam")

    criterion = nn.MSELoss()
    history = {}
    model.train()
    for epoch in range(epochs):
        current_epoch = epoch
        is_capture = (epoch % capture_every == 0) or (epoch == epochs - 1)
        _capture_this_step = is_capture
        out = model(X_train)
        loss = criterion(out, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        _capture_this_step = False     # 评估 Test MSE 时不再触发 hook 采集

        if is_capture:
            rec = {"train_mse": loss.item()}
            if X_eval is not None:
                model.eval()
                with torch.no_grad():
                    rec["test_mse"] = criterion(model(X_eval), y_eval).item()
                model.train()
            history[epoch] = rec

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Train MSE: {loss.item():.6f}")
    return history


# ─────────────────────────────────────────────
# Morala et al. 式(6)：NN 权重 → 多项式系数
# ─────────────────────────────────────────────
def all_multi_indices(p, q):
    """枚举所有 1 ≤ Σmi ≤ q 的多重指数 (m1,...,mp)"""
    for t in range(1, q + 1):
        for combo in combinations_with_replacement(range(p), t):
            m = [0] * p
            for idx in combo:
                m[idx] += 1
            yield tuple(m)


def nn_to_taylor_polynomial(model, act_name, q):
    """
    实现论文式 (5)(6)：
      β0          = v0 + Σ_j v_j Σ_{n=0}^{q} g^(n)(0)/n! · w0j^n
      β_{l1...lt} = Σ_j v_j Σ_{n=t}^{q} g^(n)(0) /
                    ((n-t)!·m1!···mp!) · w0j^{n-t} · w1j^{m1}···wpj^{mp}
    其中 w0j 为隐层偏置，v0 为输出层偏置。仅适用于单隐层网络。
    返回 (beta0, {multi_index: beta})。
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linears) != 2:
        raise ValueError("Taylor-PR 仅支持单隐层网络（论文设定）。")

    W  = linears[0].weight.detach().double().numpy()          # (h1, p)
    b  = linears[0].bias.detach().double().numpy()            # (h1,)  = w0j
    v  = linears[1].weight.detach().double().numpy().ravel()  # (h1,)
    v0 = float(linears[1].bias.detach().double().numpy()[0])

    g = activation_derivs_at_zero(act_name, q)                # g[n] = g^(n)(0)
    h1, p = W.shape

    # 截距
    beta0 = v0 + sum(
        v[j] * sum(g[n] / math.factorial(n) * b[j] ** n for n in range(q + 1))
        for j in range(h1)
    )

    # 各阶系数
    betas = {}
    for m in all_multi_indices(p, q):
        t = sum(m)
        denom_m = np.prod([math.factorial(mi) for mi in m])
        coef = 0.0
        for j in range(h1):
            wm = np.prod([W[j, i] ** m[i] for i in range(p) if m[i] > 0])
            s = sum(g[n] / math.factorial(n - t) * b[j] ** (n - t)
                    for n in range(t, q + 1))
            coef += v[j] * wm * s
        betas[m] = coef / denom_m
    return beta0, betas


def eval_taylor_polynomial(beta0, betas, X):
    X = np.asarray(X, dtype=np.float64)
    pred = np.full(X.shape[0], beta0)
    for m, beta in betas.items():
        term = np.ones(X.shape[0])
        for i, mi in enumerate(m):
            if mi > 0:
                term = term * X[:, i] ** mi
        pred += beta * term
    return pred.reshape(-1, 1)


def _taylor_sanity_check(model, act_name, q, beta0, betas, X, tol=1e-6):
    """
    验证：多项式预测 == 直接用截断泰勒级数算激活的 NN 前向。
    二者在数值精度内应一致（这是式(6)推导的恒等式）。
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    W  = linears[0].weight.detach().double().numpy()
    b  = linears[0].bias.detach().double().numpy()
    v  = linears[1].weight.detach().double().numpy().ravel()
    v0 = float(linears[1].bias.detach().double().numpy()[0])
    g  = activation_derivs_at_zero(act_name, q)

    X = np.asarray(X, dtype=np.float64)
    U = X @ W.T + b                                   # 突触电位 u_j
    Yh = sum(g[n] / math.factorial(n) * U ** n for n in range(q + 1))
    z_direct = v0 + Yh @ v
    z_poly = eval_taylor_polynomial(beta0, betas, X).ravel()
    err = np.max(np.abs(z_direct - z_poly))
    assert err < tol, f"Taylor-PR 自检失败，max|diff|={err:.3e}"
    return err


def synaptic_potential_coverage(model, X, act_name, q, err_threshold=0.1):
    """
    论文图 5(C) 的数值版诊断：统计突触电位 u_j 落在
    泰勒近似误差 < err_threshold 区间内的比例。
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    W = linears[0].weight.detach().double().numpy()
    b = linears[0].bias.detach().double().numpy()
    U = np.asarray(X, dtype=np.float64) @ W.T + b     # (n, h1)

    act = ACTIVATIONS[act_name]()
    g = activation_derivs_at_zero(act_name, q)
    grid = np.linspace(-6, 6, 2001)
    with torch.no_grad():
        true_vals = act(torch.tensor(grid)).numpy()
    taylor_vals = sum(g[n] / math.factorial(n) * grid ** n for n in range(q + 1))
    ok = np.abs(true_vals - taylor_vals) < err_threshold
    # 找包含 0 的最大连续可接受区间
    zero_idx = np.searchsorted(grid, 0.0)
    left = zero_idx
    while left > 0 and ok[left - 1]:
        left -= 1
    right = zero_idx
    while right < len(grid) - 1 and ok[right + 1]:
        right += 1
    lo, hi = grid[left], grid[right]
    inside = float(((U >= lo) & (U <= hi)).mean())
    return inside, (lo, hi)


# ─────────────────────────────────────────────
# 我原本的"全设定"多项式模型（Full-PR）
# ─────────────────────────────────────────────
def custom_features_full(X, max_order=None, include_special=True, full_poly=False):
    """
    全设定多项式特征（按用户规格）：
      - 所有总次数 1..max_order 的完整交互多项式项
      - 特殊项：每个变量的 sin(x_i) 与 e^{x_i}
    max_order 由调用方传入（实验中 = NN 隐层层数）。
    """
    base_features, _ = poly_interaction_features(X, max_order)
    all_features = [base_features]

    if include_special:
        special = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            special.extend([
                np.sin(xi),
                np.exp(np.clip(xi, -30, 30)),
            ])
        all_features.append(np.column_stack(special))

    combined = np.column_stack(all_features)
    if full_poly:
        expanded, _ = poly_interaction_features(combined, max_order)
        return expanded
    return combined


def fit_full_pr(X_train, y_train, X_test, max_order, include_special=True):
    F_tr = custom_features_full(X_train, max_order=max_order,
                                include_special=include_special)
    F_te = custom_features_full(X_test, max_order=max_order,
                                include_special=include_special)
    lr = LinearRegression().fit(F_tr, y_train)
    return lr.predict(F_te), lr


# ─────────────────────────────────────────────
# 几何距离函数（沿用原项目）
# ─────────────────────────────────────────────
def pooled_covariance(A, B, shrinkage=None, ridge=0.0):
    X = np.vstack([A, B])
    if shrinkage is None:
        cov = np.cov(X, rowvar=False)
    elif shrinkage == 'ledoitwolf':
        cov = LedoitWolf().fit(X).covariance_
    else:
        raise ValueError("Unsupported shrinkage")
    cov = np.atleast_2d(cov)
    if ridge > 0:
        cov = cov + ridge * np.eye(cov.shape[0])
    VI = np.linalg.pinv(cov)
    return cov, VI


def mahalanobis(A, B, shrinkage='ledoitwolf', ridge=0.0):
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    muA, muB = A.mean(axis=0), B.mean(axis=0)
    _, VI = pooled_covariance(A, B, shrinkage=shrinkage, ridge=ridge)
    return distance.mahalanobis(muA, muB, VI)


def kendall_shape_distance(A, B, n_landmarks=16, atol=1e-12):
    """PCA 压成地标配置后计算 Kendall 形状距离（沿用原项目）"""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n_comp = min(n_landmarks, A.shape[1], B.shape[1], A.shape[0] - 1)
    if n_comp < 2:
        return float("nan")   # 一维构形不构成形状空间

    A_proj = PCA(n_components=n_comp).fit_transform(A)
    B_proj = PCA(n_components=n_comp).fit_transform(B)
    landmark_A = A_proj[:n_comp, :]
    landmark_B = B_proj[:n_comp, :]

    k, m = landmark_A.shape
    ps = PreShapeSpace(k_landmarks=k, ambient_dim=m)
    ZA = ps.projection(landmark_A)
    ZB = ps.projection(landmark_B)
    if np.allclose(ZA, ZB, atol=atol):
        return 0.0
    return max(float(ps.metric.dist(ZA, ZB)), 0.0)


def surface_shape_distance(X, predA, predB, n_landmarks=16):
    """
    模型间的响应曲面形状距离：
    把两模型的拟合图象 {(x_i, ŷ_i)} 视为 (n, p+1) 构形，
    再用 Kendall 形状距离比较。
    """
    GA = np.hstack([X, predA.reshape(-1, 1)])
    GB = np.hstack([X, predB.reshape(-1, 1)])
    return kendall_shape_distance(GA, GB, n_landmarks=n_landmarks)


def next_name(name):
    k = int(name.split(".")[1])
    return f"model.{k + 1}"


def _pooled_cov_inv(X, Y, shrinkage='ledoitwolf', ridge=0.0):
    Z = np.vstack([X, Y])
    if shrinkage is None:
        cov = np.cov(Z, rowvar=False)
    elif shrinkage == 'ledoitwolf':
        cov = LedoitWolf().fit(Z).covariance_
    else:
        raise ValueError("shrinkage must be None or 'ledoitwolf'")
    cov = np.atleast_2d(cov)
    if ridge > 0:
        cov = cov + ridge * np.eye(cov.shape[0])
    return np.linalg.pinv(cov)


def _paired_mahalanobis(A, B, VI):
    D = A - B
    DV = D @ VI
    d2 = np.maximum(np.sum(DV * D, axis=1), 0.0)
    return np.sqrt(d2)


def paired_md_from_first(linear_concat, activation_concat,
                         source_name="model.0", align="auto",
                         k=None, shrinkage="ledoitwolf", ridge=0.0):
    assert source_name in linear_concat, f"{source_name} 不在 linear_concat 中"
    src = linear_concat[source_name]
    N, d_src = src.shape

    all_layers = {**linear_concat, **activation_concat}
    items = [(n, X) for n, X in all_layers.items() if n != source_name]
    items.sort(key=lambda pair: int(pair[0].split(".")[1]))

    results, summary = {}, {}
    for name, tgt in items:
        if tgt.shape[0] != N:
            raise ValueError(f"{name} 样本数不一致：{tgt.shape[0]} vs {N}")
        d_tgt = tgt.shape[1]
        mode = align
        if mode == "auto":
            mode = "match" if d_src == d_tgt else "cca"
        if mode == "match":
            if d_src != d_tgt:
                continue
            A_aligned, B_aligned = src, tgt
        else:
            k_eff = k or min(d_src, d_tgt, max(N - 1, 1))
            cca = CCA(n_components=k_eff, max_iter=5000)
            A_aligned, B_aligned = cca.fit_transform(src, tgt)

        VI = _pooled_cov_inv(A_aligned, B_aligned, shrinkage=shrinkage, ridge=ridge)
        dvec = _paired_mahalanobis(A_aligned, B_aligned, VI)
        results[name] = dvec
        summary[name] = (float(dvec.mean()), float(dvec.std()))
    return results, summary


def _fmt(x, w=12, prec=6):
    """统一的数值格式化：nan → '—'，否则定宽小数"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—".center(w)
    return f"{x:>{w}.{prec}f}"


def epoch_layer_distances(X_input=None, verbose=True):
    """
    新指标：NN 中每一层输出与其上一层输出之间的 Kendall 形状距离，
    并对每个 epoch 求所有相邻层对的均值（adjacent-layer shape mean）。

    - X_input 不为 None 时，把网络输入作为第 0 层，参与 input → model.0 的比较；
    - 同维的相邻层对（线性层 → 其激活层）同时给出 Mahalanobis 距离；
      维度不同的层对（激活层 → 下一线性层）Mahalanobis 不可比，记为 —；
    - 一维输出层无法构成形状空间，shape 记为 —，不计入均值。
    """
    linear_epoch_concat, activation_epoch_concat = {}, {}
    for ep, layer_dict in linear_by_epoch.items():
        linear_epoch_concat[ep] = {
            n: torch.cat(outs, dim=0).numpy() for n, outs in layer_dict.items()
        }
    for ep, layer_dict in activation_by_epoch.items():
        activation_epoch_concat[ep] = {
            n: torch.cat(outs, dim=0).numpy() for n, outs in layer_dict.items()
        }

    epoch_rows, epoch_shape_mean, epoch_io_shape = {}, {}, {}
    epoch_act_shape_mean = {}
    for ep in sorted(linear_epoch_concat.keys()):
        all_layers = {**linear_epoch_concat[ep],
                      **activation_epoch_concat.get(ep, {})}
        ordered = sorted(all_layers.items(),
                         key=lambda kv: int(kv[0].split(".")[1]))
        seq = ([("input", np.asarray(X_input))] if X_input is not None else []) \
              + ordered

        rows, shape_vals, act_shape_vals = [], [], []
        act_names = set(activation_epoch_concat.get(ep, {}).keys())
        for (n_prev, A), (n_cur, B) in zip(seq[:-1], seq[1:]):
            sd = kendall_shape_distance(A, B)
            md = (mahalanobis(A, B, shrinkage='ledoitwolf', ridge=1e-6)
                  if A.shape[1] == B.shape[1] else float("nan"))
            rows.append((f"{n_prev} → {n_cur}", md, sd))
            if not math.isnan(sd):
                shape_vals.append(sd)
                # 仅"线性层 → 其激活层"的对：逐元素激活在线性区内
                # 近似为相似变换（统一缩放+平移），shape 距离 → 0；
                # 因此这个子均值是对"该层注入了多少非线性"的直接计量。
                if n_cur in act_names:
                    act_shape_vals.append(sd)

        epoch_rows[ep] = rows
        epoch_shape_mean[ep] = (float(np.mean(shape_vals))
                                if shape_vals else float("nan"))
        epoch_act_shape_mean[ep] = (float(np.mean(act_shape_vals))
                                    if act_shape_vals else float("nan"))

        # 输入 ↔ 输出 的 shape 距离：
        # 网络输出是一维标量，本身构不成形状空间，因此沿用模型间比较的做法，
        # 用响应曲面构形 [X | z] 代表"输出端"几何，与输入构形 X 比较。
        # 它衡量整张网络对输入几何的总扭曲量（相邻层距离衡量的是逐层扭曲）。
        if X_input is not None and ordered:
            z_out = ordered[-1][1]            # 最后一层（线性输出，n×1）
            Xin = np.asarray(X_input)
            epoch_io_shape[ep] = kendall_shape_distance(
                Xin, np.hstack([Xin, z_out]))
        else:
            epoch_io_shape[ep] = float("nan")

    if verbose:
        print(f"\n  ┌{'─' * 66}")
        print(f"  │ NN 内部：逐 Epoch 相邻层几何距离")
        print(f"  ├{'─' * 66}")
        print(f"  │ {'Epoch':>6}  {'相邻层对':<22}{'Mahalanobis':>13}"
              f"{'Kendall shape':>15}")
        for ep in sorted(epoch_rows.keys()):
            for i, (pair, md, sd) in enumerate(epoch_rows[ep]):
                ep_str = f"{ep:>6}" if i == 0 else " " * 6
                print(f"  │ {ep_str}  {pair:<22}{_fmt(md, 13)}{_fmt(sd, 15)}")
            print(f"  │ {' ' * 6}  {'★ 相邻层 shape 均值':<22}{' ' * 13}"
                  f"{_fmt(epoch_shape_mean[ep], 15)}")
            print(f"  │ {' ' * 6}  {'★ 激活层对均值(非线性)':<22}{' ' * 13}"
                  f"{_fmt(epoch_act_shape_mean[ep], 15)}")
            print(f"  │ {' ' * 6}  {'★ 输入 ↔ 输出 shape':<22}{' ' * 13}"
                  f"{_fmt(epoch_io_shape[ep], 15)}")
        print(f"  └{'─' * 66}")

    return (epoch_rows, epoch_shape_mean, epoch_act_shape_mean, epoch_io_shape,
            linear_epoch_concat, activation_epoch_concat)


# ─────────────────────────────────────────────
# 单次实验
# ─────────────────────────────────────────────
def run_experiment(case="paper_poly2", activation="softplus",
                   hidden_layers=(4,), q=3, scaling=(-1.0, 1.0),
                   n=200, p=3, epochs=1500, optimizer="rprop",
                   capture_every=150, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    reset_capture()

    hidden_layers = list(hidden_layers)
    n_hidden = len(hidden_layers)
    max_order_full_pr = n_hidden        # 要求：Full-PR 最高阶 = NN 隐层层数

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"实验：case={case}  activation={activation}  "
              f"hidden={hidden_layers}  q={q}  scaling={list(scaling)}  seed={seed}")
        print(f"{'=' * 72}")

    # 1. 数据
    X_tr, X_te, y_tr, y_te = generate_dataset(case, n=n, p=p, rng=seed,
                                              scaling=scaling)
    X_te_np, y_te_np = X_te.numpy(), y_te.numpy()

    # 2. 训练 NN（带 hook 采集；history 记录采集 epoch 上的 MSE）
    model = ConfigurableNet(p, hidden_layers, activation)
    hooks = register_hooks(model)
    history = train_nn(model, X_tr, y_tr, X_eval=X_te, y_eval=y_te,
                       epochs=epochs, optimizer_name=optimizer,
                       capture_every=capture_every, verbose=verbose)
    model.eval()
    with torch.no_grad():
        pred_nn = model(X_te).numpy()
    nn_mse = mean_squared_error(y_te_np, pred_nn)
    remove_hooks(hooks)

    res = {"case": case, "activation": activation, "h": tuple(hidden_layers),
           "q": q, "seed": seed, "nn_mse": nn_mse}

    # 3. Taylor-PR（Morala et al. 式(6)，仅单隐层 + 光滑激活）
    pred_tpr = None
    if n_hidden == 1 and activation != "relu":
        beta0, betas = nn_to_taylor_polynomial(model, activation, q)
        chk = _taylor_sanity_check(model, activation, q, beta0, betas, X_te_np)
        pred_tpr = eval_taylor_polynomial(beta0, betas, X_te_np)
        res["taylor_check_err"] = chk
        res["tpr_mse_vs_y"]  = mean_squared_error(y_te_np, pred_tpr)
        res["tpr_mse_vs_nn"] = mean_squared_error(pred_nn, pred_tpr)  # 论文核心指标
        cov_ratio, (lo, hi) = synaptic_potential_coverage(
            model, X_te_np, activation, q)
        res["u_coverage"] = cov_ratio
        if verbose:
            print(f"\n  [Taylor-PR] 多项式项数: {len(betas)}（恒等自检 "
                  f"max|diff|={chk:.2e}）")
            print(f"  [Taylor-PR] 突触电位落在可接受近似区间 "
                  f"[{lo:.2f}, {hi:.2f}] 内的比例: {cov_ratio:.1%}")
    else:
        reason = ("ReLU 在 0 点不可导" if activation == "relu"
                  else f"网络有 {n_hidden} 个隐层（式(6)仅适用单隐层）")
        if verbose:
            print(f"\n  [Taylor-PR] 跳过：{reason}")

    # 4. Full-PR（全设定多项式模型：完整交互项 + sin/e^x，max_order = 隐层数）
    pred_fpr, _ = fit_full_pr(X_tr.numpy(), y_tr.numpy(), X_te_np,
                              max_order=max_order_full_pr)
    res["fpr_mse_vs_y"]  = mean_squared_error(y_te_np, pred_fpr)
    res["fpr_mse_vs_nn"] = mean_squared_error(pred_nn, pred_fpr)

    # 4b. 多项式蒸馏：用与 Full-PR 完全相同的特征族去拟合 NN 自身的预测
    #     （训练集上拟合，测试集上量残差）。
    #     残差直接衡量"NN 学到的函数离这个多项式族有多远"——
    #     不含 y、不含噪声、不依赖 Full-PR 对 y 拟合的好坏，
    #     且对多隐层 / ReLU 也有定义（不像 Taylor-PR）。
    with torch.no_grad():
        pred_nn_tr = model(X_tr).numpy()
    F_tr_d = custom_features_full(X_tr.numpy(), max_order=max_order_full_pr)
    F_te_d = custom_features_full(X_te_np,      max_order=max_order_full_pr)
    distill = LinearRegression().fit(F_tr_d, pred_nn_tr)
    res["distill_resid"] = mean_squared_error(pred_nn, distill.predict(F_te_d))

    # 5. 模型对比输出（论文指标，表格化）
    if verbose:
        print(f"\n  ┌{'─' * 66}")
        print(f"  │ 模型对比（缩放空间内）")
        print(f"  ├{'─' * 66}")
        print(f"  │ {'模型':<12}{'MSE vs y':>12}{'MSE vs NN':>14}{'R² vs y':>10}")
        print(f"  │ {'NN':<12}{nn_mse:>12.6f}{'—':>14}"
              f"{r2_score(y_te_np, pred_nn):>10.4f}")
        if pred_tpr is not None:
            print(f"  │ {'Taylor-PR':<12}{res['tpr_mse_vs_y']:>12.6f}"
                  f"{res['tpr_mse_vs_nn']:>14.3e}"
                  f"{r2_score(y_te_np, pred_tpr):>10.4f}   ← MSE vs NN 为论文核心指标")
        print(f"  │ {'Full-PR':<12}{res['fpr_mse_vs_y']:>12.6f}"
              f"{res['fpr_mse_vs_nn']:>14.3e}"
              f"{r2_score(y_te_np, pred_fpr):>10.4f}   (max_order={max_order_full_pr})")
        print(f"  ├{'─' * 66}")
        print(f"  │ 多项式蒸馏残差（同族特征拟合 NN 预测，test）: "
              f"{res['distill_resid']:.3e}")
        print(f"  └{'─' * 66}")

    # 6. 模型两两之间的几何距离（按原项目方法）
    preds = {"NN": pred_nn}
    if pred_tpr is not None:
        preds["Taylor-PR"] = pred_tpr
    preds["Full-PR"] = pred_fpr

    names = list(preds.keys())
    if verbose:
        print(f"\n  ┌{'─' * 66}")
        print(f"  │ 模型间几何距离（预测向量 Mahalanobis / 响应曲面 Kendall shape）")
        print(f"  ├{'─' * 66}")
        print(f"  │ {'模型对':<26}{'Mahalanobis':>13}{'Kendall shape':>15}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            md = mahalanobis(preds[a], preds[b],
                             shrinkage='ledoitwolf', ridge=1e-6)
            sd = surface_shape_distance(X_te_np, preds[a], preds[b])
            res[f"mahal[{a}|{b}]"] = md
            res[f"shape[{a}|{b}]"] = sd
            if verbose:
                print(f"  │ {a + ' ↔ ' + b:<26}{_fmt(md, 13)}{_fmt(sd, 15)}")
    if verbose:
        print(f"  └{'─' * 66}")

    # 7. NN 内部：相邻层 shape 距离 + 输入↔输出 shape 距离（新指标）
    _, shape_means, act_shape_means, io_shapes, lin_cc, act_cc = epoch_layer_distances(
        X_input=X_tr.numpy(), verbose=verbose)
    if shape_means:
        last_ep = max(shape_means.keys())
        res["adj_shape_mean_last"] = shape_means[last_ep]
        res["adj_shape_mean_avg"]  = float(np.nanmean(list(shape_means.values())))
        res["act_shape_mean_last"] = act_shape_means[last_ep]
        res["act_shape_mean_avg"]  = float(np.nanmean(list(act_shape_means.values())))
        res["io_shape_last"]       = io_shapes[last_ep]
        res["io_shape_avg"]        = float(np.nanmean(list(io_shapes.values())))

    # 7b. 训练轨迹：MSE 与形状距离并排（方便对比性能与几何演化）
    if verbose and history and shape_means:
        print(f"\n  ┌{'─' * 72}")
        print(f"  │ 训练轨迹：MSE 与形状距离对照")
        print(f"  ├{'─' * 72}")
        print(f"  │ {'Epoch':>6}{'Train MSE':>13}{'Test MSE':>13}"
              f"{'邻层均值':>11}{'激活对(非线性)':>15}{'输入↔输出':>12}")
        for ep in sorted(shape_means.keys()):
            rec = history.get(ep, {})
            tr = rec.get("train_mse", float("nan"))
            te = rec.get("test_mse",  float("nan"))
            print(f"  │ {ep:>6}{_fmt(tr, 13)}{_fmt(te, 13)}"
                  f"{_fmt(shape_means[ep], 11, 4)}"
                  f"{_fmt(act_shape_means[ep], 15, 4)}{_fmt(io_shapes[ep], 12, 4)}")
        print(f"  └{'─' * 72}")

    # 7c. 逐样本成对马氏距离（原项目流程）
    if lin_cc:
        last_ep = max(lin_cc.keys())
        _, stats = paired_md_from_first(
            lin_cc[last_ep], act_cc.get(last_ep, {}),
            source_name="model.0", align="auto",
            shrinkage="ledoitwolf", ridge=1e-6,
        )
        if verbose:
            print(f"\n  ┌{'─' * 66}")
            print(f"  │ 逐样本成对马氏距离（Epoch {last_ep}，以 model.0 为参照）")
            print(f"  ├{'─' * 66}")
            print(f"  │ {'目标层':<14}{'mean':>10}{'std':>10}")
            for layer, (mu, sd) in stats.items():
                print(f"  │ {layer:<14}{mu:>10.4f}{sd:>10.4f}")
            print(f"  └{'─' * 66}")

    return res


# ─────────────────────────────────────────────
# 重复模拟：复现论文 3.2 节的 MSE 分布研究（缩小版）
# ─────────────────────────────────────────────
def repeat_study(case, activation, h1, q, scaling, repeats, epochs):
    print(f"\n{'#' * 72}")
    print(f"重复模拟（论文 3.2 节风格）：{case}, {activation}, "
          f"h1={h1}, q={q}, repeats={repeats}")
    print(f"{'#' * 72}")
    vals_nn, vals_fpr = [], []
    for r in range(repeats):
        res = run_experiment(case=case, activation=activation,
                             hidden_layers=(h1,), q=q, scaling=scaling,
                             epochs=epochs, seed=r, verbose=False)
        vals_nn.append(res.get("tpr_mse_vs_nn", np.nan))
        vals_fpr.append(res.get("fpr_mse_vs_nn", np.nan))
        print(f"  rep {r:>2d}: Taylor-PR vs NN MSE = {vals_nn[-1]:.3e} | "
              f"Full-PR vs NN MSE = {vals_fpr[-1]:.3e}")
    vals_nn = np.array(vals_nn)
    print(f"\n  Taylor-PR vs NN 的 MSE 分布：median={np.nanmedian(vals_nn):.3e}, "
          f"q25={np.nanpercentile(vals_nn, 25):.3e}, "
          f"q75={np.nanpercentile(vals_nn, 75):.3e}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def default_grid(epochs):
    """
    默认实验组（每个配置都跑 1 / 2 / 3 个隐层三种深度）：
      A 组（论文复现）  ：二阶多项式数据 × {softplus, tanh, sigmoid}，q=3
      B 组（新增）      ：三阶多项式数据，softplus，q ∈ {3, 5}
      C 组（新增）      ：非多项式光滑数据，softplus
    深度变体：(4,) / (8,4) / (16,8,4)。
    注意：Taylor-PR 仅在单隐层时构建（式(6)限制）；
          Full-PR 的 max_order = 隐层层数，随深度同步升阶。
    """
    depth_variants = [(4,), (8, 4), (16, 8, 4)]
    base_configs = (
        [("paper_poly2", act, 3) for act in ("softplus", "tanh", "sigmoid")]
        + [("paper_poly3", "softplus", q) for q in (3, 5)]
        + [("smooth_nonlinear", "softplus", 3)]
    )

    results = []
    for case, act, q in base_configs:
        for hidden in depth_variants:
            results.append(run_experiment(case=case, activation=act,
                                          hidden_layers=hidden, q=q,
                                          epochs=epochs))

    # 汇总表
    h_str = lambda h: ",".join(str(x) for x in h)
    print(f"\n{'═' * 126}")
    print("汇总（缩放空间内的 MSE；形状指标取自最后采集 epoch；"
          "Taylor-PR 仅单隐层可用；Full-PR max_order = 隐层层数）")
    print(f"{'═' * 126}")
    header = (f"{'case':<17}{'act':<10}{'q':<3}{'hidden':<9}{'NN vs y':>10}"
              f"{'TPR vs NN':>12}{'TPR vs y':>10}{'FPR vs NN':>12}{'FPR vs y':>10}"
              f"{'邻层均值':>11}{'激活对(非线性)':>15}"
              f"{'NN/FPR(lg)':>12}{'蒸馏NN→PR':>13}{'输入↔输出':>12}")
    print(header)
    print('─' * 126)
    prev_case = None
    for r in results:
        if prev_case is not None and r['case'] != prev_case:
            print('─' * 126)
        prev_case = r['case']
        tpr_nn = r.get('tpr_mse_vs_nn', float('nan'))
        tpr_y  = r.get('tpr_mse_vs_y',  float('nan'))
        tpr_nn_s = f"{tpr_nn:>12.3e}" if not math.isnan(tpr_nn) else f"{'—':>12}"
        tpr_y_s  = f"{tpr_y:>10.4f}"  if not math.isnan(tpr_y)  else f"{'—':>10}"
        # NN/FPR(lg) = log10(NN_mse / FPR_mse)：负数 = NN 更好，正数 = Full-PR 反超
        fpr_y = r['fpr_mse_vs_y']
        if r['nn_mse'] > 0 and fpr_y > 0:
            ratio_s = f"{math.log10(r['nn_mse'] / fpr_y):>12.3f}"
        else:
            ratio_s = f"{'—':>12}"
        print(f"{r['case']:<17}{r['activation']:<10}{r['q']:<3}"
              f"{h_str(r['h']):<9}"
              f"{r['nn_mse']:>10.4f}"
              f"{tpr_nn_s}{tpr_y_s}"
              f"{r['fpr_mse_vs_nn']:>12.3e}"
              f"{r['fpr_mse_vs_y']:>10.4f}"
              f"{r.get('adj_shape_mean_last', float('nan')):>11.4f}"
              f"{r.get('act_shape_mean_last', float('nan')):>15.4f}"
              f"{ratio_s}"
              f"{r.get('distill_resid', float('nan')):>13.3e}"
              f"{r.get('io_shape_last', float('nan')):>12.4f}")
    print('═' * 126)
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Morala et al. 风格实验 + 几何距离测量")
    parser.add_argument("--case", type=str, default=None,
                        choices=["paper_poly2", "paper_poly3", "smooth_nonlinear"],
                        help="数据集；不指定则跑默认全部实验组")
    parser.add_argument("--activation", type=str, default="softplus",
                        choices=list(ACTIVATIONS.keys()),
                        help="激活函数接口")
    parser.add_argument("--h1", type=str, default="4",
                        help="隐层宽度，逗号分隔可指定多隐层，如 '4' 或 '8,4'。"
                             "Full-PR 的 max_order = 隐层层数；"
                             "Taylor-PR 仅在单隐层时构建")
    parser.add_argument("--q", type=int, default=3, help="泰勒展开截断阶数")
    parser.add_argument("--scaling", type=str, default="-1,1",
                        help="min-max 缩放区间，'-1,1' 或 '0,1'")
    parser.add_argument("--n", type=int, default=200, help="样本量（论文用 200）")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--optimizer", type=str, default="rprop",
                        choices=["rprop", "adam"])
    parser.add_argument("--capture-every", type=int, default=150,
                        help="每隔多少 epoch 采集一次层输出用于几何距离")
    parser.add_argument("--repeats", type=int, default=0,
                        help=">0 时做论文 3.2 节风格的重复模拟")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scaling = tuple(float(s) for s in args.scaling.split(","))
    hidden = tuple(int(s) for s in args.h1.split(","))

    if args.repeats > 0:
        repeat_study(case=args.case or "paper_poly2",
                     activation=args.activation, h1=hidden[0], q=args.q,
                     scaling=scaling, repeats=args.repeats, epochs=args.epochs)
    elif args.case is None:
        default_grid(epochs=args.epochs)
    else:
        run_experiment(case=args.case, activation=args.activation,
                       hidden_layers=hidden, q=args.q, scaling=scaling,
                       n=args.n, epochs=args.epochs, optimizer=args.optimizer,
                       capture_every=args.capture_every, seed=args.seed)
