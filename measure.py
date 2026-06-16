import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from itertools import product
from collections import defaultdict
from scipy.spatial import distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from geomstats.geometry.pre_shape import PreShapeSpace
from sklearn.datasets import make_classification
import re

# ─────────────────────────────────────────────
# Hook 系统：按 epoch 采集线性层/激活层输出
# ─────────────────────────────────────────────
linear_by_epoch     = defaultdict(lambda: defaultdict(list))
activation_by_epoch = defaultdict(lambda: defaultdict(list))
current_epoch = -1

CAPTURE_MODE = "first_batch_per_epoch"
_capture_this_step = True

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
        global _capture_this_step, current_epoch
        if _capture_this_step:
            linear_by_epoch[current_epoch][name].append(to_cpu_detached(output))
    return hook

def make_activation_hook(name):
    def hook(module, inputs, output):
        global _capture_this_step, current_epoch
        if _capture_this_step:
            activation_by_epoch[current_epoch][name].append(to_cpu_detached(output))
    return hook

def register_hooks(model):
    hooks = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(make_linear_hook(name)))
            elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Identity)):
                hooks.append(m.register_forward_hook(make_activation_hook(name)))
    return hooks

def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────
# 数据集生成
# ─────────────────────────────────────────────
d = 30

def poly_data(n=1000, d=30):
    mean = np.zeros(d)
    cov = np.eye(d)
    data_x = np.random.multivariate_normal(mean, cov, size=n)
    beta = np.ones(d)
    data_y = data_x @ beta + np.random.normal(0, 0.1, size=n)
    X_tensor = torch.tensor(data_x, dtype=torch.float32)
    y_tensor = torch.tensor(data_y, dtype=torch.float32).unsqueeze(1)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def make_classified(n=1000, d=30, rng=0):
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=d-5, n_redundant=0,
        n_clusters_per_class=1, class_sep=1.5, random_state=42
    )
    X = 8 * (X - X.min()) / (X.max() - X.min()) - 4
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def make_highdim_step(n=1000, d=30, rng=0):
    rng = np.random.default_rng(rng)
    X = rng.normal(0, 1, size=(n, d))
    y = (X > 0).sum(axis=1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def make_highdim_xor(n=1000, d=30, rng=0):
    rng = np.random.default_rng(rng)
    X = rng.normal(0, 1, size=(n, d))
    bits = (X > 0).astype(int)
    y = bits.sum(axis=1) % 2
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def make_logistic(n=1000, d=30, rng=0):
    torch.manual_seed(rng)
    mean = torch.zeros(d)
    cov = torch.eye(d) * 0.5 + torch.ones((d, d)) * 0.5
    dist = torch.distributions.MultivariateNormal(mean, cov)
    X = dist.sample((n,))
    beta0 = 1.0
    beta = torch.ones(d, 1)
    logit = 1 / (1 + torch.exp(-(beta0 + X @ beta)))
    X_tensor = X.float()
    y_tensor = logit.clone().detach().float()
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def make_smooth_nonlinear(n=1000, d=30, rng=0):
    """
    y = sin(x1) + x2² + x3*x4 + Σx[4:15] + noise
    混合非线性与线性项，两种模型都能得到合理拟合，适合对比。
    非线性强度可通过调整线性特征数量控制（当前 11 个线性特征）。
    """
    rng_gen = np.random.default_rng(rng)
    X = rng_gen.normal(0, 1, size=(n, d))
    y = (
        np.sin(X[:, 0])               # 非线性：sin
        + X[:, 1] ** 2                # 非线性：平方
        + X[:, 2] * X[:, 3]           # 非线性：交互项
        + X[:, 4:15].sum(axis=1)      # 线性部分（11 个特征）
        + rng_gen.normal(0, 0.3, size=n)  # 噪声
    )
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def generate_dataset(case="smooth_nonlinear", n=500, d=30, rng=0):
    if case == "normal":
        return poly_data(n=n, d=d)
    elif case == "classified":
        return make_classified(n=n, d=d, rng=rng)
    elif case == "hiddenstep":
        return make_highdim_step(n=n, d=d, rng=rng)
    elif case == "hiddenXOR":
        return make_highdim_xor(n=n, d=d, rng=rng)
    elif case == "Logit":
        return make_logistic(n=n, d=d, rng=rng)
    elif case == "smooth_nonlinear":
        return make_smooth_nonlinear(n=n, d=d, rng=rng)
    else:
        raise ValueError(f"未知数据集：{case}")

# ── 使用 smooth_nonlinear 数据集 ──
X_train, X_test, y_train, y_test = generate_dataset(case="smooth_nonlinear", n=10000, d=30)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=64)


# ─────────────────────────────────────────────
# 神经网络定义
# ─────────────────────────────────────────────
class YieldRegressor(nn.Module):
    def __init__(self, input_size):
        super(YieldRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


input_size = d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YieldRegressor(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

_hooks = register_hooks(model)


# ─────────────────────────────────────────────
# 训练 & 测试
# ─────────────────────────────────────────────
def train_model(epochs=20, capture_mode="first_batch_per_epoch"):
    global _capture_this_step, current_epoch
    model.train()
    for epoch in range(epochs):
        current_epoch = epoch
        total_loss = 0.0
        _capture_this_step = (capture_mode in ("all", "first_batch_per_epoch"))

        for step, (X_here, Y_here) in enumerate(train_loader):
            X_here, Y_here = X_here.to(device), Y_here.to(device)
            outputs = model(X_here)
            loss = criterion(outputs, Y_here)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

            if capture_mode == "first_batch_per_epoch":
                _capture_this_step = False

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

def test_model():
    global _capture_this_step
    model.eval()
    total_loss = 0.0
    _capture_this_step = (CAPTURE_MODE == "first_batch_per_epoch")

    with torch.no_grad():
        for step, (X_here, Y_here) in enumerate(test_loader):
            X_here, Y_here = X_here.to(device), Y_here.to(device)
            outputs = model(X_here)
            loss = criterion(outputs, Y_here)
            total_loss += loss.item()
            if CAPTURE_MODE == "first_batch_per_epoch":
                _capture_this_step = False

    avg_loss = total_loss / len(test_loader)
    print(f"神经网络 Test MSE: {avg_loss:.4f}")
    return avg_loss


# ─────────────────────────────────────────────
# 多项式特征工程（线性回归用）
# ─────────────────────────────────────────────
def poly_interaction_features(X, max_order=None):
    n_samples, n_features = X.shape
    if max_order is None:
        max_order = 1
    features, powers = [], []
    for total_degree in range(1, max_order + 1):
        for exponents in product(range(total_degree + 1), repeat=n_features):
            if sum(exponents) == total_degree:
                feature = np.prod(X ** exponents, axis=1)
                features.append(feature)
                powers.append(exponents)
    return np.column_stack(features), powers

def custom_features_full(X, max_order=None, include_special=True, full_poly=True):
    base_features, powers = poly_interaction_features(X, max_order)
    all_features = [base_features]

    if include_special:
        special_features = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            xi_safe_log = np.where(xi <= 0, 1e-6, xi)
            special_features.extend([
                np.log(xi_safe_log),
                np.exp(np.clip(xi, -30, 30)),
                np.exp(np.clip(xi ** 2, 0, 30))
            ])
        all_features.append(np.column_stack(special_features))

    combined = np.column_stack(all_features)

    if full_poly:
        expanded, _ = poly_interaction_features(combined, max_order)
        return expanded
    else:
        return combined


# ─────────────────────────────────────────────
# 几何距离函数
# ─────────────────────────────────────────────
def pooled_covariance(A, B, shrinkage=None, ridge=0.0):
    X = np.vstack([A, B])
    if shrinkage is None:
        cov = np.cov(X, rowvar=False)
    elif shrinkage == 'ledoitwolf':
        cov = LedoitWolf().fit(X).covariance_
    else:
        raise ValueError("Unsupported shrinkage")
    if ridge > 0:
        cov = cov + ridge * np.eye(cov.shape[0])
    VI = np.linalg.pinv(cov)
    return cov, VI

def mahalanobis(A, B, shrinkage='ledoitwolf', ridge=0.0):
    muA, muB = A.mean(axis=0), B.mean(axis=0)
    _, VI = pooled_covariance(A, B, shrinkage=shrinkage, ridge=ridge)
    return distance.mahalanobis(muA, muB, VI)

def kendall_shape_distance(A: np.ndarray,
                           B: np.ndarray,
                           n_landmarks: int = 16,
                           atol: float = 1e-12) -> float:
    """
    先用 PCA 把 (n_samples, d) 压成 (n_landmarks, n_landmarks) 的地标配置矩阵，
    再算 Kendall 形状距离。
    避免原始 k*m 过大导致 geomstats 分配巨型单位矩阵而爆内存。
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    n_comp = min(n_landmarks, A.shape[1], B.shape[1], A.shape[0] - 1)

    # PCA 降维：(n_samples, d) → (n_samples, n_comp)
    A_proj = PCA(n_components=n_comp).fit_transform(A)
    B_proj = PCA(n_components=n_comp).fit_transform(B)

    # 取前 n_comp 行作为地标配置：(n_comp, n_comp)
    landmark_A = A_proj[:n_comp, :]
    landmark_B = B_proj[:n_comp, :]

    k, m = landmark_A.shape
    ps = PreShapeSpace(k_landmarks=k, ambient_dim=m)

    ZA = ps.projection(landmark_A)
    ZB = ps.projection(landmark_B)

    if np.allclose(ZA, ZB, atol=atol):
        return 0.0

    d = float(ps.metric.dist(ZA, ZB))
    return max(d, 0.0)

def next_name(name):
    k = int(name.split(".")[1])
    return f"model.{k+1}"

def _pooled_cov_inv(X, Y, shrinkage='ledoitwolf', ridge=0.0):
    Z = np.vstack([X, Y])
    if shrinkage is None:
        cov = np.cov(Z, rowvar=False)
    elif shrinkage == 'ledoitwolf':
        cov = LedoitWolf().fit(Z).covariance_
    else:
        raise ValueError("shrinkage must be None or 'ledoitwolf'")
    if ridge > 0:
        cov = cov + ridge * np.eye(cov.shape[0])
    return np.linalg.pinv(cov)

def _paired_mahalanobis(A, B, VI):
    D  = A - B
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
    items.sort(key=lambda p: int(p[0].split(".")[1]))

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

        VI   = _pooled_cov_inv(A_aligned, B_aligned, shrinkage=shrinkage, ridge=ridge)
        dvec = _paired_mahalanobis(A_aligned, B_aligned, VI)

        results[name] = dvec
        summary[name] = (float(dvec.mean()), float(dvec.std()))

    return results, summary


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. 训练神经网络
    train_model(epochs=20)
    nn_test_mse = test_model()
    torch.save(model.state_dict(), "yield_regressor.pth")

    # 2. 线性回归对比
    X_train_np = X_train.cpu().detach().numpy()
    X_test_np  = X_test.cpu().detach().numpy()
    y_train_np = y_train.cpu().detach().numpy()
    y_test_np  = y_test.cpu().detach().numpy()

    print("\n正在构建多项式特征...")
    X_features      = custom_features_full(X_train_np, full_poly=None)
    X_test_features = custom_features_full(X_test_np,  full_poly=None)
    print(f"特征维度：训练 {X_features.shape}，测试 {X_test_features.shape}")

    lr_model = LinearRegression()
    lr_model.fit(X_features, y_train_np)
    y_pred = lr_model.predict(X_test_features)

    from sklearn.metrics import mean_squared_error, r2_score
    lr_mse = mean_squared_error(y_test_np, y_pred)
    lr_r2  = r2_score(y_test_np, y_pred)

    print("\n========== 模型对比 ==========")
    print(f"神经网络  Test MSE : {nn_test_mse:.4f}")
    print(f"线性回归  Test MSE : {lr_mse:.4f}")
    print(f"线性回归  R²       : {lr_r2:.4f}")
    print("================================\n")

    # 3. 整理 hook 采集到的激活数据
    linear_epoch_concat     = {}
    activation_epoch_concat = {}

    for ep, layer_dict in linear_by_epoch.items():
        linear_epoch_concat[ep] = {
            lname: torch.cat(outs, dim=0).numpy()
            for lname, outs in layer_dict.items()
        }
    for ep, layer_dict in activation_by_epoch.items():
        activation_epoch_concat[ep] = {
            aname: torch.cat(outs, dim=0).numpy()
            for aname, outs in layer_dict.items()
        }

    # 4. 逐 epoch 计算马氏距离 & Kendall 形状距离
    print("========== 逐 Epoch 几何距离 ==========")
    epoch_stats = {}

    for ep in sorted(linear_epoch_concat.keys()):
        mdists, sdists = {}, {}
        Ls = linear_epoch_concat[ep]
        As = activation_epoch_concat.get(ep, {})

        for lname, L in Ls.items():
            aname = next_name(lname)
            if aname in As and L.shape[1] == As[aname].shape[1]:
                mdists[f"{lname} vs {aname}"] = mahalanobis(
                    L, As[aname], shrinkage='ledoitwolf', ridge=1e-6
                )
                sdists[f"{lname} vs {aname}"] = kendall_shape_distance(
                    L, As[aname]
                )

        epoch_stats[ep] = {"mahal": mdists, "shape": sdists}

    for ep in sorted(epoch_stats.keys()):
        print(f"\nEpoch {ep}")
        pairs = sorted(
            set(epoch_stats[ep]["mahal"].keys()) | set(epoch_stats[ep]["shape"].keys())
        )
        for p in pairs:
            m = epoch_stats[ep]["mahal"].get(p, float("nan"))
            s = epoch_stats[ep]["shape"].get(p, float("nan"))
            print(f"  {p}: mahal={m:.6f} | shape={s:.6f}")

    # 5. 逐样本成对马氏距离（以第一层输出为参照）
    # 取最后一个 epoch 的数据做示例
    last_ep = max(linear_epoch_concat.keys())
    pair_dists, stats = paired_md_from_first(
        linear_epoch_concat[last_ep],
        activation_epoch_concat.get(last_ep, {}),
        source_name="model.0",
        align="auto",
        shrinkage="ledoitwolf",
        ridge=1e-6,
    )

    print("\n========== 逐样本成对马氏距离（最后一个 Epoch） ==========")
    for layer, dvec in pair_dists.items():
        print(f"  {layer}: mean={dvec.mean():.4f}  std={dvec.std():.4f}")
