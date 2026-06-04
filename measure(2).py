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

linear_by_epoch     = defaultdict(lambda: defaultdict(list))
activation_by_epoch = defaultdict(lambda: defaultdict(list))
current_epoch = -1   # 训练循环里会更新
# 两个“同一个变量”：分别保存
linear_outputs = defaultdict(list)   # 线性层（未激活）
activations = defaultdict(list)      # 激活层（已激活）

# 采集频率：每个 epoch 只采第一个 batch；若想每个 batch 都采，用 "all"
CAPTURE_MODE = "first_batch_per_epoch"
_capture_this_step = True            # 运行时开关，训练/测试循环里会改它

# 递归搬到 CPU 并 detach，避免显存/计算图占用
def to_cpu_detached(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    elif isinstance(x, (list, tuple)):
        return type(x)(to_cpu_detached(t) for t in x)
    elif isinstance(x, dict):
        return {k: to_cpu_detached(v) for k, v in x.items()}
    return x

# —— 两类 hook —— #
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
# 只给叶子层注册：Linear → 线性hook；ReLU/Sigmoid/Tanh → 激活hook

def register_hooks(model):
    hooks = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:  # 叶子模块
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(make_linear_hook(name)))
            elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Tanh,nn.Identity)):
                hooks.append(m.register_forward_hook(make_activation_hook(name)))
    return hooks

def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# 1. 数据预处理和加载
#df = pd.read_csv('crop_yield.csv')
#X = df.drop(columns=['Yield_tons_per_hectare'])
#y = df['Yield_tons_per_hectare']

#categorical_cols = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
#boolean_cols = ['Fertilizer_Used', 'Irrigation_Used']
#numeric_cols = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
#preprocessor = ColumnTransformer(transformers=[
#    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#    ('bool', 'passthrough', boolean_cols),
#    ('num', StandardScaler(), numeric_cols)
#])


#X_transformed = preprocessor.fit_transform(X)

# 转为 PyTorch 张量
#if hasattr(X_transformed, "toarray"):
#    X_tensor = torch.tensor(X_transformed.toarray(), dtype=torch.float32)
#else:
#    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)

#y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

# 划分训练测试集
#X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

d = 30



# 构建 DataLoader



def poly_data(n=1000, d=30):
    mean = np.zeros(d)

    cov = np.eye(d)

    # 生成 100 个样本，每个样本是 100 维
    data_x = np.random.multivariate_normal(mean, cov, size=n)
    beta = np.ones(d)
    data_y= data_x @ beta + np.random.normal(0, 0.1, size=n)  # 加点噪声
    X_tensor = torch.tensor(data_x, dtype=torch.float32)
    y_tensor = torch.tensor(data_y, dtype=torch.float32).unsqueeze(1)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def make_classified(n=1000, d=30, rng=0):
     
    X, y = make_classification(
    n_samples=n, 
    n_features=d, 
    n_informative=d-5, 
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
    )

    # 将X缩放到[-4, 4]
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
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def make_highdim_xor(n=1000, d=30, rng=0):
    rng = np.random.default_rng(rng)
    X = rng.normal(0, 1, size=(n, d))
    bits = (X > 0).astype(int)  # (n, d)
    y = bits.sum(axis=1) % 2
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def make_logistic(n=1000, d=30, rng=0):
    torch.manual_seed(rng)
    mean = torch.zeros(d)
    # 构造一个正定协方差矩阵（稍微带点相关性）
    #A = torch.randn(d, d)
    #cov = A @ A.T + torch.eye(d) * 0.1
    cov = torch.eye(d) * 0.5+ torch.ones((d, d)) * 0.5# 保证正定性
    dist = torch.distributions.MultivariateNormal(mean, cov)
    X = dist.sample((n,))
    beta0=1.0# n个样本
    beta = torch.ones(d, 1)
    logit = 1/(1+torch.exp(-(beta0+ X @ beta)))
    
    X_tensor = X.float()
    y_tensor = logit.clone().detach().float()
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def binomial(n=1000,d=30,rng=0):
    rng=np.random.default_rng(rng)
    X = np.random.randint(0, 2, (n, d))
    prob=rng.normal(0,1,size=(1,d))



def generate_dataset(case="normal",n=500,d=30,rng=0):
    if case=="normal":
        X_train, X_test, y_train, y_test=poly_data(n=n,d=30)
    elif case=="classified": 
        X_train, X_test, y_train, y_test=make_classified(n=n , d=d,rng=rng)
    elif case=="hiddenstep":
        X_train, X_test, y_train, y_test=make_highdim_step(n=n,d=d,rng=rng)
    elif case=="hiddenXOR":
        X_train, X_test, y_train, y_test=make_highdim_xor(n=n,d=d,rng=rng)
    elif case=="Logit":
        X_train, X_test, y_train, y_test=make_logistic(n=n,d=d,rng=rng)

    return X_train, X_test, y_train, y_test
        
X_train, X_test, y_train, y_test=generate_dataset(case="Logit",n=128,d=30)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128)


# 2. 定义神经网络模型
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
        
    #def __init__(self, input_size):
    #    super(YieldRegressor, self).__init__()
    #    self.model = nn.Sequential(
    #        nn.Linear(input_size, 64),
    #        nn.Identity(),
    #        nn.Linear(64, 128),
    #        nn.Identity(),
    #        nn.Linear(128, 32),
    #        nn.Identity(),
    #        nn.Linear(32, 1)
    #    )

    def forward(self, x):
        return self.model(x)




# 3. 初始化模型、损失函数和优化器
input_size = d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YieldRegressor(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 注册 hooks（很关键）
_hooks = register_hooks(model)

# 4. 训练模型
def train_model(epochs=5, capture_mode="first_batch_per_epoch"):
    global _capture_this_step, current_epoch
    model.train()
    for epoch in range(epochs):
        current_epoch = epoch
        total_loss = 0.0

        # 这一行决定本 epoch 的采样频率
        _capture_this_step = (capture_mode == "all") or (capture_mode == "first_batch_per_epoch")

        for step, (X_here, Y_here) in enumerate(train_loader):
            X_here, Y_here = X_here.to(device), Y_here.to(device)
            outputs = model(X_here)     # ← 这里触发 hook，数据会进 linear_by_epoch[epoch][layer]
            loss = criterion(outputs, Y_here)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

            # 如果只想要“每个 epoch 的第一个 batch”，关掉剩余 batch 的开关
            if capture_mode == "first_batch_per_epoch":
                _capture_this_step = False

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 5. 测试模型
def test_model():
    global _capture_this_step
    model.eval()
    total_loss = 0.0

    # 测试时是否记录：想记第一个测试 batch 就保持 True；不想就改成 False
    _capture_this_step = (CAPTURE_MODE == "first_batch_per_epoch")

    with torch.no_grad():
        for step, (X_here, Y_here) in enumerate(test_loader):
            X_here, Y_here = X_here.to(device), Y_here.to(device)
            outputs = model(X_here)                 # ← 同样触发 hook
            loss = criterion(outputs, Y_here)
            total_loss += loss.item()

            if CAPTURE_MODE == "first_batch_per_epoch":
                _capture_this_step = False

    avg_loss = total_loss / len(test_loader)
    print(f"Test MSE: {avg_loss:.4f}")


def poly_interaction_features(X, max_order=None ):
    n_samples, n_features = X.shape
    if max_order is None:
        max_order = 1
    features = []
    powers = []
    

    for total_degree in range(1, max_order + 1):
        for exponents in product(range(total_degree + 1), repeat=n_features):
            if sum(exponents) == total_degree:
                # 每个样本点的组合值 = 所有特征按幂相乘
                feature = np.prod(X ** exponents, axis=1)
                features.append(feature)
                powers.append(exponents)

    return np.column_stack(features), powers

def custom_features_full(X, max_order=None, include_special=True,full_poly=True):
    
    base_features, powers = poly_interaction_features(X, max_order)
    
    all_features = [base_features]

    if include_special:
        special_features = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            xi_safe_log = np.where(xi <= 0, 1e-6, xi)

            special_features.extend([
                np.log(xi_safe_log),
                np.exp(xi),
                np.exp(xi ** 2)
            ])
        special_matrix = np.column_stack(special_features)
        all_features.append(special_matrix)

    # 合并 base 和 special
    combined = np.column_stack(all_features)

    if full_poly:
        # 对所有特征（包含特殊项）做交叉组合
        expanded, _ = poly_interaction_features(combined, max_order)
        return expanded
    else:
        # 仅合并但不再组合
        return combined

def pooled_covariance(A: np.ndarray, B: np.ndarray, shrinkage=None, ridge=0.0):
    """
    A: (nA, d), B: (nB, d)
    shrinkage: None | 'ledoitwolf'
    ridge: 额外对角正则系数 λ（>=0)
    return: covariance matrix Σ (d,d) and its inverse (or pseudo-inverse) VI
    """
    X = np.vstack([A, B])  # 合并样本
    if shrinkage is None:
        cov = np.cov(X, rowvar=False)
    elif shrinkage == 'ledoitwolf':
        cov = LedoitWolf().fit(X).covariance_
    else:
        raise ValueError("Unsupported shrinkage")

    if ridge > 0:
        cov = cov + ridge * np.eye(cov.shape[0])

    # 用伪逆更稳健（高维或病态时避免报错）
    VI = np.linalg.pinv(cov)
    return cov, VI

# === 用法 1：两个分布中心（均值向量）的马氏距离 ===
def mahalanobis(A, B, shrinkage='ledoitwolf', ridge=0.0):
    muA, muB = A.mean(axis=0), B.mean(axis=0)
    _, VI = pooled_covariance(A, B, shrinkage=shrinkage, ridge=ridge)
    return distance.mahalanobis(muA, muB, VI)
def next_name(name):
    # "model.0" -> "model.1"
    k = int(name.split(".")[1])
    return f"model.{k+1}"
#def kendall_shape_distance(A: np.ndarray, B: np.ndarray) -> float:
   # k, m = A.shape
    #ps = PreShapeSpace(k_landmarks=k, ambient_dim=m)
   # # 你可以直接传原始 A, B；也可以显式投影后再传
   # ZA = ps.projection(A)
   # ZB = ps.projection(B)
   # return ps.metric.dist(ZA, ZB)

def kendall_shape_distance(A: np.ndarray,
                           B: np.ndarray,
                           atol: float = 1e-12) -> float:
    """
    与你原函数等价，但：
    1) 先用 allclose 早退（避免极小角度下 arccos 放大误差）
    2) 对 dist 结果做数值裁剪：负零/极小正数直接置 0
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    k, m = A.shape
    ps = PreShapeSpace(k_landmarks=k, ambient_dim=m)

    ZA = ps.projection(A)  # 去平移+尺度 -> 预形
    ZB = ps.projection(B)

    # 早退：预形上几乎相等就直接返回 0
    if np.allclose(ZA, ZB, atol=atol):
        return 0.0

    d = float(ps.metric.dist(ZA, ZB))

    # 安全裁剪：消掉负零和微小的下溢
    if d < 0:
        d = 0.0
    if d < atol:
        return 0.0
    return d

# 6. 执行训练和测试
if __name__ == "__main__":
    train_model(epochs=5)
    test_model()
    torch.save(model.state_dict(), "yield_regressor.pth")
    
    X_train_np = X_train.cpu().detach().numpy()
    X_test_np = X_test.cpu().detach().numpy()
    y_train_np = y_train.cpu().detach().numpy()
    y_test_np = y_test.cpu().detach().numpy()
    
    X_features = custom_features_full(X_train_np,full_poly=None)
    print("完成了这一步")
    X_test_features = custom_features_full(X_test_np,full_poly=None)
    print("完成了第二部")
    model = LinearRegression()
    model.fit(X_features, y_train_np)
    y_pred = model.predict(X_test_features)

    from sklearn.metrics import mean_squared_error, r2_score
    print("MSE:", mean_squared_error(y_test_np, y_pred))
    print("R²:", r2_score(y_test_np, y_pred))
    
#linear_concat = {}
#for name, outs in linear_outputs.items():
  #  linear_concat[name] = torch.cat(outs, dim=0).numpy()
 #   print(f"[Linear] {name} -> {linear_concat[name].shape}")

# 收集所有激活层的完整输出到一个字典
#activation_concat = {}
#for name, outs in activations.items():
 #   activation_concat[name] = torch.cat(outs, dim=0).numpy()
 #   print(f"[Act] {name} -> {activation_concat[name].shape}")


# 把 list[Tensor] → (N_epoch, d) 的 numpy
linear_epoch_concat = {}
activation_epoch_concat = {}

for ep, layer_dict in linear_by_epoch.items():
    linear_epoch_concat[ep] = {lname: torch.cat(outs, dim=0).numpy()
                               for lname, outs in layer_dict.items()}

for ep, layer_dict in activation_by_epoch.items():
    activation_epoch_concat[ep] = {aname: torch.cat(outs, dim=0).numpy()
                                   for aname, outs in layer_dict.items()}






#mdists = {}
#sdists = {}
#for lname, L in linear_concat.items():
#    aname = next_name(lname)
#    if aname in activation_concat and L.shape[1] == activation_concat[aname].shape[1]:
#        mdists[f"{lname} vs {aname}"] = mahalanobis(L, activation_concat[aname])
#        sdists[f"{lname} vs {aname}"] = kendall_shape_distance(L, activation_concat[aname])


#print("Mdistancenc")

#for k, v in mdists.items():
#    print(k, "->", v)

#print("shape distance")

#for k, v in sdists.items():
#    print(k, "->", v)


def _pooled_cov_inv(X: np.ndarray, Y: np.ndarray,
                    shrinkage: str | None = 'ledoitwolf',
                    ridge: float = 0.0) -> np.ndarray:
    """合并样本估计 Σ，并返回其(伪)逆 VI。"""
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

def _paired_mahalanobis(A: np.ndarray, B: np.ndarray, VI: np.ndarray) -> np.ndarray:
    """
    逐样本成对马氏距离，向量化实现：
    d_i = sqrt( (a_i - b_i)^T VI (a_i - b_i) )
    """
    D = A - B                     # (N, d)
    DV = D @ VI                   # (N, d)
    d2 = np.sum(DV * D, axis=1)   # (N,)
    d2 = np.maximum(d2, 0.0)      # 数值稳定
    return np.sqrt(d2)


def paired_md_from_first(
    linear_concat: dict[str, np.ndarray],
    activation_concat: dict[str, np.ndarray],
    source_name: str = "model.0",
    align: str = "auto",          # "auto" | "match" | "cca"
    k: int | None = None,         # CCA 目标维；None→min(d_src,d_tgt,N-1)
    shrinkage: str | None = "ledoitwolf",
    ridge: float = 0.0,
):
    """
    返回：{层名: 距离向量(N,)}，另外返回一个简单的统计表 {层名: (mean, std)}
    """
    assert source_name in linear_concat, f"{source_name} 不在 linear_concat 中"
    src = linear_concat[source_name]        # (N, d_src)
    N, d_src = src.shape

    # 合并 linear 与 activation，按编号排序
    all_layers = {**linear_concat, **activation_concat}
    items = [(n, X) for n, X in all_layers.items() if n != source_name]
    items.sort(key=lambda p: int(p[0].split(".")[1]))

    results = {}     # 层名 -> (N,) 距离向量
    summary = {}     # 层名 -> (mean, std)

    for name, tgt in items:
        if tgt.shape[0] != N:
            raise ValueError(f"{name} 的样本数与 {source_name} 不一致：{tgt.shape[0]} vs {N}")

        d_tgt = tgt.shape[1]

        # 选择对齐策略
        mode = align
        if mode not in {"auto", "match", "cca"}:
            raise ValueError("align must be 'auto', 'match', or 'cca'")
        if mode == "auto":
            mode = "match" if d_src == d_tgt else "cca"

        if mode == "match":
            if d_src != d_tgt:
                # 跳过不同维度
                continue
            A_aligned, B_aligned = src, tgt
        else:  # "cca"
            k_eff = k or min(d_src, d_tgt, max(N - 1, 1))
            cca = CCA(n_components=k_eff, max_iter=5000)
            A_aligned, B_aligned = cca.fit_transform(src, tgt)  # (N, k_eff)

        # 协方差(伪)逆（pooled）
        VI = _pooled_cov_inv(A_aligned, B_aligned, shrinkage=shrinkage, ridge=ridge)
        # 逐样本成对马氏距离（向量化）
        dvec = _paired_mahalanobis(A_aligned, B_aligned, VI)

        results[name] = dvec
        summary[name] = (float(dvec.mean()), float(dvec.std()))

    return results, summary


#pair_dists, stats = paired_md_from_first(
    #linear_concat, activation_concat,
    #source_name="model.0",
    #align="auto",          # 同维直接算，不同维用 CCA
    #k=None,                # CCA 维度自动取最小可行
    #shrinkage="ledoitwolf",
    #ridge=1e-6,            # 轻微正则，更稳
#)

# 看看每层的均值/方差
#for layer, dvec in pair_dists.items():
#    print(layer, "mean:", dvec.mean(), "std:", dvec.std())

#all_means = [dvec.mean() for dvec in pair_dists.values()]
#overall_mean = np.mean(all_means)
#print("所有层均值距离的均值:", overall_mean)
def next_name(name):
    k = int(name.split(".")[1])
    return f"model.{k+1}"

epoch_stats = {}  # ep -> { "mahal":{pair:val}, "shape":{pair:val} }

for ep in sorted(linear_epoch_concat.keys()):
    mdists, sdists = {}, {}
    Ls = linear_epoch_concat[ep]
    As = activation_epoch_concat.get(ep, {})
    for lname, L in Ls.items():
        aname = next_name(lname)
        if aname in As and L.shape[1] == As[aname].shape[1]:
            mdists[f"{lname} vs {aname}"] = mahalanobis(L, As[aname], shrinkage='ledoitwolf', ridge=1e-6)
            sdists[f"{lname} vs {aname}"] = kendall_shape_distance(L, As[aname])
    epoch_stats[ep] = {"mahal": mdists, "shape": sdists}


for ep in sorted(epoch_stats.keys()):
    print(f"Epoch {ep}")
    pairs = sorted(set(epoch_stats[ep]["mahal"].keys()) | set(epoch_stats[ep]["shape"].keys()))
    for p in pairs:
        m = epoch_stats[ep]["mahal"].get(p, float("nan"))
        s = epoch_stats[ep]["shape"].get(p, float("nan"))
        print(f"  {p}: mahal={m:.6f} | shape={s:.6f}")

