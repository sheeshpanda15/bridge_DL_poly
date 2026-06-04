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
import re


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
        global _capture_this_step
        if _capture_this_step:
            linear_outputs[name].append(to_cpu_detached(output))
        
    return hook

def make_activation_hook(name):
    def hook(module, inputs, output):
        global _capture_this_step
        if _capture_this_step:
            activations[name].append(to_cpu_detached(output))
            # 想实时看形状就打开下一行：
            # print(f"[Activation] {name}: {output.shape}")
    return hook

# 只给叶子层注册：Linear → 线性hook；ReLU/Sigmoid/Tanh → 激活hook
def register_hooks(model):
    hooks = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:  # 叶子模块
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(make_linear_hook(name)))
            elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
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
mean = np.zeros(d)
cov = np.eye(d)
data_x = np.random.multivariate_normal(mean, cov, size=d**2)
beta = np.ones(d)
data_y= data_x @ beta + np.random.normal(0, 0.1, size=d**2) 

X_tensor = torch.tensor(data_x, dtype=torch.float32)
y_tensor = torch.tensor(data_y, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
# 构建 DataLoader
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
def train_model(epochs=5):
    global _capture_this_step
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        # 每个 epoch 只采第一个 batch（若 CAPTURE_MODE="all"，下面的 if 不会生效）
        _capture_this_step = (CAPTURE_MODE == "first_batch_per_epoch")

        for step, (X_here, Y_here) in enumerate(train_loader):
            X_here, Y_here = X_here.to(device), Y_here.to(device)

            outputs = model(X_here)                 # ← 这里会触发 hook
            loss = criterion(outputs, Y_here)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if CAPTURE_MODE == "first_batch_per_epoch":
                _capture_this_step = False         # 只记第一个 batch

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

def kendall_shape_distance(A: np.ndarray, B: np.ndarray) -> float:
    k, m = A.shape
    ps = PreShapeSpace(k_landmarks=k, ambient_dim=m)
    # 你可以直接传原始 A, B；也可以显式投影后再传
    ZA = ps.projection(A)
    ZB = ps.projection(B)
    return ps.metric.dist(ZA, ZB)



def pooled_covariance(A: np.ndarray, B: np.ndarray, shrinkage=None, ridge=0.0):
    """
    A: (nA, d), B: (nB, d)
    shrinkage: None | 'ledoitwolf'
    ridge: 额外对角正则系数 λ（>=0）
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
    X_test_features = custom_features_full(X_test_np,full_poly=None)
    model = LinearRegression()
    model.fit(X_features, y_train_np)
    y_pred = model.predict(X_test_features)

    from sklearn.metrics import mean_squared_error, r2_score
    print("MSE:", mean_squared_error(y_test_np, y_pred))
    print("R²:", r2_score(y_test_np, y_pred))
    
linear_concat = {}
for name, outs in linear_outputs.items():
    linear_concat[name] = torch.cat(outs, dim=0).numpy()
    #print(f"[Linear] {name} -> {linear_concat[name].shape}")

# 收集所有激活层的完整输出到一个字典
activation_concat = {}
for name, outs in activations.items():
    activation_concat[name] = torch.cat(outs, dim=0).numpy()
    #print(f"[Act] {name} -> {activation_concat[name].shape}")


mdists = {}
sdists = {}
for lname, L in linear_concat.items():
    aname = next_name(lname)
    if aname in activation_concat and L.shape[1] == activation_concat[aname].shape[1]:
        mdists[f"{lname} vs {aname}"] = mahalanobis(L, activation_concat[aname])
        sdists[f"{lname} vs {aname}"] = kendall_shape_distance(L, activation_concat[aname])


print("Mdistancenc")

for k, v in mdists.items():
    print(k, "->", v)

print("shape distance")

for k, v in sdists.items():
    print(k, "->", v)















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


pair_dists, stats = paired_md_from_first(
    linear_concat, activation_concat,
    source_name="model.0",
    align="auto",          # 同维直接算，不同维用 CCA
    k=None,                # CCA 维度自动取最小可行
    shrinkage="ledoitwolf",
    ridge=1e-6,            # 轻微正则，更稳
)

# 看看每层的均值/方差
#for layer, dvec in pair_dists.items():
    #print(layer, "mean:", dvec.mean(), "std:", dvec.std())

#all_means = [dvec.mean() for dvec in pair_dists.values()]
#overall_mean = np.mean(all_means)
#print("所有层均值距离的均值:", overall_mean)


    