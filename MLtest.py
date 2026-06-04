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
from geomstats.geometry.pre_shape import PreShapeSpace
from itertools import product



# 1. 数据预处理和加载
df = pd.read_csv('crop_yield.csv')
X = df.drop(columns=['Yield_tons_per_hectare'])
y = df['Yield_tons_per_hectare']

categorical_cols = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
boolean_cols = ['Fertilizer_Used', 'Irrigation_Used']
numeric_cols = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']



preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('bool', 'passthrough', boolean_cols),
    ('num', StandardScaler(), numeric_cols)
])


X_transformed = preprocessor.fit_transform(X)

# 转为 PyTorch 张量
if hasattr(X_transformed, "toarray"):
    X_tensor = torch.tensor(X_transformed.toarray(), dtype=torch.float32)
else:
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)

y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 构建 DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

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
input_size = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YieldRegressor(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
def train_model(epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_here, Y_here in train_loader:
            X_here, Y_here = X_here.to(device), Y_here.to(device)

            outputs = model(X_here)
            loss = criterion(outputs, Y_here)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 5. 测试模型
def test_model():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_here, Y_here in test_loader:
            X_here, Y_here = X_here.to(device), Y_here.to(device)
            outputs = model(X_here)
            loss = criterion(outputs, Y_here)
            total_loss += loss.item()
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
        

