"""
taylor_expand.py
─────────────────────────────────────────────────────────────────
逐层泰勒展开模块：把任意（含多隐层）前馈神经网络展开成一个多项式模型，
精度（泰勒阶数）可调，便于与 NN / Full-PR 做对比。

与 Morala et al. 式(6) 的区别：
  - 式(6) 只处理【单隐层】，一次泰勒展开即可。
  - 本模块处理【任意层数】：逐层展开。第 k 层把激活函数 g 在该层
    突触电位的工作点附近做泰勒展开，得到一个多项式；这个多项式再作为
    下一层的输入，继续展开。多层即多项式的复合，最终合并成单个多项式。

核心思想——“多项式在多项式上的复合仍是多项式”：
  设第 1 层输出 p_1(x) 是 x 的多项式（来自 g(W_1 x + b_1) 的泰勒展开），
  第 2 层 g(W_2 · p_1(x) + b_2) 再展开时，把 (W_2·p_1 + b_2) 当作新的“变量”，
  代入 g 的泰勒级数，再把 p_1 的多项式代进去、合并同类项，得到 p_2(x)，仍是多项式。
  以此类推。每层展开都会截断到 `order`，控制精度与项数。

用法（被主文件读取）：
  from taylor_expand import expand_nn_to_polynomial, PolynomialModel

  poly = expand_nn_to_polynomial(model, input_dim=p, order=3,
                                 expansion_point="data", X_ref=X_train)
  pred = poly.predict(X_test)          # 多项式预测，(n,1)
  print(poly.n_terms, poly.max_total_degree)

精度调节：
  - order：每层泰勒展开的截断阶数（越大越精确，但项数随层数指数增长）。
  - max_total_degree：可选，对最终多项式的总次数封顶，防止多层复合后次数爆炸。
  - expansion_point："zero" 在 0 处展开（Morala 原版）；
                     "data" 在该层突触电位的均值处展开（多层更稳，工作点更贴近实际）。
"""

import math
from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# 多项式的内部表示：{单项式指数元组: 系数}
#   指数元组长度 = 当前“变量”个数。
#   例如 3 维输入下 {(2,0,0): 1.5, (0,1,1): -0.3} 表示 1.5 x0^2 - 0.3 x1 x2
# ─────────────────────────────────────────────
class MultiPoly:
    """多元多项式（稀疏表示），支持加、数乘、乘、幂，便于逐层复合。"""

    def __init__(self, nvars, terms=None):
        self.nvars = nvars
        self.terms = defaultdict(float)
        if terms:
            for k, v in terms.items():
                self.terms[k] += v

    @classmethod
    def constant(cls, nvars, c):
        p = cls(nvars)
        if c != 0:
            p.terms[(0,) * nvars] = float(c)
        return p

    @classmethod
    def variable(cls, nvars, i):
        """第 i 个变量本身：x_i"""
        p = cls(nvars)
        e = [0] * nvars
        e[i] = 1
        p.terms[tuple(e)] = 1.0
        return p

    def copy(self):
        return MultiPoly(self.nvars, dict(self.terms))

    def __add__(self, other):
        r = self.copy()
        if isinstance(other, MultiPoly):
            for k, v in other.terms.items():
                r.terms[k] += v
        else:
            r.terms[(0,) * self.nvars] += float(other)
        return r

    __radd__ = __add__

    def scale(self, c):
        r = MultiPoly(self.nvars)
        for k, v in self.terms.items():
            r.terms[k] = v * c
        return r

    def __mul__(self, other):
        if not isinstance(other, MultiPoly):
            return self.scale(float(other))
        r = MultiPoly(self.nvars)
        for k1, v1 in self.terms.items():
            if v1 == 0:
                continue
            for k2, v2 in other.terms.items():
                if v2 == 0:
                    continue
                key = tuple(a + b for a, b in zip(k1, k2))
                r.terms[key] += v1 * v2
        return r

    def power(self, n, max_degree=None):
        """快速幂；可选按总次数截断，防爆。"""
        result = MultiPoly.constant(self.nvars, 1.0)
        base = self.copy()
        while n > 0:
            if n & 1:
                result = result * base
                if max_degree is not None:
                    result = result.truncate(max_degree)
            n >>= 1
            if n > 0:
                base = base * base
                if max_degree is not None:
                    base = base.truncate(max_degree)
        return result

    def truncate(self, max_total_degree):
        """丢弃总次数 > max_total_degree 的项。"""
        r = MultiPoly(self.nvars)
        for k, v in self.terms.items():
            if sum(k) <= max_total_degree and v != 0:
                r.terms[k] = v
        return r

    def clean(self, tol=1e-15):
        r = MultiPoly(self.nvars)
        for k, v in self.terms.items():
            if abs(v) > tol:
                r.terms[k] = v
        return r

    @property
    def max_total_degree(self):
        return max((sum(k) for k in self.terms), default=0)

    @property
    def n_terms(self):
        return len([1 for v in self.terms.values() if v != 0])

    def eval(self, X):
        """在 X (n, nvars) 上求值，返回 (n,)。"""
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(X.shape[0])
        for k, v in self.terms.items():
            if v == 0:
                continue
            term = np.full(X.shape[0], v)
            for i, e in enumerate(k):
                if e:
                    term = term * X[:, i] ** e
            out += term
        return out


# ─────────────────────────────────────────────
# 激活函数在某点的泰勒系数
# ─────────────────────────────────────────────
_ACT = {"softplus": nn.Softplus, "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid, "relu": nn.ReLU}


def activation_taylor_coeffs(act_name, order, point=0.0):
    """
    返回 g 在 x=point 处泰勒展开的系数 [c_0, c_1, ..., c_order]，
    其中 g(point + t) ≈ Σ_k c_k t^k，c_k = g^(k)(point)/k!。
    用 autograd 精确求导。ReLU 不可导，报错。
    """
    if act_name == "relu":
        raise ValueError("ReLU 在折点不可导，无法泰勒展开（用 softplus 近似）。")
    act = _ACT[act_name]()
    x = torch.tensor([float(point)], dtype=torch.float64, requires_grad=True)
    g = act(x)
    coeffs = [g.item()]              # c_0 = g(point)
    cur = g
    for k in range(1, order + 1):
        (cur,) = torch.autograd.grad(cur.sum(), x, create_graph=True)
        coeffs.append(cur.item() / math.factorial(k))
    return coeffs


# ─────────────────────────────────────────────
# 把"线性组合的多项式"代入激活函数的泰勒展开
# ─────────────────────────────────────────────
def apply_activation_poly(u_poly, act_name, order, point, max_degree):
    """
    给定突触电位的多项式 u(x)（某个神经元的 W·prev + b），
    返回 g(u(x)) 的泰勒多项式近似：
        g(u) ≈ Σ_{k=0}^{order} c_k (u - point)^k
    其中 c_k = g^(k)(point)/k!。展开 (u-point)^k 并按 max_degree 截断。
    """
    coeffs = activation_taylor_coeffs(act_name, order, point)
    shifted = u_poly + MultiPoly.constant(u_poly.nvars, -point)  # (u - point)
    result = MultiPoly.constant(u_poly.nvars, coeffs[0])
    powk = MultiPoly.constant(u_poly.nvars, 1.0)                 # (u-point)^0
    for k in range(1, order + 1):
        powk = (powk * shifted)
        if max_degree is not None:
            powk = powk.truncate(max_degree)
        if coeffs[k] != 0:
            result = result + powk.scale(coeffs[k])
    if max_degree is not None:
        result = result.truncate(max_degree)
    return result


# ─────────────────────────────────────────────
# 主函数：逐层展开整个网络
# ─────────────────────────────────────────────
class PolynomialModel:
    """展开得到的多项式模型，封装预测接口。"""

    def __init__(self, poly, input_dim, order, n_layers, expansion_point):
        self.poly = poly.clean()
        self.input_dim = input_dim
        self.order = order
        self.n_layers = n_layers
        self.expansion_point = expansion_point

    @property
    def n_terms(self):
        return self.poly.n_terms

    @property
    def max_total_degree(self):
        return self.poly.max_total_degree

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.poly.eval(X).reshape(-1, 1)

    def coefficients(self):
        """返回 {指数元组: 系数} 字典，便于与真实多项式系数对比。"""
        return dict(self.poly.terms)

    def __repr__(self):
        return (f"<PolynomialModel layers={self.n_layers} order={self.order} "
                f"point={self.expansion_point} terms={self.n_terms} "
                f"max_degree={self.max_total_degree}>")


def expand_nn_to_polynomial(model, input_dim, order=3,
                            expansion_point="zero", X_ref=None,
                            max_total_degree=None):
    """
    把一个前馈网络（nn.Sequential 结构：Linear, act, Linear, act, ..., Linear）
    逐层泰勒展开为单个多项式模型。

    参数
    ----
    model            : 训练好的 nn.Module（内部含 .model 为 Sequential，或本身可遍历）
    input_dim        : 输入维度 p
    order            : 每层泰勒展开的截断阶数（精度旋钮，越大越精确）
    expansion_point  : "zero" —— 每层都在 0 处展开（Morala 原版，输入需缩放到 0 附近）
                       "data" —— 每层在该层突触电位的样本均值处展开（多层更稳）
    X_ref            : expansion_point="data" 时必需，用于估计每层工作点的参考输入
                       （通常传训练集），(n, input_dim)
    max_total_degree : 最终多项式总次数上限（None=不限）。多层 + 高 order 时
                       次数会指数增长，强烈建议设一个上限（如 order 或 2*order）。

    返回
    ----
    PolynomialModel
    """
    # 取出 Linear / 激活 的有序列表
    seq = model.model if hasattr(model, "model") else model
    layers = [m for m in seq] if isinstance(seq, nn.Sequential) else list(seq.children())

    # 解析成 [(W,b), act_name, (W,b), act_name, ..., (W,b)] 的结构
    parsed = []
    act_map = {nn.Softplus: "softplus", nn.Tanh: "tanh",
               nn.Sigmoid: "sigmoid", nn.ReLU: "relu"}
    for m in layers:
        if isinstance(m, nn.Linear):
            parsed.append(("linear",
                           m.weight.detach().cpu().double().numpy(),
                           m.bias.detach().cpu().double().numpy()))
        else:
            for cls, name in act_map.items():
                if isinstance(m, cls):
                    parsed.append(("act", name))
                    break

    # 初始：每个输入变量是一个一次多项式 x_i
    # current[i] = 表示第 i 个“当前特征”的多项式（关于原始输入 x）
    current = [MultiPoly.variable(input_dim, i) for i in range(input_dim)]

    # 若在数据点展开，需要一份参考输入来估计每层工作点
    if expansion_point == "data":
        if X_ref is None:
            raise ValueError("expansion_point='data' 需要提供 X_ref。")
        X_ref = np.asarray(X_ref, dtype=np.float64)

    n_act_layers = sum(1 for p in parsed if p[0] == "act")

    for item in parsed:
        if item[0] == "linear":
            W, b = item[1], item[2]                 # W: (out, in), b: (out,)
            out_dim, in_dim = W.shape
            new_current = []
            for o in range(out_dim):
                # 线性组合：u_o = Σ_i W[o,i] * current[i] + b[o]
                acc = MultiPoly.constant(input_dim, b[o])
                for i in range(in_dim):
                    if W[o, i] != 0:
                        acc = acc + current[i].scale(W[o, i])
                if max_total_degree is not None:
                    acc = acc.truncate(max_total_degree)
                new_current.append(acc)
            current = new_current
        else:  # 激活层
            act_name = item[1]
            new_current = []
            for o, u_poly in enumerate(current):
                # 决定展开点
                if expansion_point == "zero":
                    point = 0.0
                else:  # data：在该神经元突触电位的样本均值处展开
                    u_vals = u_poly.eval(X_ref)
                    point = float(np.mean(u_vals))
                gpoly = apply_activation_poly(
                    u_poly, act_name, order, point, max_total_degree)
                new_current.append(gpoly)
            current = new_current

    # 最后一层是 linear（输出），current 此时应只有 1 个元素
    assert len(current) == 1, f"输出维度应为 1，得到 {len(current)}"
    final_poly = current[0]
    if max_total_degree is not None:
        final_poly = final_poly.truncate(max_total_degree)

    return PolynomialModel(final_poly, input_dim, order,
                           n_act_layers, expansion_point)


# ─────────────────────────────────────────────
# 自测：验证展开正确性 + 演示精度调节
# ─────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    p = 3
    # 构造一个两隐层网络
    net = nn.Sequential(
        nn.Linear(p, 6), nn.Softplus(),
        nn.Linear(6, 4), nn.Softplus(),
        nn.Linear(4, 1),
    )

    class Wrap(nn.Module):
        def __init__(self, seq):
            super().__init__()
            self.model = seq

        def forward(self, x):
            return self.model(x)

    model = Wrap(net)

    # 参考/测试数据（缩放到小范围，保证泰勒近似有效）
    X = np.random.uniform(-0.5, 0.5, size=(200, p))
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        nn_pred = model(Xt).numpy().ravel()

    print("逐层泰勒展开演示（两隐层 softplus 网络）\n")
    print(f"{'设置':<34}{'项数':>8}{'最高次':>8}{'vs NN 的MSE':>16}")
    print("-" * 66)

    for point in ("zero", "data"):
        for order in (1, 2, 3, 4):
            poly = expand_nn_to_polynomial(
                model, input_dim=p, order=order,
                expansion_point=point, X_ref=X,
                max_total_degree=max(4, order * 2))
            pred = poly.predict(X).ravel()
            mse = float(np.mean((pred - nn_pred) ** 2))
            tag = f"point={point}, order={order}"
            print(f"{tag:<34}{poly.n_terms:>8}{poly.max_total_degree:>8}{mse:>16.3e}")

    print("\n观察：order 越高、在 data 点展开，通常 MSE 越小（多项式越接近 NN）。")
    print("注意 order 升高时项数与最高次同步增长——这就是精度与复杂度的权衡。")
