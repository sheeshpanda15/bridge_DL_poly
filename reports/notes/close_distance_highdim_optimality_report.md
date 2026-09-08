# 高维近距离 D/A/I-optimal 迁移与 activation 对照报告

## 实验目的

本实验在高维 close-distance D-optimal 测试基础上加入 A-optimal 和 I-optimal 选点准则，并把 NN activation 从默认 tanh 扩展到 identity 线性对照。核心问题是：当 NN oracle 与二阶 FullPR 在同一数据区域内距离较近时，不同实验设计准则是否比同预算随机采样更高效，以及这种优势是否依赖 NN 的非线性激活。

## 设置

- p values：5, 10, 20, 50。
- activations：tanh, identity。
- cases：highdim_poly2, highdim_smooth, highdim_strong。
- criteria：D-optimal, A-optimal, I-optimal, Latin hypercube。
- n：10000；训练/评估：7500/2500。
- NN hidden：128,64,32；epochs=1000。
- FullPR：degree=2，include_special=True。
- close filter：`shape_nn_fpr <= 5.000e-03`。
- base runs：144；通过 close filter：137；跳过：7。

## 准则含义

- D-optimal：增大信息矩阵行列式，减少参数置信椭球体积。
- A-optimal：减少信息矩阵逆矩阵的 trace，偏向降低平均参数方差。
- I-optimal：减少评估区域的平均预测方差，更直接面向目标区域预测稳定性。
- Latin hypercube：几何覆盖基线，不使用 FullPR 信息矩阵。

## 近距离筛选摘要

| activation | p | case | used runs | median shape | median FPR mimic MSE | median RMSE gap | design params |
|---|---:|---|---:|---:|---:|---:|---:|
| identity | 5 | Quadratic | 6 | 3.303e-14 | 1.320e-14 | 2.944e-09 | 31 |
| identity | 5 | Smooth | 6 | 9.106e-14 | 3.869e-14 | 1.799e-08 | 31 |
| identity | 5 | Strong nonlinear | 6 | 8.973e-14 | 4.194e-14 | 2.631e-08 | 31 |
| identity | 10 | Quadratic | 6 | 1.616e-14 | 1.148e-14 | 1.089e-09 | 86 |
| identity | 10 | Smooth | 6 | 4.762e-14 | 3.315e-14 | 1.288e-08 | 86 |
| identity | 10 | Strong nonlinear | 6 | 4.958e-14 | 3.285e-14 | 3.662e-08 | 86 |
| identity | 20 | Quadratic | 6 | 8.211e-15 | 1.351e-14 | 1.788e-09 | 271 |
| identity | 20 | Smooth | 6 | 2.637e-14 | 4.338e-14 | 1.577e-08 | 271 |
| identity | 20 | Strong nonlinear | 6 | 3.091e-14 | 7.454e-14 | 3.847e-08 | 271 |
| identity | 50 | Quadratic | 6 | 7.321e-15 | 2.864e-14 | 8.225e-10 | 1426 |
| identity | 50 | Smooth | 6 | 2.581e-14 | 1.535e-13 | 2.525e-08 | 1426 |
| identity | 50 | Strong nonlinear | 6 | 2.716e-14 | 1.461e-13 | 6.730e-08 | 1426 |
| tanh | 5 | Quadratic | 6 | 1.007e-03 | 4.339e-04 | 3.336e-03 | 31 |
| tanh | 5 | Smooth | 6 | 3.122e-04 | 1.535e-04 | 1.533e-03 | 31 |
| tanh | 5 | Strong nonlinear | 5 | 1.250e-03 | 6.930e-04 | 9.183e-03 | 31 |
| tanh | 10 | Quadratic | 6 | 1.636e-03 | 1.227e-03 | 9.355e-03 | 86 |
| tanh | 10 | Smooth | 6 | 4.397e-04 | 3.448e-04 | 4.433e-03 | 86 |
| tanh | 10 | Strong nonlinear | 6 | 6.355e-04 | 5.765e-04 | 6.066e-03 | 86 |
| tanh | 20 | Quadratic | 6 | 2.402e-03 | 3.591e-03 | 2.290e-02 | 271 |
| tanh | 20 | Smooth | 6 | 5.566e-04 | 8.038e-04 | 9.899e-03 | 271 |
| tanh | 20 | Strong nonlinear | 6 | 5.524e-04 | 9.579e-04 | 2.518e-03 | 271 |
| tanh | 50 | Smooth | 6 | 2.304e-03 | 8.359e-03 | 5.817e-02 | 1426 |
| tanh | 50 | Strong nonlinear | 6 | 8.055e-04 | 2.955e-03 | 1.880e-02 | 1426 |

## 总体 gain 摘要

| activation | p | criterion | mean gain | median gain | win rate | reps |
|---|---:|---|---:|---:|---:|---:|
| identity | 5 | D-optimal | 99.06% | 99.76% | 100.0% | 54 |
| identity | 5 | A-optimal | 99.14% | 99.31% | 100.0% | 54 |
| identity | 5 | I-optimal | 92.06% | 94.32% | 100.0% | 54 |
| identity | 5 | Latin hypercube | 97.76% | 98.60% | 100.0% | 54 |
| identity | 10 | D-optimal | 98.56% | 98.73% | 100.0% | 54 |
| identity | 10 | A-optimal | 96.34% | 97.38% | 100.0% | 54 |
| identity | 10 | I-optimal | 73.60% | 78.69% | 100.0% | 54 |
| identity | 10 | Latin hypercube | 85.94% | 89.10% | 100.0% | 54 |
| identity | 20 | D-optimal | 95.45% | 95.44% | 100.0% | 54 |
| identity | 20 | A-optimal | 94.91% | 94.96% | 100.0% | 54 |
| identity | 20 | I-optimal | 80.40% | 82.11% | 100.0% | 54 |
| identity | 20 | Latin hypercube | 51.46% | 54.76% | 98.1% | 54 |
| identity | 50 | D-optimal | 54.56% | 55.62% | 100.0% | 54 |
| identity | 50 | A-optimal | 59.14% | 61.40% | 100.0% | 54 |
| identity | 50 | I-optimal | 52.45% | 54.82% | 100.0% | 54 |
| identity | 50 | Latin hypercube | -19.21% | -5.55% | 38.9% | 54 |
| tanh | 5 | D-optimal | -71.98% | -49.14% | 9.8% | 51 |
| tanh | 5 | A-optimal | -66.18% | -27.71% | 21.6% | 51 |
| tanh | 5 | I-optimal | -124.60% | -94.11% | 3.9% | 51 |
| tanh | 5 | Latin hypercube | 0.74% | 4.30% | 54.9% | 51 |
| tanh | 10 | D-optimal | -19.02% | 5.29% | 55.6% | 54 |
| tanh | 10 | A-optimal | -30.94% | -7.05% | 44.4% | 54 |
| tanh | 10 | I-optimal | -90.33% | -41.82% | 9.3% | 54 |
| tanh | 10 | Latin hypercube | 7.02% | 9.38% | 83.3% | 54 |
| tanh | 20 | D-optimal | 7.27% | 10.27% | 77.8% | 54 |
| tanh | 20 | A-optimal | 5.27% | 9.08% | 75.9% | 54 |
| tanh | 20 | I-optimal | -11.33% | -2.06% | 37.0% | 54 |
| tanh | 20 | Latin hypercube | 3.26% | 5.46% | 75.9% | 54 |
| tanh | 50 | D-optimal | 2.01% | 1.74% | 94.4% | 36 |
| tanh | 50 | A-optimal | 1.70% | 1.45% | 83.3% | 36 |
| tanh | 50 | I-optimal | 1.65% | 1.45% | 80.6% | 36 |
| tanh | 50 | Latin hypercube | -0.40% | -0.47% | 27.8% | 36 |

## 代表预算下的 case-level gain

| activation | p | case | budget x params | D-optimal | A-optimal | I-optimal | Latin hypercube |
|---|---:|---|---:|---:|---:|---:|---:|
| identity | 5 | Quadratic | 3.0 | 99.72% | 99.43% | 96.66% | 99.05% |
| identity | 5 | Smooth | 3.0 | 98.60% | 99.06% | 92.52% | 97.14% |
| identity | 5 | Strong nonlinear | 3.0 | 98.87% | 99.23% | 91.25% | 97.41% |
| identity | 10 | Quadratic | 3.0 | 98.96% | 98.05% | 80.63% | 89.20% |
| identity | 10 | Smooth | 3.0 | 98.42% | 95.20% | 68.42% | 81.17% |
| identity | 10 | Strong nonlinear | 3.0 | 98.72% | 96.68% | 77.33% | 83.14% |
| identity | 20 | Quadratic | 3.0 | 94.90% | 93.70% | 74.45% | 51.45% |
| identity | 20 | Smooth | 3.0 | 95.80% | 95.10% | 76.76% | 53.64% |
| identity | 20 | Strong nonlinear | 3.0 | 95.95% | 95.15% | 78.25% | 55.13% |
| identity | 50 | Quadratic | 3.0 | 60.05% | 63.82% | 57.40% | -3.10% |
| identity | 50 | Smooth | 3.0 | 53.83% | 60.66% | 52.63% | -36.55% |
| identity | 50 | Strong nonlinear | 3.0 | 53.13% | 59.73% | 51.94% | -37.60% |
| tanh | 5 | Quadratic | 3.0 | -28.40% | -8.13% | -32.36% | 4.50% |
| tanh | 5 | Smooth | 3.0 | -88.96% | -14.16% | -121.31% | 10.39% |
| tanh | 5 | Strong nonlinear | 3.0 | -145.22% | -172.29% | -274.11% | -5.54% |
| tanh | 10 | Quadratic | 3.0 | 13.95% | 10.54% | -15.33% | 11.61% |
| tanh | 10 | Smooth | 3.0 | 8.84% | -6.80% | -52.25% | 10.83% |
| tanh | 10 | Strong nonlinear | 3.0 | -84.03% | -104.26% | -206.35% | -21.71% |
| tanh | 20 | Quadratic | 3.0 | 12.00% | 9.57% | -4.00% | 4.61% |
| tanh | 20 | Smooth | 3.0 | 12.36% | 8.43% | -5.75% | 3.47% |
| tanh | 20 | Strong nonlinear | 3.0 | -2.45% | -6.06% | -33.70% | 2.69% |
| tanh | 50 | Smooth | 3.0 | 1.53% | 1.49% | 1.81% | -0.51% |
| tanh | 50 | Strong nonlinear | 3.0 | 2.50% | 2.09% | 2.42% | -0.76% |

## 图

- `figures\close_distance_highdim_optimality\close_distance_highdim_optimality_close_filter.png`
- `figures\close_distance_highdim_optimality\close_distance_highdim_optimality_criterion_gain_by_p_activation.png`
- `figures\close_distance_highdim_optimality\close_distance_highdim_optimality_criteria_heatmap_3x.png`

## 解读

这组实验应和上一轮 D-opt-only 结果配合阅读。tanh 条件测试非线性 NN oracle 下的设计准则稳定性；identity 条件是线性网络对照，用来判断失败是否来自 NN 非线性与 FullPR basis 的不匹配。若 identity 下 D/A/I-optimal 显著更稳定，而 tanh 下不稳定，则说明主要限制来自 oracle 非线性或 surrogate 欠设定；若两者都不稳定，则更可能是候选区域、预算或 optimality 准则本身与目标 MSE 不匹配。