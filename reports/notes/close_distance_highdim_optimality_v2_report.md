# 高维近距离 D/A/I-optimal 迁移与 activation 对照报告

## 实验目的

本实验在高维 close-distance D-optimal 测试基础上加入 A-optimal 和 I-optimal 选点准则，并加入 error-regularized D/A/I 作为一组额外对照。NN 使用双隐藏层，使网络深度与二阶 FullPR 对齐；FullPR 默认只使用二阶多项式项，不再加入逐变量 sin/exp 特征。核心问题是：当 NN oracle 与二阶 FullPR 在同一数据区域内距离较近时，不同实验设计准则是否比同预算随机采样更高效，以及显式惩罚难拟合区域能否改善 classical optimal design 的稳定性。

## 设置

- p values：5, 10, 20。
- activations：softplus, tanh, sigmoid, relu, identity。
- data families：iterative_dopt, measure_weighted, gpu_geometry。
- cases：highdim_poly2, highdim_smooth, highdim_strong, highdim_local, highdim_highfreq, highdim_poly3, highdim_poly4, highdim_nonlinear。
- criteria：D-optimal, A-optimal, I-optimal, D-optimal + error reg., A-optimal + error reg., I-optimal + error reg., Latin hypercube。
- n：10000；训练/评估：7500/2500。
- NN hidden：128,64；epochs=1000。
- FullPR：degree=2，include_special=False。
- optimal selection：sequential_ridge_rank1。
- error regularizer：source=train_abs_nn_minus_full_data_fullpr，strength=1，power=1，cap=10。
- close filter：`shape_nn_fpr <= 5.000e-03`。
- base runs：720；通过 close filter：443；跳过：277。

## 准则含义

- D-optimal：增大信息矩阵行列式，减少参数置信椭球体积。
- A-optimal：减少信息矩阵逆矩阵的 trace，偏向降低平均参数方差。
- I-optimal：减少评估区域的平均预测方差，更直接面向目标区域预测稳定性。
- D/A/I + error reg.：把训练域中 `|NN - full-data FullPR|` 作为难拟合代理，对高误差候选点施加分母惩罚。
- Latin hypercube：几何覆盖基线，不使用 FullPR 信息矩阵。

## 近距离筛选摘要

| activation | family | p | case | used runs | median shape | median FPR mimic MSE | median RMSE gap | design params |
|---|---|---:|---|---:|---:|---:|---:|---:|
| identity | gpu_geometry | 5 | Random nonlinear | 6 | 3.400e-15 | 1.199e-15 | 8.825e-10 | 21 |
| identity | gpu_geometry | 5 | Sparse cubic | 6 | 2.140e-15 | 7.298e-16 | 3.767e-10 | 21 |
| identity | gpu_geometry | 5 | Sparse quartic | 6 | 2.342e-15 | 8.133e-16 | 3.961e-10 | 21 |
| identity | gpu_geometry | 10 | Random nonlinear | 6 | 1.219e-15 | 8.659e-16 | 4.029e-10 | 66 |
| identity | gpu_geometry | 10 | Sparse cubic | 6 | 8.802e-16 | 6.415e-16 | 2.895e-10 | 66 |
| identity | gpu_geometry | 10 | Sparse quartic | 6 | 5.844e-16 | 4.375e-16 | 3.729e-10 | 66 |
| identity | gpu_geometry | 20 | Random nonlinear | 6 | 6.157e-16 | 8.377e-16 | 5.142e-10 | 231 |
| identity | gpu_geometry | 20 | Sparse cubic | 6 | 3.065e-16 | 4.382e-16 | 2.075e-10 | 231 |
| identity | gpu_geometry | 20 | Sparse quartic | 6 | 4.396e-16 | 6.220e-16 | 9.403e-10 | 231 |
| identity | iterative_dopt | 5 | Quadratic | 6 | 1.949e-15 | 7.114e-16 | 1.871e-10 | 21 |
| identity | iterative_dopt | 5 | Smooth | 6 | 2.124e-15 | 8.122e-16 | 8.352e-10 | 21 |
| identity | iterative_dopt | 5 | Strong nonlinear | 6 | 2.080e-15 | 8.455e-16 | 6.320e-10 | 21 |
| identity | iterative_dopt | 10 | Quadratic | 6 | 4.739e-16 | 3.259e-16 | 5.696e-10 | 66 |
| identity | iterative_dopt | 10 | Smooth | 6 | 6.609e-16 | 5.073e-16 | 8.824e-10 | 66 |
| identity | iterative_dopt | 10 | Strong nonlinear | 6 | 7.433e-16 | 5.436e-16 | 6.303e-10 | 66 |
| identity | iterative_dopt | 20 | Quadratic | 6 | 2.003e-16 | 2.915e-16 | 3.862e-10 | 231 |
| identity | iterative_dopt | 20 | Smooth | 6 | 5.209e-16 | 7.464e-16 | 4.020e-10 | 231 |
| identity | iterative_dopt | 20 | Strong nonlinear | 6 | 5.436e-16 | 7.995e-16 | 3.023e-10 | 231 |
| identity | measure_weighted | 5 | High-frequency | 6 | 1.392e-15 | 4.922e-16 | 3.165e-10 | 21 |
| identity | measure_weighted | 5 | Local interior | 6 | 4.805e-15 | 1.598e-15 | 5.797e-10 | 21 |
| identity | measure_weighted | 10 | High-frequency | 6 | 3.434e-16 | 2.434e-16 | 1.261e-10 | 66 |
| identity | measure_weighted | 10 | Local interior | 6 | 1.374e-15 | 9.599e-16 | 6.874e-10 | 66 |
| identity | measure_weighted | 20 | High-frequency | 6 | 1.786e-16 | 2.450e-16 | 6.820e-10 | 231 |
| identity | measure_weighted | 20 | Local interior | 6 | 6.273e-16 | 9.134e-16 | 5.028e-10 | 231 |
| relu | gpu_geometry | 5 | Sparse cubic | 6 | 2.197e-03 | 7.885e-04 | 2.186e-02 | 21 |
| relu | gpu_geometry | 5 | Sparse quartic | 6 | 3.548e-03 | 1.327e-03 | 2.973e-02 | 21 |
| relu | gpu_geometry | 10 | Random nonlinear | 2 | 3.673e-03 | 2.675e-03 | 4.113e-02 | 66 |
| relu | gpu_geometry | 10 | Sparse cubic | 6 | 8.090e-04 | 6.331e-04 | 1.509e-02 | 66 |
| relu | gpu_geometry | 10 | Sparse quartic | 6 | 1.203e-03 | 9.295e-04 | 2.013e-02 | 66 |
| relu | gpu_geometry | 20 | Random nonlinear | 4 | 3.266e-03 | 4.637e-03 | 4.286e-02 | 231 |
| relu | gpu_geometry | 20 | Sparse cubic | 6 | 6.758e-04 | 9.640e-04 | 1.466e-02 | 231 |
| relu | gpu_geometry | 20 | Sparse quartic | 6 | 7.792e-04 | 1.054e-03 | 1.594e-02 | 231 |
| relu | iterative_dopt | 5 | Quadratic | 5 | 1.486e-03 | 6.729e-04 | 6.312e-03 | 21 |
| relu | iterative_dopt | 5 | Smooth | 2 | 2.795e-03 | 1.029e-03 | 9.042e-03 | 21 |
| relu | iterative_dopt | 10 | Quadratic | 6 | 2.465e-03 | 1.860e-03 | 1.253e-02 | 66 |
| relu | iterative_dopt | 10 | Smooth | 6 | 2.821e-03 | 2.157e-03 | 1.328e-02 | 66 |
| relu | iterative_dopt | 20 | Quadratic | 6 | 2.376e-03 | 3.499e-03 | 2.275e-02 | 231 |
| relu | iterative_dopt | 20 | Smooth | 6 | 1.603e-03 | 2.448e-03 | 8.732e-03 | 231 |
| relu | iterative_dopt | 20 | Strong nonlinear | 1 | 3.994e-03 | 8.360e-03 | 5.304e-02 | 231 |
| sigmoid | gpu_geometry | 5 | Sparse cubic | 6 | 1.931e-03 | 7.699e-04 | 2.284e-02 | 21 |
| sigmoid | gpu_geometry | 5 | Sparse quartic | 6 | 3.564e-03 | 1.291e-03 | 2.949e-02 | 21 |
| sigmoid | gpu_geometry | 10 | Random nonlinear | 2 | 4.226e-03 | 2.749e-03 | 4.004e-02 | 66 |
| sigmoid | gpu_geometry | 10 | Sparse cubic | 6 | 7.954e-04 | 6.166e-04 | 1.680e-02 | 66 |
| sigmoid | gpu_geometry | 10 | Sparse quartic | 6 | 1.249e-03 | 9.673e-04 | 1.988e-02 | 66 |
| sigmoid | gpu_geometry | 20 | Random nonlinear | 3 | 1.712e-03 | 2.405e-03 | 3.954e-02 | 231 |
| sigmoid | gpu_geometry | 20 | Sparse cubic | 6 | 6.436e-04 | 9.514e-04 | 1.397e-02 | 231 |
| sigmoid | gpu_geometry | 20 | Sparse quartic | 6 | 7.576e-04 | 1.073e-03 | 1.831e-02 | 231 |
| sigmoid | iterative_dopt | 5 | Quadratic | 6 | 7.880e-04 | 2.962e-04 | 2.849e-03 | 21 |
| sigmoid | iterative_dopt | 5 | Smooth | 4 | 3.508e-03 | 1.475e-03 | 1.579e-02 | 21 |
| sigmoid | iterative_dopt | 10 | Quadratic | 6 | 1.755e-03 | 1.280e-03 | 1.076e-02 | 66 |
| sigmoid | iterative_dopt | 10 | Smooth | 6 | 2.909e-03 | 2.186e-03 | 1.476e-02 | 66 |
| sigmoid | iterative_dopt | 20 | Quadratic | 4 | 4.139e-03 | 6.367e-03 | 3.857e-02 | 231 |
| sigmoid | iterative_dopt | 20 | Smooth | 6 | 2.292e-03 | 3.152e-03 | 4.092e-03 | 231 |
| softplus | gpu_geometry | 5 | Sparse cubic | 6 | 1.989e-03 | 7.758e-04 | 2.309e-02 | 21 |
| softplus | gpu_geometry | 5 | Sparse quartic | 6 | 3.962e-03 | 1.354e-03 | 3.185e-02 | 21 |
| softplus | gpu_geometry | 10 | Random nonlinear | 2 | 3.852e-03 | 2.700e-03 | 4.421e-02 | 66 |
| softplus | gpu_geometry | 10 | Sparse cubic | 6 | 8.663e-04 | 6.382e-04 | 1.968e-02 | 66 |
| softplus | gpu_geometry | 10 | Sparse quartic | 6 | 1.229e-03 | 9.141e-04 | 2.362e-02 | 66 |
| softplus | gpu_geometry | 20 | Random nonlinear | 4 | 3.283e-03 | 4.628e-03 | 5.524e-02 | 231 |
| softplus | gpu_geometry | 20 | Sparse cubic | 6 | 6.422e-04 | 9.213e-04 | 2.188e-02 | 231 |
| softplus | gpu_geometry | 20 | Sparse quartic | 6 | 7.173e-04 | 1.034e-03 | 2.453e-02 | 231 |
| softplus | iterative_dopt | 5 | Quadratic | 6 | 1.043e-03 | 3.561e-04 | 2.931e-03 | 21 |
| softplus | iterative_dopt | 5 | Smooth | 3 | 2.179e-03 | 8.463e-04 | 1.155e-02 | 21 |
| softplus | iterative_dopt | 10 | Quadratic | 6 | 1.404e-03 | 9.769e-04 | 6.808e-03 | 66 |
| softplus | iterative_dopt | 10 | Smooth | 6 | 2.566e-03 | 1.885e-03 | 1.789e-02 | 66 |
| softplus | iterative_dopt | 20 | Quadratic | 6 | 1.872e-03 | 2.700e-03 | 1.836e-02 | 231 |
| softplus | iterative_dopt | 20 | Smooth | 6 | 1.570e-03 | 2.372e-03 | 1.275e-02 | 231 |
| softplus | iterative_dopt | 20 | Strong nonlinear | 1 | 4.742e-03 | 8.350e-03 | 6.081e-02 | 231 |
| tanh | gpu_geometry | 5 | Sparse cubic | 6 | 2.001e-03 | 7.734e-04 | 2.239e-02 | 21 |
| tanh | gpu_geometry | 5 | Sparse quartic | 6 | 3.491e-03 | 1.305e-03 | 2.975e-02 | 21 |
| tanh | gpu_geometry | 10 | Random nonlinear | 2 | 3.655e-03 | 2.672e-03 | 4.206e-02 | 66 |
| tanh | gpu_geometry | 10 | Sparse cubic | 6 | 9.251e-04 | 6.597e-04 | 1.799e-02 | 66 |
| tanh | gpu_geometry | 10 | Sparse quartic | 6 | 1.191e-03 | 9.246e-04 | 2.279e-02 | 66 |
| tanh | gpu_geometry | 20 | Random nonlinear | 4 | 3.225e-03 | 4.582e-03 | 5.493e-02 | 231 |
| tanh | gpu_geometry | 20 | Sparse cubic | 6 | 6.304e-04 | 8.880e-04 | 1.628e-02 | 231 |
| tanh | gpu_geometry | 20 | Sparse quartic | 6 | 7.076e-04 | 1.051e-03 | 2.150e-02 | 231 |
| tanh | iterative_dopt | 5 | Quadratic | 6 | 7.504e-04 | 2.968e-04 | 2.985e-03 | 21 |
| tanh | iterative_dopt | 5 | Smooth | 4 | 3.511e-03 | 1.465e-03 | 1.631e-02 | 21 |
| tanh | iterative_dopt | 10 | Quadratic | 6 | 1.950e-03 | 1.445e-03 | 1.019e-02 | 66 |
| tanh | iterative_dopt | 10 | Smooth | 6 | 2.547e-03 | 2.073e-03 | 1.587e-02 | 66 |
| tanh | iterative_dopt | 20 | Quadratic | 6 | 2.830e-03 | 4.368e-03 | 2.703e-02 | 231 |
| tanh | iterative_dopt | 20 | Smooth | 6 | 1.765e-03 | 2.597e-03 | 9.067e-03 | 231 |

## 总体 gain 摘要

| activation | p | criterion | mean gain | median gain | win rate | reps |
|---|---:|---|---:|---:|---:|---:|
| identity | 5 | D-optimal | 74.31% | 82.22% | 99.3% | 144 |
| identity | 5 | A-optimal | 76.92% | 82.54% | 100.0% | 144 |
| identity | 5 | I-optimal | 76.06% | 81.73% | 100.0% | 144 |
| identity | 5 | D-optimal + error reg. | 78.43% | 83.56% | 100.0% | 144 |
| identity | 5 | A-optimal + error reg. | 78.18% | 82.59% | 100.0% | 144 |
| identity | 5 | I-optimal + error reg. | 77.62% | 82.44% | 100.0% | 144 |
| identity | 5 | Latin hypercube | 70.54% | 76.60% | 99.3% | 144 |
| identity | 10 | D-optimal | 50.61% | 51.01% | 97.2% | 144 |
| identity | 10 | A-optimal | 55.39% | 53.72% | 100.0% | 144 |
| identity | 10 | I-optimal | 52.20% | 50.56% | 100.0% | 144 |
| identity | 10 | D-optimal + error reg. | 58.96% | 58.06% | 100.0% | 144 |
| identity | 10 | A-optimal + error reg. | 58.77% | 58.44% | 100.0% | 144 |
| identity | 10 | I-optimal + error reg. | 56.80% | 56.77% | 100.0% | 144 |
| identity | 10 | Latin hypercube | 46.72% | 44.57% | 100.0% | 144 |
| identity | 20 | D-optimal | 24.21% | 21.29% | 98.6% | 144 |
| identity | 20 | A-optimal | 26.40% | 23.03% | 100.0% | 144 |
| identity | 20 | I-optimal | 21.70% | 20.34% | 100.0% | 144 |
| identity | 20 | D-optimal + error reg. | 37.35% | 33.21% | 100.0% | 144 |
| identity | 20 | A-optimal + error reg. | 39.16% | 36.04% | 100.0% | 144 |
| identity | 20 | I-optimal + error reg. | 38.06% | 35.36% | 100.0% | 144 |
| identity | 20 | Latin hypercube | 14.59% | 12.37% | 100.0% | 144 |
| relu | 5 | D-optimal | -101.82% | -126.15% | 22.8% | 57 |
| relu | 5 | A-optimal | -76.04% | -72.43% | 21.1% | 57 |
| relu | 5 | I-optimal | -66.46% | -62.01% | 22.8% | 57 |
| relu | 5 | D-optimal + error reg. | 18.94% | 37.38% | 93.0% | 57 |
| relu | 5 | A-optimal + error reg. | 43.16% | 41.44% | 100.0% | 57 |
| relu | 5 | I-optimal + error reg. | 43.26% | 41.52% | 100.0% | 57 |
| relu | 5 | Latin hypercube | -73.65% | -52.63% | 10.5% | 57 |
| relu | 10 | D-optimal | -34.82% | -14.23% | 26.9% | 78 |
| relu | 10 | A-optimal | -11.31% | -12.01% | 33.3% | 78 |
| relu | 10 | I-optimal | -5.14% | -5.94% | 38.5% | 78 |
| relu | 10 | D-optimal + error reg. | 38.05% | 36.70% | 100.0% | 78 |
| relu | 10 | A-optimal + error reg. | 40.64% | 38.95% | 100.0% | 78 |
| relu | 10 | I-optimal + error reg. | 40.73% | 39.36% | 100.0% | 78 |
| relu | 10 | Latin hypercube | 0.85% | 2.64% | 60.3% | 78 |
| relu | 20 | D-optimal | -7.25% | -5.61% | 40.2% | 87 |
| relu | 20 | A-optimal | 3.11% | 1.59% | 56.3% | 87 |
| relu | 20 | I-optimal | 1.13% | 0.91% | 55.2% | 87 |
| relu | 20 | D-optimal + error reg. | 31.91% | 29.14% | 100.0% | 87 |
| relu | 20 | A-optimal + error reg. | 34.70% | 32.26% | 100.0% | 87 |
| relu | 20 | I-optimal + error reg. | 34.81% | 32.39% | 100.0% | 87 |
| relu | 20 | Latin hypercube | 3.89% | 3.70% | 73.6% | 87 |
| sigmoid | 5 | D-optimal | -99.21% | -110.96% | 22.7% | 66 |
| sigmoid | 5 | A-optimal | -75.45% | -73.81% | 13.6% | 66 |
| sigmoid | 5 | I-optimal | -45.74% | -44.74% | 21.2% | 66 |
| sigmoid | 5 | D-optimal + error reg. | 22.89% | 37.67% | 87.9% | 66 |
| sigmoid | 5 | A-optimal + error reg. | 43.47% | 41.10% | 100.0% | 66 |
| sigmoid | 5 | I-optimal + error reg. | 43.50% | 41.33% | 100.0% | 66 |
| sigmoid | 5 | Latin hypercube | -57.73% | -45.75% | 24.2% | 66 |
| sigmoid | 10 | D-optimal | -33.14% | -13.95% | 29.5% | 78 |
| sigmoid | 10 | A-optimal | -9.97% | -9.78% | 37.2% | 78 |
| sigmoid | 10 | I-optimal | -8.05% | -5.08% | 39.7% | 78 |
| sigmoid | 10 | D-optimal + error reg. | 36.73% | 35.41% | 100.0% | 78 |
| sigmoid | 10 | A-optimal + error reg. | 39.26% | 38.91% | 100.0% | 78 |
| sigmoid | 10 | I-optimal + error reg. | 39.42% | 38.76% | 100.0% | 78 |
| sigmoid | 10 | Latin hypercube | 4.26% | 5.59% | 67.9% | 78 |
| sigmoid | 20 | D-optimal | -7.93% | -7.92% | 32.0% | 75 |
| sigmoid | 20 | A-optimal | 2.15% | -0.02% | 49.3% | 75 |
| sigmoid | 20 | I-optimal | 0.12% | -2.07% | 41.3% | 75 |
| sigmoid | 20 | D-optimal + error reg. | 29.91% | 28.54% | 100.0% | 75 |
| sigmoid | 20 | A-optimal + error reg. | 32.56% | 31.00% | 100.0% | 75 |
| sigmoid | 20 | I-optimal + error reg. | 32.58% | 31.68% | 100.0% | 75 |
| sigmoid | 20 | Latin hypercube | 4.57% | 4.93% | 74.7% | 75 |
| softplus | 5 | D-optimal | -97.64% | -103.47% | 11.1% | 63 |
| softplus | 5 | A-optimal | -71.86% | -62.73% | 12.7% | 63 |
| softplus | 5 | I-optimal | -51.27% | -41.30% | 17.5% | 63 |
| softplus | 5 | D-optimal + error reg. | 15.59% | 37.46% | 90.5% | 63 |
| softplus | 5 | A-optimal + error reg. | 42.44% | 42.65% | 100.0% | 63 |
| softplus | 5 | I-optimal + error reg. | 42.49% | 42.91% | 100.0% | 63 |
| softplus | 5 | Latin hypercube | -44.10% | -36.63% | 9.5% | 63 |
| softplus | 10 | D-optimal | -34.51% | -15.12% | 30.8% | 78 |
| softplus | 10 | A-optimal | -11.79% | -11.46% | 35.9% | 78 |
| softplus | 10 | I-optimal | -7.80% | -4.66% | 38.5% | 78 |
| softplus | 10 | D-optimal + error reg. | 37.30% | 35.60% | 100.0% | 78 |
| softplus | 10 | A-optimal + error reg. | 40.14% | 39.11% | 100.0% | 78 |
| softplus | 10 | I-optimal + error reg. | 40.22% | 39.02% | 100.0% | 78 |
| softplus | 10 | Latin hypercube | -3.72% | -3.88% | 41.0% | 78 |
| softplus | 20 | D-optimal | -8.19% | -6.66% | 37.9% | 87 |
| softplus | 20 | A-optimal | 2.40% | 2.50% | 57.5% | 87 |
| softplus | 20 | I-optimal | -0.17% | 0.69% | 51.7% | 87 |
| softplus | 20 | D-optimal + error reg. | 31.68% | 29.73% | 100.0% | 87 |
| softplus | 20 | A-optimal + error reg. | 34.78% | 33.00% | 100.0% | 87 |
| softplus | 20 | I-optimal + error reg. | 34.77% | 32.99% | 100.0% | 87 |
| softplus | 20 | Latin hypercube | 3.58% | 3.66% | 70.1% | 87 |
| tanh | 5 | D-optimal | -100.96% | -113.33% | 22.7% | 66 |
| tanh | 5 | A-optimal | -75.03% | -72.12% | 18.2% | 66 |
| tanh | 5 | I-optimal | -48.67% | -45.20% | 21.2% | 66 |
| tanh | 5 | D-optimal + error reg. | 18.36% | 38.35% | 90.9% | 66 |
| tanh | 5 | A-optimal + error reg. | 43.47% | 41.93% | 100.0% | 66 |
| tanh | 5 | I-optimal + error reg. | 43.48% | 42.11% | 100.0% | 66 |
| tanh | 5 | Latin hypercube | -44.94% | -31.73% | 16.7% | 66 |
| tanh | 10 | D-optimal | -32.44% | -14.30% | 34.6% | 78 |
| tanh | 10 | A-optimal | -11.09% | -9.93% | 37.2% | 78 |
| tanh | 10 | I-optimal | -9.64% | -4.71% | 41.0% | 78 |
| tanh | 10 | D-optimal + error reg. | 37.69% | 36.12% | 100.0% | 78 |
| tanh | 10 | A-optimal + error reg. | 39.86% | 38.41% | 100.0% | 78 |
| tanh | 10 | I-optimal + error reg. | 39.93% | 38.38% | 100.0% | 78 |
| tanh | 10 | Latin hypercube | -2.53% | 0.44% | 51.3% | 78 |
| tanh | 20 | D-optimal | -7.31% | -4.99% | 39.3% | 84 |
| tanh | 20 | A-optimal | 2.94% | 2.57% | 54.8% | 84 |
| tanh | 20 | I-optimal | 1.56% | 1.79% | 54.8% | 84 |
| tanh | 20 | D-optimal + error reg. | 30.77% | 29.90% | 100.0% | 84 |
| tanh | 20 | A-optimal + error reg. | 33.61% | 32.45% | 100.0% | 84 |
| tanh | 20 | I-optimal + error reg. | 33.73% | 32.21% | 100.0% | 84 |
| tanh | 20 | Latin hypercube | 2.70% | 4.42% | 70.2% | 84 |

## 代表预算下的 case-level gain

| activation | p | case | budget x params | D-optimal | A-optimal | I-optimal | D-optimal + error reg. | A-optimal + error reg. | I-optimal + error reg. | Latin hypercube |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 5 | High-frequency | 3.0 | 81.49% | 82.73% | 80.87% | 82.63% | 82.52% | 82.70% | 72.29% |
| identity | 5 | Local interior | 3.0 | 33.51% | 38.92% | 38.87% | 44.21% | 45.33% | 45.21% | 25.53% |
| identity | 5 | Random nonlinear | 3.0 | 57.18% | 65.55% | 66.12% | 71.10% | 70.88% | 70.58% | 60.46% |
| identity | 5 | Quadratic | 3.0 | 81.71% | 85.74% | 84.84% | 85.39% | 86.23% | 84.87% | 84.64% |
| identity | 5 | Sparse cubic | 3.0 | 78.00% | 80.32% | 79.89% | 81.00% | 80.85% | 80.72% | 75.96% |
| identity | 5 | Sparse quartic | 3.0 | 78.08% | 77.93% | 76.99% | 79.26% | 79.58% | 79.31% | 74.69% |
| identity | 5 | Smooth | 3.0 | 92.09% | 92.54% | 91.12% | 92.75% | 90.63% | 90.14% | 82.11% |
| identity | 5 | Strong nonlinear | 3.0 | 90.67% | 90.37% | 89.58% | 91.43% | 90.90% | 90.33% | 87.78% |
| identity | 10 | High-frequency | 3.0 | 51.40% | 54.37% | 50.61% | 58.67% | 58.95% | 57.66% | 46.11% |
| identity | 10 | Local interior | 3.0 | 24.58% | 32.07% | 30.82% | 38.77% | 40.16% | 40.13% | 24.46% |
| identity | 10 | Random nonlinear | 3.0 | 35.41% | 37.11% | 35.08% | 42.51% | 45.14% | 45.06% | 28.86% |
| identity | 10 | Quadratic | 3.0 | 58.50% | 61.13% | 57.19% | 63.60% | 64.51% | 62.35% | 54.47% |
| identity | 10 | Sparse cubic | 3.0 | 43.08% | 51.44% | 52.10% | 56.91% | 57.43% | 56.14% | 45.36% |
| identity | 10 | Sparse quartic | 3.0 | 37.85% | 43.70% | 40.60% | 46.97% | 49.80% | 48.01% | 33.98% |
| identity | 10 | Smooth | 3.0 | 60.00% | 69.24% | 66.92% | 71.96% | 70.49% | 67.35% | 63.15% |
| identity | 10 | Strong nonlinear | 3.0 | 63.63% | 69.17% | 65.84% | 70.49% | 69.97% | 67.21% | 61.15% |
| identity | 20 | High-frequency | 3.0 | 22.40% | 24.47% | 20.70% | 33.95% | 36.12% | 35.84% | 10.95% |
| identity | 20 | Local interior | 3.0 | 22.37% | 23.29% | 20.36% | 32.22% | 35.50% | 35.41% | 12.69% |
| identity | 20 | Random nonlinear | 3.0 | 20.39% | 22.04% | 17.47% | 31.77% | 34.95% | 34.78% | 12.11% |
| identity | 20 | Quadratic | 3.0 | 17.71% | 20.89% | 19.50% | 33.02% | 35.73% | 34.82% | 9.99% |
| identity | 20 | Sparse cubic | 3.0 | 20.71% | 22.21% | 19.94% | 34.00% | 36.04% | 35.34% | 10.04% |
| identity | 20 | Sparse quartic | 3.0 | 21.49% | 22.68% | 20.69% | 33.65% | 36.28% | 35.92% | 12.25% |
| identity | 20 | Smooth | 3.0 | 19.87% | 24.92% | 20.84% | 36.59% | 38.48% | 37.40% | 13.35% |
| identity | 20 | Strong nonlinear | 3.0 | 22.03% | 21.49% | 20.68% | 34.25% | 36.56% | 35.75% | 15.11% |
| relu | 5 | Quadratic | 3.0 | 14.98% | 9.47% | 13.12% | 32.64% | 35.24% | 35.52% | -13.16% |
| relu | 5 | Sparse cubic | 3.0 | -135.23% | -67.67% | -71.31% | 40.07% | 43.16% | 43.17% | -67.15% |
| relu | 5 | Sparse quartic | 3.0 | -164.86% | -120.51% | -118.74% | 39.00% | 43.05% | 43.15% | -61.22% |
| relu | 5 | Smooth | 3.0 | -179.79% | -196.18% | -122.97% | -251.45% | 41.14% | 41.31% | -86.13% |
| relu | 10 | Random nonlinear | 3.0 | 4.47% | 1.58% | 3.37% | 37.03% | 39.41% | 39.79% | 6.07% |
| relu | 10 | Quadratic | 3.0 | 7.25% | 19.07% | 17.39% | 30.15% | 32.38% | 32.77% | 7.86% |
| relu | 10 | Sparse cubic | 3.0 | -72.71% | -28.31% | -17.27% | 37.61% | 40.43% | 40.50% | -1.10% |
| relu | 10 | Sparse quartic | 3.0 | -3.07% | -14.40% | -6.13% | 37.19% | 40.78% | 41.04% | -0.20% |
| relu | 10 | Smooth | 3.0 | -57.44% | -35.82% | -26.93% | 38.59% | 39.38% | 39.17% | 11.67% |
| relu | 20 | Random nonlinear | 3.0 | 8.26% | 10.98% | 9.70% | 27.16% | 30.45% | 30.87% | 8.45% |
| relu | 20 | Quadratic | 3.0 | 13.01% | 12.33% | 12.48% | 24.02% | 26.80% | 26.89% | 8.48% |
| relu | 20 | Sparse cubic | 3.0 | -6.17% | -1.83% | -3.88% | 31.69% | 34.31% | 34.45% | -0.98% |
| relu | 20 | Sparse quartic | 3.0 | -31.74% | -1.44% | -5.10% | 32.18% | 35.93% | 35.82% | -1.81% |
| relu | 20 | Smooth | 3.0 | -11.02% | -4.84% | -4.22% | 29.53% | 31.17% | 31.50% | 4.64% |
| relu | 20 | Strong nonlinear | 3.0 | -25.94% | -1.40% | -39.17% | 31.47% | 34.39% | 34.19% | -1.57% |
| sigmoid | 5 | Quadratic | 3.0 | 8.80% | 1.49% | 6.32% | 33.24% | 38.59% | 38.77% | -27.54% |
| sigmoid | 5 | Sparse cubic | 3.0 | -114.53% | -59.13% | -54.39% | 39.34% | 42.57% | 42.75% | -32.49% |
| sigmoid | 5 | Sparse quartic | 3.0 | -157.77% | -115.08% | -81.01% | 41.40% | 43.38% | 43.23% | -93.48% |
| sigmoid | 5 | Smooth | 3.0 | -218.67% | -160.73% | -100.48% | -91.07% | 42.35% | 42.18% | -79.89% |
| sigmoid | 10 | Random nonlinear | 3.0 | 6.52% | 2.59% | 5.86% | 37.32% | 38.95% | 39.37% | 11.16% |
| sigmoid | 10 | Quadratic | 3.0 | 2.71% | 17.96% | 12.24% | 28.55% | 30.18% | 30.44% | 6.12% |
| sigmoid | 10 | Sparse cubic | 3.0 | -61.59% | -20.48% | -19.02% | 35.92% | 39.00% | 39.29% | -3.74% |
| sigmoid | 10 | Sparse quartic | 3.0 | -1.53% | -11.32% | -8.35% | 35.16% | 39.95% | 40.23% | -4.05% |
| sigmoid | 10 | Smooth | 3.0 | -65.71% | -40.77% | -26.83% | 37.13% | 38.26% | 38.34% | 1.95% |
| sigmoid | 20 | Random nonlinear | 3.0 | 4.31% | 9.04% | 6.96% | 27.83% | 31.96% | 31.55% | 7.69% |
| sigmoid | 20 | Quadratic | 3.0 | 11.20% | 11.57% | 12.85% | 17.06% | 18.53% | 18.68% | 5.71% |
| sigmoid | 20 | Sparse cubic | 3.0 | -5.49% | -1.97% | -4.88% | 31.26% | 34.28% | 34.25% | 2.81% |
| sigmoid | 20 | Sparse quartic | 3.0 | -27.61% | -0.73% | -6.70% | 30.29% | 33.51% | 33.64% | -0.01% |
| sigmoid | 20 | Smooth | 3.0 | -9.17% | -3.65% | -0.76% | 27.28% | 28.56% | 28.45% | 2.36% |
| softplus | 5 | Quadratic | 3.0 | -1.12% | -1.61% | 0.76% | 26.90% | 34.65% | 34.51% | -55.88% |
| softplus | 5 | Sparse cubic | 3.0 | -108.91% | -53.82% | -43.09% | 40.21% | 43.99% | 44.23% | -40.78% |
| softplus | 5 | Sparse quartic | 3.0 | -149.23% | -107.58% | -78.34% | 39.14% | 42.59% | 42.44% | -62.63% |
| softplus | 5 | Smooth | 3.0 | -231.20% | -180.59% | -138.58% | -166.85% | 42.86% | 42.90% | -17.90% |
| softplus | 10 | Random nonlinear | 3.0 | 4.31% | 0.81% | 15.45% | 35.38% | 39.32% | 39.11% | 11.56% |
| softplus | 10 | Quadratic | 3.0 | -4.97% | 14.11% | 11.04% | 28.48% | 31.14% | 31.27% | -0.05% |
| softplus | 10 | Sparse cubic | 3.0 | -56.79% | -22.11% | -22.03% | 36.03% | 39.41% | 39.62% | -17.77% |
| softplus | 10 | Sparse quartic | 3.0 | -0.25% | -9.75% | -9.29% | 37.56% | 41.32% | 41.21% | 3.05% |
| softplus | 10 | Smooth | 3.0 | -68.18% | -43.10% | -27.01% | 37.88% | 39.66% | 39.81% | -2.80% |
| softplus | 20 | Random nonlinear | 3.0 | 8.22% | 11.20% | 10.07% | 26.83% | 30.34% | 30.69% | 7.90% |
| softplus | 20 | Quadratic | 3.0 | 8.99% | 10.50% | 7.71% | 22.26% | 24.75% | 24.72% | 3.95% |
| softplus | 20 | Sparse cubic | 3.0 | -5.84% | -1.69% | -5.01% | 32.13% | 35.43% | 35.37% | 1.06% |
| softplus | 20 | Sparse quartic | 3.0 | -26.98% | 0.68% | -6.06% | 31.91% | 35.69% | 35.64% | 4.59% |
| softplus | 20 | Smooth | 3.0 | -17.41% | -8.18% | -7.84% | 30.09% | 32.42% | 32.30% | 5.06% |
| softplus | 20 | Strong nonlinear | 3.0 | -25.63% | -1.21% | 1.63% | 30.58% | 32.71% | 33.88% | 15.33% |
| tanh | 5 | Quadratic | 3.0 | 15.41% | 2.23% | 9.37% | 34.46% | 37.22% | 36.92% | -5.43% |
| tanh | 5 | Sparse cubic | 3.0 | -116.78% | -58.27% | -52.41% | 40.58% | 43.64% | 43.56% | -62.12% |
| tanh | 5 | Sparse quartic | 3.0 | -162.54% | -111.94% | -75.77% | 39.29% | 43.33% | 43.35% | -67.35% |
| tanh | 5 | Smooth | 3.0 | -230.53% | -165.09% | -126.79% | -80.52% | 41.98% | 42.30% | -12.64% |
| tanh | 10 | Random nonlinear | 3.0 | 2.44% | -1.36% | 16.75% | 36.35% | 38.57% | 38.55% | 8.94% |
| tanh | 10 | Quadratic | 3.0 | 15.10% | 19.12% | 16.73% | 30.07% | 31.43% | 31.49% | 9.56% |
| tanh | 10 | Sparse cubic | 3.0 | -63.11% | -23.28% | -21.81% | 37.49% | 39.87% | 39.96% | -6.15% |
| tanh | 10 | Sparse quartic | 3.0 | -1.96% | -9.12% | -5.68% | 37.42% | 40.83% | 41.00% | -17.96% |
| tanh | 10 | Smooth | 3.0 | -67.28% | -43.44% | -37.90% | 37.03% | 38.68% | 38.83% | 2.14% |
| tanh | 20 | Random nonlinear | 3.0 | 7.94% | 11.25% | 7.66% | 27.43% | 30.68% | 31.04% | 11.36% |
| tanh | 20 | Quadratic | 3.0 | 11.10% | 12.05% | 11.13% | 20.71% | 22.32% | 22.95% | 7.02% |
| tanh | 20 | Sparse cubic | 3.0 | -5.42% | -1.86% | -6.37% | 32.09% | 34.97% | 35.26% | 4.84% |
| tanh | 20 | Sparse quartic | 3.0 | -31.99% | -1.94% | -6.98% | 31.76% | 35.10% | 35.10% | -3.05% |
| tanh | 20 | Smooth | 3.0 | -13.53% | -5.03% | -2.20% | 28.71% | 31.30% | 31.43% | 2.42% |

## 图

- `figures\close_distance_highdim_optimality\close_distance_highdim_optimality_v2_close_filter.png`
- `figures\close_distance_highdim_optimality\close_distance_highdim_optimality_v2_criterion_gain_by_p_activation.png`
- `figures\close_distance_highdim_optimality\close_distance_highdim_optimality_v2_criteria_heatmap_3x.png`

## 解读

这组实验应和上一轮 D-opt-only 结果配合阅读。tanh 条件测试非线性 NN oracle 下的设计准则稳定性；identity 条件是线性网络对照，用来判断失败是否来自 NN 非线性与 FullPR basis 的不匹配。若 identity 下 D/A/I-optimal 显著更稳定，而 tanh 下不稳定，则说明主要限制来自 oracle 非线性或 surrogate 欠设定；若两者都不稳定，则更可能是候选区域、预算或 optimality 准则本身与目标 MSE 不匹配。