# 高维近距离 D/A/I-optimal 迁移与 activation 对照报告

## 实验目的

本实验在高维 close-distance D-optimal 测试基础上加入 A-optimal 和 I-optimal 选点准则，并把 NN activation 从默认 tanh 扩展到 identity 线性对照。核心问题是：当 NN oracle 与二阶 FullPR 在同一数据区域内距离较近时，不同实验设计准则是否比同预算随机采样更高效，以及这种优势是否依赖 NN 的非线性激活。

## 设置

- p values：5, 10, 20, 50。
- activations：softplus, tanh, sigmoid, relu, identity。
- data families：iterative_dopt, measure_weighted, gpu_geometry。
- cases：highdim_poly2, highdim_smooth, highdim_strong, highdim_local, highdim_highfreq, highdim_poly3, highdim_poly4, highdim_nonlinear。
- criteria：D-optimal, A-optimal, I-optimal, Latin hypercube。
- n：10000；训练/评估：7500/2500。
- NN hidden：128,64,32；epochs=1000。
- FullPR：degree=2，include_special=True。
- close filter：`shape_nn_fpr <= 5.000e-03`。
- base runs：960；通过 close filter：694；跳过：266。

## 准则含义

- D-optimal：增大信息矩阵行列式，减少参数置信椭球体积。
- A-optimal：减少信息矩阵逆矩阵的 trace，偏向降低平均参数方差。
- I-optimal：减少评估区域的平均预测方差，更直接面向目标区域预测稳定性。
- Latin hypercube：几何覆盖基线，不使用 FullPR 信息矩阵。

## 近距离筛选摘要

| activation | family | p | case | used runs | median shape | median FPR mimic MSE | median RMSE gap | design params |
|---|---|---:|---|---:|---:|---:|---:|---:|
| identity | gpu_geometry | 5 | Random nonlinear | 6 | 1.792e-14 | 6.544e-15 | 1.399e-09 | 31 |
| identity | gpu_geometry | 5 | Sparse cubic | 6 | 2.744e-14 | 9.166e-15 | 4.957e-09 | 31 |
| identity | gpu_geometry | 5 | Sparse quartic | 6 | 2.931e-14 | 8.753e-15 | 5.024e-09 | 31 |
| identity | gpu_geometry | 10 | Random nonlinear | 6 | 6.637e-15 | 4.710e-15 | 2.150e-09 | 86 |
| identity | gpu_geometry | 10 | Sparse cubic | 6 | 4.029e-14 | 2.795e-14 | 5.667e-09 | 86 |
| identity | gpu_geometry | 10 | Sparse quartic | 6 | 2.952e-14 | 1.490e-14 | 7.402e-09 | 86 |
| identity | gpu_geometry | 20 | Random nonlinear | 6 | 4.204e-15 | 5.871e-15 | 1.510e-09 | 271 |
| identity | gpu_geometry | 20 | Sparse cubic | 6 | 1.019e-14 | 1.938e-14 | 1.052e-08 | 271 |
| identity | gpu_geometry | 20 | Sparse quartic | 6 | 7.188e-15 | 1.035e-14 | 5.954e-09 | 271 |
| identity | gpu_geometry | 50 | Random nonlinear | 6 | 1.753e-15 | 9.370e-15 | 2.990e-09 | 1426 |
| identity | gpu_geometry | 50 | Sparse cubic | 6 | 9.440e-15 | 3.493e-14 | 3.616e-09 | 1426 |
| identity | gpu_geometry | 50 | Sparse quartic | 6 | 4.270e-15 | 1.209e-14 | 1.479e-08 | 1426 |
| identity | iterative_dopt | 5 | Quadratic | 6 | 3.303e-14 | 1.320e-14 | 2.944e-09 | 31 |
| identity | iterative_dopt | 5 | Smooth | 6 | 9.106e-14 | 3.869e-14 | 1.799e-08 | 31 |
| identity | iterative_dopt | 5 | Strong nonlinear | 6 | 8.973e-14 | 4.194e-14 | 2.631e-08 | 31 |
| identity | iterative_dopt | 10 | Quadratic | 6 | 1.616e-14 | 1.148e-14 | 1.089e-09 | 86 |
| identity | iterative_dopt | 10 | Smooth | 6 | 4.762e-14 | 3.315e-14 | 1.288e-08 | 86 |
| identity | iterative_dopt | 10 | Strong nonlinear | 6 | 4.958e-14 | 3.285e-14 | 3.662e-08 | 86 |
| identity | iterative_dopt | 20 | Quadratic | 6 | 8.211e-15 | 1.351e-14 | 1.788e-09 | 271 |
| identity | iterative_dopt | 20 | Smooth | 6 | 2.637e-14 | 4.338e-14 | 1.577e-08 | 271 |
| identity | iterative_dopt | 20 | Strong nonlinear | 6 | 3.091e-14 | 7.454e-14 | 3.847e-08 | 271 |
| identity | iterative_dopt | 50 | Quadratic | 6 | 7.321e-15 | 2.864e-14 | 8.225e-10 | 1426 |
| identity | iterative_dopt | 50 | Smooth | 6 | 2.581e-14 | 1.535e-13 | 2.525e-08 | 1426 |
| identity | iterative_dopt | 50 | Strong nonlinear | 6 | 2.716e-14 | 1.461e-13 | 6.730e-08 | 1426 |
| identity | measure_weighted | 5 | High-frequency | 6 | 2.023e-14 | 1.162e-14 | 2.895e-08 | 31 |
| identity | measure_weighted | 5 | Local interior | 6 | 1.284e-14 | 5.144e-15 | 6.259e-10 | 31 |
| identity | measure_weighted | 10 | High-frequency | 6 | 1.035e-14 | 7.332e-15 | 1.941e-08 | 86 |
| identity | measure_weighted | 10 | Local interior | 6 | 3.635e-15 | 2.526e-15 | 3.284e-09 | 86 |
| identity | measure_weighted | 20 | High-frequency | 6 | 4.343e-15 | 9.571e-15 | 2.186e-08 | 271 |
| identity | measure_weighted | 20 | Local interior | 6 | 1.478e-15 | 2.239e-15 | 1.254e-09 | 271 |
| identity | measure_weighted | 50 | High-frequency | 6 | 3.996e-15 | 2.084e-14 | 1.660e-08 | 1426 |
| identity | measure_weighted | 50 | Local interior | 6 | 1.290e-15 | 5.498e-15 | 7.798e-09 | 1426 |
| relu | gpu_geometry | 5 | Random nonlinear | 2 | 2.760e-03 | 1.233e-03 | 2.459e-02 | 31 |
| relu | gpu_geometry | 5 | Sparse cubic | 6 | 1.114e-03 | 4.240e-04 | 1.367e-02 | 31 |
| relu | gpu_geometry | 5 | Sparse quartic | 6 | 2.144e-03 | 6.576e-04 | 1.970e-02 | 31 |
| relu | gpu_geometry | 10 | Random nonlinear | 2 | 7.560e-04 | 6.089e-04 | 1.132e-02 | 86 |
| relu | gpu_geometry | 10 | Sparse cubic | 6 | 7.898e-04 | 5.806e-04 | 1.192e-02 | 86 |
| relu | gpu_geometry | 10 | Sparse quartic | 6 | 1.154e-03 | 8.188e-04 | 1.737e-02 | 86 |
| relu | gpu_geometry | 20 | Random nonlinear | 3 | 8.412e-04 | 1.268e-03 | 1.490e-02 | 271 |
| relu | gpu_geometry | 20 | Sparse cubic | 6 | 6.446e-04 | 9.430e-04 | 8.275e-03 | 271 |
| relu | gpu_geometry | 20 | Sparse quartic | 6 | 7.009e-04 | 9.917e-04 | 1.117e-02 | 271 |
| relu | gpu_geometry | 50 | Random nonlinear | 6 | 6.492e-04 | 2.399e-03 | 1.166e-02 | 1426 |
| relu | gpu_geometry | 50 | Sparse cubic | 6 | 2.553e-04 | 9.545e-04 | 3.184e-03 | 1426 |
| relu | gpu_geometry | 50 | Sparse quartic | 6 | 4.079e-04 | 1.434e-03 | 4.484e-03 | 1426 |
| relu | iterative_dopt | 5 | Quadratic | 5 | 2.248e-03 | 8.413e-04 | 8.625e-03 | 31 |
| relu | iterative_dopt | 5 | Smooth | 6 | 8.260e-04 | 4.076e-04 | 4.517e-03 | 31 |
| relu | iterative_dopt | 5 | Strong nonlinear | 5 | 1.303e-03 | 7.769e-04 | 5.875e-03 | 31 |
| relu | iterative_dopt | 10 | Quadratic | 6 | 3.153e-03 | 2.328e-03 | 1.501e-02 | 86 |
| relu | iterative_dopt | 10 | Smooth | 6 | 8.370e-04 | 6.534e-04 | 8.255e-03 | 86 |
| relu | iterative_dopt | 10 | Strong nonlinear | 6 | 8.942e-04 | 8.504e-04 | 8.652e-04 | 86 |
| relu | iterative_dopt | 20 | Quadratic | 6 | 3.191e-03 | 4.824e-03 | 2.947e-02 | 271 |
| relu | iterative_dopt | 20 | Smooth | 6 | 9.760e-04 | 1.426e-03 | 1.602e-02 | 271 |
| relu | iterative_dopt | 20 | Strong nonlinear | 6 | 1.016e-03 | 1.591e-03 | 7.255e-03 | 271 |
| relu | iterative_dopt | 50 | Quadratic | 6 | 2.987e-03 | 1.134e-02 | 4.787e-02 | 1426 |
| relu | iterative_dopt | 50 | Smooth | 6 | 1.168e-03 | 4.327e-03 | 3.552e-02 | 1426 |
| relu | iterative_dopt | 50 | Strong nonlinear | 6 | 1.146e-03 | 4.226e-03 | 2.626e-02 | 1426 |
| relu | measure_weighted | 50 | Local interior | 1 | 4.338e-03 | 1.566e-02 | 3.266e-02 | 1426 |
| sigmoid | gpu_geometry | 5 | Random nonlinear | 2 | 2.755e-03 | 1.225e-03 | 2.569e-02 | 31 |
| sigmoid | gpu_geometry | 5 | Sparse cubic | 6 | 1.096e-03 | 4.052e-04 | 1.430e-02 | 31 |
| sigmoid | gpu_geometry | 5 | Sparse quartic | 6 | 2.012e-03 | 6.614e-04 | 2.058e-02 | 31 |
| sigmoid | gpu_geometry | 10 | Random nonlinear | 2 | 1.009e-03 | 5.694e-04 | 1.410e-02 | 86 |
| sigmoid | gpu_geometry | 10 | Sparse cubic | 6 | 7.228e-04 | 5.611e-04 | 1.147e-02 | 86 |
| sigmoid | gpu_geometry | 10 | Sparse quartic | 6 | 1.196e-03 | 7.806e-04 | 1.641e-02 | 86 |
| sigmoid | gpu_geometry | 20 | Random nonlinear | 4 | 2.405e-03 | 3.521e-03 | 3.667e-02 | 271 |
| sigmoid | gpu_geometry | 20 | Sparse cubic | 6 | 6.266e-04 | 9.543e-04 | 9.115e-03 | 271 |
| sigmoid | gpu_geometry | 20 | Sparse quartic | 6 | 6.813e-04 | 9.855e-04 | 1.398e-02 | 271 |
| sigmoid | gpu_geometry | 50 | Random nonlinear | 5 | 7.858e-04 | 3.007e-03 | 1.444e-02 | 1426 |
| sigmoid | gpu_geometry | 50 | Sparse cubic | 6 | 2.455e-04 | 9.253e-04 | 7.327e-03 | 1426 |
| sigmoid | gpu_geometry | 50 | Sparse quartic | 6 | 5.388e-04 | 1.936e-03 | 1.094e-02 | 1426 |
| sigmoid | iterative_dopt | 5 | Quadratic | 4 | 1.018e-03 | 4.158e-04 | 3.953e-03 | 31 |
| sigmoid | iterative_dopt | 5 | Smooth | 6 | 4.809e-04 | 2.130e-04 | 2.004e-03 | 31 |
| sigmoid | iterative_dopt | 5 | Strong nonlinear | 6 | 1.375e-03 | 6.702e-04 | 8.228e-03 | 31 |
| sigmoid | iterative_dopt | 10 | Quadratic | 6 | 2.811e-03 | 2.211e-03 | 1.627e-02 | 86 |
| sigmoid | iterative_dopt | 10 | Smooth | 6 | 6.588e-04 | 4.936e-04 | 6.154e-03 | 86 |
| sigmoid | iterative_dopt | 10 | Strong nonlinear | 6 | 1.021e-03 | 6.801e-04 | 1.740e-03 | 86 |
| sigmoid | iterative_dopt | 20 | Quadratic | 4 | 4.482e-03 | 6.694e-03 | 3.895e-02 | 271 |
| sigmoid | iterative_dopt | 20 | Smooth | 6 | 1.100e-03 | 1.633e-03 | 1.897e-02 | 271 |
| sigmoid | iterative_dopt | 20 | Strong nonlinear | 6 | 1.066e-03 | 1.900e-03 | 8.379e-03 | 271 |
| sigmoid | iterative_dopt | 50 | Smooth | 6 | 3.604e-03 | 1.335e-02 | 7.614e-02 | 1426 |
| sigmoid | iterative_dopt | 50 | Strong nonlinear | 6 | 1.795e-03 | 7.265e-03 | 4.633e-02 | 1426 |
| softplus | gpu_geometry | 5 | Random nonlinear | 2 | 3.215e-03 | 1.174e-03 | 2.887e-02 | 31 |
| softplus | gpu_geometry | 5 | Sparse cubic | 6 | 1.130e-03 | 4.011e-04 | 1.672e-02 | 31 |
| softplus | gpu_geometry | 5 | Sparse quartic | 6 | 1.941e-03 | 6.510e-04 | 2.202e-02 | 31 |
| softplus | gpu_geometry | 10 | Random nonlinear | 2 | 6.107e-04 | 4.915e-04 | 1.439e-02 | 86 |
| softplus | gpu_geometry | 10 | Sparse cubic | 6 | 7.367e-04 | 5.383e-04 | 1.805e-02 | 86 |
| softplus | gpu_geometry | 10 | Sparse quartic | 6 | 1.091e-03 | 8.050e-04 | 2.203e-02 | 86 |
| softplus | gpu_geometry | 20 | Random nonlinear | 3 | 7.741e-04 | 9.912e-04 | 2.536e-02 | 271 |
| softplus | gpu_geometry | 20 | Sparse cubic | 6 | 5.403e-04 | 8.277e-04 | 2.062e-02 | 271 |
| softplus | gpu_geometry | 20 | Sparse quartic | 6 | 5.990e-04 | 8.835e-04 | 2.162e-02 | 271 |
| softplus | gpu_geometry | 50 | Random nonlinear | 6 | 4.270e-04 | 1.498e-03 | 2.399e-02 | 1426 |
| softplus | gpu_geometry | 50 | Sparse cubic | 6 | 1.437e-04 | 5.409e-04 | 1.231e-02 | 1426 |
| softplus | gpu_geometry | 50 | Sparse quartic | 6 | 2.260e-04 | 8.350e-04 | 1.719e-02 | 1426 |
| softplus | iterative_dopt | 5 | Quadratic | 5 | 7.145e-04 | 3.478e-04 | 3.082e-03 | 31 |
| softplus | iterative_dopt | 5 | Smooth | 6 | 3.653e-04 | 1.706e-04 | 1.570e-03 | 31 |
| softplus | iterative_dopt | 5 | Strong nonlinear | 5 | 1.308e-03 | 6.901e-04 | 8.519e-03 | 31 |
| softplus | iterative_dopt | 10 | Quadratic | 6 | 3.486e-03 | 2.851e-03 | 2.001e-02 | 86 |
| softplus | iterative_dopt | 10 | Smooth | 6 | 4.831e-04 | 4.032e-04 | 5.121e-03 | 86 |
| softplus | iterative_dopt | 10 | Strong nonlinear | 6 | 7.189e-04 | 6.470e-04 | 5.957e-03 | 86 |
| softplus | iterative_dopt | 20 | Smooth | 6 | 1.059e-03 | 1.483e-03 | 1.698e-02 | 271 |
| softplus | iterative_dopt | 20 | Strong nonlinear | 6 | 7.519e-04 | 1.025e-03 | 4.793e-03 | 271 |
| softplus | iterative_dopt | 50 | Quadratic | 2 | 4.681e-03 | 1.682e-02 | 6.648e-02 | 1426 |
| softplus | iterative_dopt | 50 | Smooth | 6 | 1.059e-03 | 4.452e-03 | 3.521e-02 | 1426 |
| softplus | iterative_dopt | 50 | Strong nonlinear | 6 | 7.398e-04 | 2.919e-03 | 1.748e-02 | 1426 |
| tanh | gpu_geometry | 5 | Random nonlinear | 2 | 2.784e-03 | 1.211e-03 | 2.828e-02 | 31 |
| tanh | gpu_geometry | 5 | Sparse cubic | 6 | 1.067e-03 | 3.825e-04 | 1.648e-02 | 31 |
| tanh | gpu_geometry | 5 | Sparse quartic | 6 | 2.197e-03 | 6.777e-04 | 2.169e-02 | 31 |
| tanh | gpu_geometry | 10 | Random nonlinear | 2 | 7.874e-04 | 5.314e-04 | 1.713e-02 | 86 |
| tanh | gpu_geometry | 10 | Sparse cubic | 6 | 7.106e-04 | 5.462e-04 | 1.694e-02 | 86 |
| tanh | gpu_geometry | 10 | Sparse quartic | 6 | 9.908e-04 | 7.547e-04 | 1.939e-02 | 86 |
| tanh | gpu_geometry | 20 | Random nonlinear | 3 | 8.483e-04 | 1.015e-03 | 2.319e-02 | 271 |
| tanh | gpu_geometry | 20 | Sparse cubic | 6 | 5.736e-04 | 8.206e-04 | 1.717e-02 | 271 |
| tanh | gpu_geometry | 20 | Sparse quartic | 6 | 5.976e-04 | 8.358e-04 | 2.019e-02 | 271 |
| tanh | gpu_geometry | 50 | Random nonlinear | 6 | 3.754e-04 | 1.380e-03 | 2.369e-02 | 1426 |
| tanh | gpu_geometry | 50 | Sparse cubic | 6 | 1.470e-04 | 5.082e-04 | 8.365e-03 | 1426 |
| tanh | gpu_geometry | 50 | Sparse quartic | 6 | 2.422e-04 | 8.951e-04 | 1.238e-02 | 1426 |
| tanh | iterative_dopt | 5 | Quadratic | 6 | 1.007e-03 | 4.339e-04 | 3.336e-03 | 31 |
| tanh | iterative_dopt | 5 | Smooth | 6 | 3.122e-04 | 1.535e-04 | 1.533e-03 | 31 |
| tanh | iterative_dopt | 5 | Strong nonlinear | 5 | 1.250e-03 | 6.930e-04 | 9.183e-03 | 31 |
| tanh | iterative_dopt | 10 | Quadratic | 6 | 1.636e-03 | 1.227e-03 | 9.355e-03 | 86 |
| tanh | iterative_dopt | 10 | Smooth | 6 | 4.397e-04 | 3.448e-04 | 4.433e-03 | 86 |
| tanh | iterative_dopt | 10 | Strong nonlinear | 6 | 6.355e-04 | 5.765e-04 | 6.066e-03 | 86 |
| tanh | iterative_dopt | 20 | Quadratic | 6 | 2.402e-03 | 3.591e-03 | 2.290e-02 | 271 |
| tanh | iterative_dopt | 20 | Smooth | 6 | 5.566e-04 | 8.038e-04 | 9.899e-03 | 271 |
| tanh | iterative_dopt | 20 | Strong nonlinear | 6 | 5.524e-04 | 9.579e-04 | 2.518e-03 | 271 |
| tanh | iterative_dopt | 50 | Smooth | 6 | 2.304e-03 | 8.359e-03 | 5.817e-02 | 1426 |
| tanh | iterative_dopt | 50 | Strong nonlinear | 6 | 8.055e-04 | 2.955e-03 | 1.880e-02 | 1426 |

## 总体 gain 摘要

| activation | p | criterion | mean gain | median gain | win rate | reps |
|---|---:|---|---:|---:|---:|---:|
| identity | 5 | D-optimal | 99.26% | 99.70% | 100.0% | 144 |
| identity | 5 | A-optimal | 99.18% | 99.34% | 100.0% | 144 |
| identity | 5 | I-optimal | 91.41% | 94.47% | 100.0% | 144 |
| identity | 5 | Latin hypercube | 97.84% | 98.55% | 100.0% | 144 |
| identity | 10 | D-optimal | 98.72% | 98.92% | 100.0% | 144 |
| identity | 10 | A-optimal | 96.89% | 98.04% | 100.0% | 144 |
| identity | 10 | I-optimal | 74.99% | 80.73% | 99.3% | 144 |
| identity | 10 | Latin hypercube | 85.43% | 89.38% | 100.0% | 144 |
| identity | 20 | D-optimal | 95.34% | 95.51% | 100.0% | 144 |
| identity | 20 | A-optimal | 95.09% | 95.29% | 100.0% | 144 |
| identity | 20 | I-optimal | 79.59% | 82.03% | 100.0% | 144 |
| identity | 20 | Latin hypercube | 50.69% | 56.07% | 94.4% | 144 |
| identity | 50 | D-optimal | 53.85% | 54.80% | 100.0% | 144 |
| identity | 50 | A-optimal | 58.24% | 60.05% | 100.0% | 144 |
| identity | 50 | I-optimal | 51.26% | 52.78% | 100.0% | 144 |
| identity | 50 | Latin hypercube | -15.22% | -5.23% | 39.6% | 144 |
| relu | 5 | D-optimal | -117.72% | -89.13% | 14.4% | 90 |
| relu | 5 | A-optimal | -150.82% | -108.67% | 17.8% | 90 |
| relu | 5 | I-optimal | -225.89% | -202.89% | 0.0% | 90 |
| relu | 5 | Latin hypercube | -29.63% | -7.62% | 40.0% | 90 |
| relu | 10 | D-optimal | -16.59% | -8.96% | 38.5% | 96 |
| relu | 10 | A-optimal | -24.91% | -22.75% | 26.0% | 96 |
| relu | 10 | I-optimal | -78.16% | -57.59% | 7.3% | 96 |
| relu | 10 | Latin hypercube | 4.39% | 8.29% | 67.7% | 96 |
| relu | 20 | D-optimal | 4.02% | 3.84% | 57.6% | 99 |
| relu | 20 | A-optimal | 1.82% | 2.16% | 56.6% | 99 |
| relu | 20 | I-optimal | -15.07% | -9.44% | 28.3% | 99 |
| relu | 20 | Latin hypercube | 3.64% | 3.35% | 71.7% | 99 |
| relu | 50 | D-optimal | 2.14% | 1.50% | 77.5% | 111 |
| relu | 50 | A-optimal | 1.71% | 1.08% | 71.2% | 111 |
| relu | 50 | I-optimal | 2.24% | 1.47% | 74.8% | 111 |
| relu | 50 | Latin hypercube | -0.30% | 0.05% | 51.4% | 111 |
| sigmoid | 5 | D-optimal | -133.15% | -118.61% | 7.8% | 90 |
| sigmoid | 5 | A-optimal | -152.62% | -119.64% | 14.4% | 90 |
| sigmoid | 5 | I-optimal | -227.19% | -197.94% | 0.0% | 90 |
| sigmoid | 5 | Latin hypercube | -29.61% | -7.41% | 42.2% | 90 |
| sigmoid | 10 | D-optimal | -19.67% | -13.53% | 37.5% | 96 |
| sigmoid | 10 | A-optimal | -24.63% | -24.15% | 29.2% | 96 |
| sigmoid | 10 | I-optimal | -80.51% | -47.90% | 6.2% | 96 |
| sigmoid | 10 | Latin hypercube | 0.60% | 4.63% | 57.3% | 96 |
| sigmoid | 20 | D-optimal | 3.26% | 1.49% | 51.0% | 96 |
| sigmoid | 20 | A-optimal | 1.14% | 0.97% | 52.1% | 96 |
| sigmoid | 20 | I-optimal | -15.22% | -10.06% | 31.2% | 96 |
| sigmoid | 20 | Latin hypercube | 4.70% | 5.46% | 80.2% | 96 |
| sigmoid | 50 | D-optimal | 1.22% | 1.31% | 73.6% | 87 |
| sigmoid | 50 | A-optimal | 1.10% | 0.85% | 67.8% | 87 |
| sigmoid | 50 | I-optimal | 1.25% | 0.85% | 73.6% | 87 |
| sigmoid | 50 | Latin hypercube | 0.12% | 0.08% | 52.9% | 87 |
| softplus | 5 | D-optimal | -118.45% | -102.26% | 12.2% | 90 |
| softplus | 5 | A-optimal | -152.38% | -109.36% | 14.4% | 90 |
| softplus | 5 | I-optimal | -228.48% | -239.53% | 1.1% | 90 |
| softplus | 5 | Latin hypercube | -27.97% | -18.87% | 34.4% | 90 |
| softplus | 10 | D-optimal | -23.01% | -11.01% | 36.5% | 96 |
| softplus | 10 | A-optimal | -30.49% | -25.33% | 22.9% | 96 |
| softplus | 10 | I-optimal | -87.35% | -49.57% | 2.1% | 96 |
| softplus | 10 | Latin hypercube | 0.91% | 4.59% | 65.6% | 96 |
| softplus | 20 | D-optimal | -1.69% | 0.68% | 53.1% | 81 |
| softplus | 20 | A-optimal | -3.37% | -2.19% | 43.2% | 81 |
| softplus | 20 | I-optimal | -22.32% | -17.04% | 22.2% | 81 |
| softplus | 20 | Latin hypercube | 3.08% | 2.21% | 59.3% | 81 |
| softplus | 50 | D-optimal | 1.48% | 1.24% | 65.6% | 96 |
| softplus | 50 | A-optimal | 1.06% | 0.47% | 59.4% | 96 |
| softplus | 50 | I-optimal | 1.41% | 1.02% | 65.6% | 96 |
| softplus | 50 | Latin hypercube | -0.20% | -0.13% | 47.9% | 96 |
| tanh | 5 | D-optimal | -109.41% | -78.49% | 5.4% | 93 |
| tanh | 5 | A-optimal | -137.71% | -98.81% | 11.8% | 93 |
| tanh | 5 | I-optimal | -207.10% | -214.68% | 2.2% | 93 |
| tanh | 5 | Latin hypercube | -32.99% | -4.70% | 36.6% | 93 |
| tanh | 10 | D-optimal | -23.41% | -15.70% | 34.4% | 96 |
| tanh | 10 | A-optimal | -32.10% | -28.32% | 25.0% | 96 |
| tanh | 10 | I-optimal | -88.03% | -70.52% | 5.2% | 96 |
| tanh | 10 | Latin hypercube | 0.04% | 6.11% | 61.5% | 96 |
| tanh | 20 | D-optimal | 2.78% | 2.79% | 54.5% | 99 |
| tanh | 20 | A-optimal | 0.66% | 0.51% | 52.5% | 99 |
| tanh | 20 | I-optimal | -19.31% | -13.42% | 23.2% | 99 |
| tanh | 20 | Latin hypercube | 2.36% | 2.26% | 63.6% | 99 |
| tanh | 50 | D-optimal | 1.75% | 1.20% | 73.3% | 90 |
| tanh | 50 | A-optimal | 1.37% | 0.72% | 63.3% | 90 |
| tanh | 50 | I-optimal | 1.92% | 1.54% | 72.2% | 90 |
| tanh | 50 | Latin hypercube | -1.86% | -0.56% | 33.3% | 90 |

## 代表预算下的 case-level gain

| activation | p | case | budget x params | D-optimal | A-optimal | I-optimal | Latin hypercube |
|---|---:|---|---:|---:|---:|---:|---:|
| identity | 5 | High-frequency | 3.0 | 99.35% | 99.25% | 96.13% | 97.77% |
| identity | 5 | Local interior | 3.0 | 99.12% | 99.27% | 92.57% | 97.68% |
| identity | 5 | Random nonlinear | 3.0 | 99.46% | 99.19% | 96.02% | 98.02% |
| identity | 5 | Quadratic | 3.0 | 99.72% | 99.43% | 96.66% | 99.05% |
| identity | 5 | Sparse cubic | 3.0 | 99.71% | 99.63% | 92.02% | 98.45% |
| identity | 5 | Sparse quartic | 3.0 | 99.48% | 99.30% | 86.63% | 98.67% |
| identity | 5 | Smooth | 3.0 | 98.60% | 99.06% | 92.52% | 97.14% |
| identity | 5 | Strong nonlinear | 3.0 | 98.87% | 99.23% | 91.25% | 97.41% |
| identity | 10 | High-frequency | 3.0 | 98.87% | 96.34% | 76.84% | 82.04% |
| identity | 10 | Local interior | 3.0 | 98.55% | 96.56% | 71.69% | 84.18% |
| identity | 10 | Random nonlinear | 3.0 | 98.23% | 95.50% | 55.67% | 84.22% |
| identity | 10 | Quadratic | 3.0 | 98.96% | 98.05% | 80.63% | 89.20% |
| identity | 10 | Sparse cubic | 3.0 | 99.24% | 99.00% | 89.12% | 91.09% |
| identity | 10 | Sparse quartic | 3.0 | 99.24% | 98.96% | 81.39% | 90.02% |
| identity | 10 | Smooth | 3.0 | 98.42% | 95.20% | 68.42% | 81.17% |
| identity | 10 | Strong nonlinear | 3.0 | 98.72% | 96.68% | 77.33% | 83.14% |
| identity | 20 | High-frequency | 3.0 | 95.91% | 95.62% | 79.57% | 50.66% |
| identity | 20 | Local interior | 3.0 | 94.52% | 93.95% | 67.62% | 54.84% |
| identity | 20 | Random nonlinear | 3.0 | 94.65% | 94.85% | 74.56% | 47.08% |
| identity | 20 | Quadratic | 3.0 | 94.90% | 93.70% | 74.45% | 51.45% |
| identity | 20 | Sparse cubic | 3.0 | 95.81% | 95.13% | 74.89% | 52.37% |
| identity | 20 | Sparse quartic | 3.0 | 95.21% | 95.10% | 75.66% | 50.72% |
| identity | 20 | Smooth | 3.0 | 95.80% | 95.10% | 76.76% | 53.64% |
| identity | 20 | Strong nonlinear | 3.0 | 95.95% | 95.15% | 78.25% | 55.13% |
| identity | 50 | High-frequency | 3.0 | 53.94% | 60.56% | 52.73% | -33.16% |
| identity | 50 | Local interior | 3.0 | 52.97% | 58.52% | 51.40% | 1.23% |
| identity | 50 | Random nonlinear | 3.0 | 51.54% | 57.36% | 46.14% | -8.53% |
| identity | 50 | Quadratic | 3.0 | 60.05% | 63.82% | 57.40% | -3.10% |
| identity | 50 | Sparse cubic | 3.0 | 56.11% | 61.06% | 51.60% | -11.06% |
| identity | 50 | Sparse quartic | 3.0 | 55.90% | 60.14% | 51.51% | -9.72% |
| identity | 50 | Smooth | 3.0 | 53.83% | 60.66% | 52.63% | -36.55% |
| identity | 50 | Strong nonlinear | 3.0 | 53.13% | 59.73% | 51.94% | -37.60% |
| relu | 5 | Random nonlinear | 3.0 | -48.62% | -357.58% | -173.70% | 47.62% |
| relu | 5 | Quadratic | 3.0 | -5.39% | 4.58% | -37.70% | 11.51% |
| relu | 5 | Sparse cubic | 3.0 | -101.32% | -182.14% | -320.39% | -53.38% |
| relu | 5 | Sparse quartic | 3.0 | -302.33% | -331.43% | -471.94% | -91.37% |
| relu | 5 | Smooth | 3.0 | -81.95% | -21.77% | -65.24% | 13.41% |
| relu | 5 | Strong nonlinear | 3.0 | -167.94% | -152.54% | -249.49% | 16.94% |
| relu | 10 | Random nonlinear | 3.0 | -77.52% | -62.05% | -26.39% | 21.90% |
| relu | 10 | Quadratic | 3.0 | 17.31% | 10.44% | -16.70% | 15.01% |
| relu | 10 | Sparse cubic | 3.0 | -22.40% | -34.99% | -96.16% | -8.08% |
| relu | 10 | Sparse quartic | 3.0 | -33.41% | -23.83% | -108.17% | -7.39% |
| relu | 10 | Smooth | 3.0 | 17.12% | 5.61% | -30.48% | 12.66% |
| relu | 10 | Strong nonlinear | 3.0 | -50.83% | -73.59% | -167.98% | 8.94% |
| relu | 20 | Random nonlinear | 3.0 | 9.79% | 7.90% | -15.52% | 19.08% |
| relu | 20 | Quadratic | 3.0 | 12.53% | 8.54% | -1.34% | 5.98% |
| relu | 20 | Sparse cubic | 3.0 | -6.38% | -6.61% | -31.99% | -0.96% |
| relu | 20 | Sparse quartic | 3.0 | -4.42% | -12.29% | -33.92% | -1.71% |
| relu | 20 | Smooth | 3.0 | 9.74% | 6.04% | -3.39% | 3.57% |
| relu | 20 | Strong nonlinear | 3.0 | 3.51% | 1.98% | -15.86% | -0.84% |
| relu | 50 | Local interior | 3.0 | 9.13% | 5.62% | 7.09% | 0.77% |
| relu | 50 | Random nonlinear | 3.0 | 3.86% | 3.67% | 4.36% | -0.28% |
| relu | 50 | Quadratic | 3.0 | 3.55% | 1.76% | 2.81% | -1.22% |
| relu | 50 | Sparse cubic | 3.0 | -0.27% | -0.06% | -0.01% | 0.30% |
| relu | 50 | Sparse quartic | 3.0 | -0.55% | -0.76% | 0.30% | -1.76% |
| relu | 50 | Smooth | 3.0 | 1.46% | 1.18% | 0.86% | -0.55% |
| relu | 50 | Strong nonlinear | 3.0 | 3.82% | 3.11% | 2.92% | -0.19% |
| sigmoid | 5 | Random nonlinear | 3.0 | -42.85% | -337.90% | -109.12% | 36.01% |
| sigmoid | 5 | Quadratic | 3.0 | 3.36% | -8.75% | -86.01% | 4.37% |
| sigmoid | 5 | Sparse cubic | 3.0 | -101.43% | -165.97% | -312.06% | -53.15% |
| sigmoid | 5 | Sparse quartic | 3.0 | -288.90% | -314.55% | -422.30% | -151.88% |
| sigmoid | 5 | Smooth | 3.0 | -91.13% | -22.86% | -49.06% | -6.18% |
| sigmoid | 5 | Strong nonlinear | 3.0 | -210.29% | -161.80% | -246.38% | -5.32% |
| sigmoid | 10 | Random nonlinear | 3.0 | -77.78% | -59.41% | -30.59% | 19.55% |
| sigmoid | 10 | Quadratic | 3.0 | 18.99% | 10.22% | -15.91% | 15.98% |
| sigmoid | 10 | Sparse cubic | 3.0 | -20.21% | -32.05% | -86.74% | 3.13% |
| sigmoid | 10 | Sparse quartic | 3.0 | -31.92% | -22.29% | -102.83% | -18.38% |
| sigmoid | 10 | Smooth | 3.0 | 13.47% | 4.22% | -24.33% | 10.72% |
| sigmoid | 10 | Strong nonlinear | 3.0 | -68.65% | -77.61% | -199.78% | -0.75% |
| sigmoid | 20 | Random nonlinear | 3.0 | 13.20% | 11.65% | -10.15% | 18.15% |
| sigmoid | 20 | Quadratic | 3.0 | 10.43% | 7.72% | 0.56% | 6.77% |
| sigmoid | 20 | Sparse cubic | 3.0 | -6.53% | -5.71% | -26.44% | 0.77% |
| sigmoid | 20 | Sparse quartic | 3.0 | -3.83% | -13.54% | -33.25% | -4.66% |
| sigmoid | 20 | Smooth | 3.0 | 10.00% | 7.04% | -2.50% | 4.18% |
| sigmoid | 20 | Strong nonlinear | 3.0 | 0.70% | -2.98% | -19.80% | 7.48% |
| sigmoid | 50 | Random nonlinear | 3.0 | 3.89% | 3.06% | 4.10% | 1.83% |
| sigmoid | 50 | Sparse cubic | 3.0 | 0.21% | 0.38% | 0.55% | -1.49% |
| sigmoid | 50 | Sparse quartic | 3.0 | -0.70% | -0.09% | 0.16% | -1.35% |
| sigmoid | 50 | Smooth | 3.0 | 1.72% | 1.24% | 1.62% | 0.10% |
| sigmoid | 50 | Strong nonlinear | 3.0 | 1.97% | 1.68% | 1.60% | 0.17% |
| softplus | 5 | Random nonlinear | 3.0 | -52.44% | -366.89% | -164.55% | 63.59% |
| softplus | 5 | Quadratic | 3.0 | -9.02% | 10.83% | -37.41% | 7.99% |
| softplus | 5 | Sparse cubic | 3.0 | -99.86% | -168.07% | -298.55% | -36.50% |
| softplus | 5 | Sparse quartic | 3.0 | -280.21% | -311.24% | -437.80% | -54.21% |
| softplus | 5 | Smooth | 3.0 | -83.69% | -38.18% | -98.34% | -24.06% |
| softplus | 5 | Strong nonlinear | 3.0 | -157.59% | -159.44% | -241.83% | -14.94% |
| softplus | 10 | Random nonlinear | 3.0 | -104.26% | -91.93% | -38.77% | 23.16% |
| softplus | 10 | Quadratic | 3.0 | 13.18% | 9.30% | -15.50% | 6.38% |
| softplus | 10 | Sparse cubic | 3.0 | -19.04% | -32.03% | -89.64% | 5.67% |
| softplus | 10 | Sparse quartic | 3.0 | -32.88% | -21.18% | -106.43% | -6.10% |
| softplus | 10 | Smooth | 3.0 | 7.82% | -2.52% | -41.63% | 3.18% |
| softplus | 10 | Strong nonlinear | 3.0 | -68.97% | -91.12% | -204.83% | 1.27% |
| softplus | 20 | Random nonlinear | 3.0 | 7.89% | 6.77% | -17.48% | 25.95% |
| softplus | 20 | Sparse cubic | 3.0 | -7.68% | -7.26% | -29.24% | -1.69% |
| softplus | 20 | Sparse quartic | 3.0 | -0.97% | -10.56% | -30.24% | 0.65% |
| softplus | 20 | Smooth | 3.0 | 8.13% | 5.71% | -3.61% | 4.82% |
| softplus | 20 | Strong nonlinear | 3.0 | -9.06% | -10.81% | -33.80% | 8.60% |
| softplus | 50 | Random nonlinear | 3.0 | 6.82% | 5.36% | 6.72% | 4.21% |
| softplus | 50 | Quadratic | 3.0 | 1.50% | 1.71% | 1.60% | 0.05% |
| softplus | 50 | Sparse cubic | 3.0 | -0.22% | 0.70% | 0.72% | -0.45% |
| softplus | 50 | Sparse quartic | 3.0 | -0.64% | -0.85% | -0.20% | -1.05% |
| softplus | 50 | Smooth | 3.0 | 0.91% | 0.90% | 0.58% | 0.26% |
| softplus | 50 | Strong nonlinear | 3.0 | 2.47% | 1.85% | 1.17% | 0.36% |
| tanh | 5 | Random nonlinear | 3.0 | -27.16% | -280.39% | -130.90% | 29.61% |
| tanh | 5 | Quadratic | 3.0 | -28.40% | -8.13% | -32.36% | 4.50% |
| tanh | 5 | Sparse cubic | 3.0 | -96.33% | -170.18% | -278.36% | -64.32% |
| tanh | 5 | Sparse quartic | 3.0 | -270.89% | -277.33% | -375.56% | -91.78% |
| tanh | 5 | Smooth | 3.0 | -88.96% | -14.16% | -121.31% | 10.39% |
| tanh | 5 | Strong nonlinear | 3.0 | -145.22% | -172.29% | -274.11% | -5.54% |
| tanh | 10 | Random nonlinear | 3.0 | -73.53% | -60.83% | -28.05% | 18.37% |
| tanh | 10 | Quadratic | 3.0 | 13.95% | 10.54% | -15.33% | 11.61% |
| tanh | 10 | Sparse cubic | 3.0 | -23.41% | -34.66% | -95.36% | -3.38% |
| tanh | 10 | Sparse quartic | 3.0 | -29.35% | -22.21% | -94.26% | -3.76% |
| tanh | 10 | Smooth | 3.0 | 8.84% | -6.80% | -52.25% | 10.83% |
| tanh | 10 | Strong nonlinear | 3.0 | -84.03% | -104.26% | -206.35% | -21.71% |
| tanh | 20 | Random nonlinear | 3.0 | 10.50% | 8.42% | -12.15% | 22.50% |
| tanh | 20 | Quadratic | 3.0 | 12.00% | 9.57% | -4.00% | 4.61% |
| tanh | 20 | Sparse cubic | 3.0 | -9.48% | -6.80% | -31.88% | 0.12% |
| tanh | 20 | Sparse quartic | 3.0 | -2.67% | -13.18% | -37.01% | -4.38% |
| tanh | 20 | Smooth | 3.0 | 12.36% | 8.43% | -5.75% | 3.47% |
| tanh | 20 | Strong nonlinear | 3.0 | -2.45% | -6.06% | -33.70% | 2.69% |
| tanh | 50 | Random nonlinear | 3.0 | 7.87% | 6.12% | 7.50% | -11.23% |
| tanh | 50 | Sparse cubic | 3.0 | -0.46% | 0.33% | 0.83% | -0.67% |
| tanh | 50 | Sparse quartic | 3.0 | -0.37% | -0.65% | 1.09% | -1.84% |
| tanh | 50 | Smooth | 3.0 | 1.53% | 1.49% | 1.81% | -0.51% |
| tanh | 50 | Strong nonlinear | 3.0 | 2.50% | 2.09% | 2.42% | -0.76% |

## 图

- `figures\close_distance_highdim_optimality\close_distance_highdim_optimality_fullgrid_close_filter.png`
- `figures\close_distance_highdim_optimality\close_distance_highdim_optimality_fullgrid_criterion_gain_by_p_activation.png`
- `figures\close_distance_highdim_optimality\close_distance_highdim_optimality_fullgrid_criteria_heatmap_3x.png`

## 解读

这组实验应和上一轮 D-opt-only 结果配合阅读。tanh 条件测试非线性 NN oracle 下的设计准则稳定性；identity 条件是线性网络对照，用来判断失败是否来自 NN 非线性与 FullPR basis 的不匹配。若 identity 下 D/A/I-optimal 显著更稳定，而 tanh 下不稳定，则说明主要限制来自 oracle 非线性或 surrogate 欠设定；若两者都不稳定，则更可能是候选区域、预算或 optimality 准则本身与目标 MSE 不匹配。