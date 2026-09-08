# 高维近距离 D-optimal 迁移实验报告

## 实验目的

本实验把近距离 D-optimal 迁移测试扩展到高维设置，和其他实验的 `p=5,10,20,50` 保持一致。核心问题是：当 NN oracle 能被二阶 FullPR surrogate 在同一数据区域内较好复刻时，FullPR-based D-optimal 选点是否比同预算随机采样更高效。

## 设置

- p values：5, 10, 20, 50。
- cases：highdim_poly2, highdim_smooth, highdim_strong。
- n：10000；训练/评估：7500/2500。
- NN：hidden=128,64,32，activation=tanh，epochs=1000。
- FullPR：degree=2，include_special=True。
- close filter：`shape_nn_fpr <= 5.000e-03`。
- base runs：72；通过 close filter：66；跳过：6。

## 近距离筛选摘要

| p | case | used runs | median shape | median FPR mimic MSE | median RMSE gap | design params |
|---:|---|---:|---:|---:|---:|---:|
| 5 | Quadratic | 6 | 1.116e-03 | 4.339e-04 | 3.336e-03 | 31 |
| 5 | Smooth | 6 | 2.982e-04 | 1.535e-04 | 1.533e-03 | 31 |
| 5 | Strong nonlinear | 6 | 1.107e-03 | 6.947e-04 | 9.232e-03 | 31 |
| 10 | Quadratic | 6 | 1.622e-03 | 1.227e-03 | 9.355e-03 | 86 |
| 10 | Smooth | 6 | 4.557e-04 | 3.448e-04 | 4.433e-03 | 86 |
| 10 | Strong nonlinear | 6 | 7.246e-04 | 5.765e-04 | 6.066e-03 | 86 |
| 20 | Quadratic | 6 | 2.447e-03 | 3.591e-03 | 2.290e-02 | 271 |
| 20 | Smooth | 6 | 5.145e-04 | 8.038e-04 | 9.899e-03 | 271 |
| 20 | Strong nonlinear | 6 | 6.082e-04 | 9.579e-04 | 2.518e-03 | 271 |
| 50 | Smooth | 6 | 2.214e-03 | 8.359e-03 | 5.817e-02 | 1426 |
| 50 | Strong nonlinear | 6 | 7.398e-04 | 2.955e-03 | 1.880e-02 | 1426 |

## D-optimal gain 摘要

| p | case | budget x params | D-opt gain mean +/-95%CI | Latin gain mean +/-95%CI | reps |
|---:|---|---:|---:|---:|---:|
| 5 | Quadratic | 2.0 | -13.23% +/- 22.99% | -20.46% +/- 20.22% | 6 |
| 5 | Quadratic | 3.0 | -28.83% +/- 11.83% | 9.10% +/- 14.09% | 6 |
| 5 | Quadratic | 4.0 | -34.96% +/- 25.46% | -16.76% +/- 29.05% | 6 |
| 5 | Smooth | 2.0 | -46.98% +/- 26.54% | -28.39% +/- 30.66% | 6 |
| 5 | Smooth | 3.0 | -87.27% +/- 33.52% | 10.46% +/- 14.00% | 6 |
| 5 | Smooth | 4.0 | -77.64% +/- 32.14% | -1.85% +/- 25.72% | 6 |
| 5 | Strong nonlinear | 2.0 | -130.68% +/- 121.86% | 4.37% +/- 37.33% | 6 |
| 5 | Strong nonlinear | 3.0 | -195.18% +/- 152.99% | -1.97% +/- 42.13% | 6 |
| 5 | Strong nonlinear | 4.0 | -199.21% +/- 140.55% | -21.31% +/- 57.17% | 6 |
| 10 | Quadratic | 2.0 | 19.68% +/- 17.44% | 20.42% +/- 5.27% | 6 |
| 10 | Quadratic | 3.0 | 13.20% +/- 8.08% | 8.50% +/- 4.42% | 6 |
| 10 | Quadratic | 4.0 | 9.50% +/- 6.19% | 5.07% +/- 3.74% | 6 |
| 10 | Smooth | 2.0 | 16.79% +/- 13.58% | 13.20% +/- 10.14% | 6 |
| 10 | Smooth | 3.0 | 9.07% +/- 10.69% | -0.41% +/- 16.39% | 6 |
| 10 | Smooth | 4.0 | 2.69% +/- 10.44% | 3.66% +/- 5.93% | 6 |
| 10 | Strong nonlinear | 2.0 | -94.46% +/- 49.13% | 9.24% +/- 12.29% | 6 |
| 10 | Strong nonlinear | 3.0 | -78.75% +/- 23.01% | -8.94% +/- 19.19% | 6 |
| 10 | Strong nonlinear | 4.0 | -68.16% +/- 18.81% | -16.24% +/- 24.23% | 6 |
| 20 | Quadratic | 2.0 | 22.15% +/- 5.89% | 6.74% +/- 8.17% | 6 |
| 20 | Quadratic | 3.0 | 12.04% +/- 1.79% | 6.87% +/- 4.01% | 6 |
| 20 | Quadratic | 4.0 | 7.82% +/- 1.84% | 3.25% +/- 2.73% | 6 |
| 20 | Smooth | 2.0 | 17.39% +/- 4.54% | 10.19% +/- 4.81% | 6 |
| 20 | Smooth | 3.0 | 12.09% +/- 2.65% | 5.86% +/- 5.12% | 6 |
| 20 | Smooth | 4.0 | 8.26% +/- 3.16% | 4.00% +/- 1.59% | 6 |
| 20 | Strong nonlinear | 2.0 | -12.15% +/- 26.20% | 3.52% +/- 8.46% | 6 |
| 20 | Strong nonlinear | 3.0 | -2.28% +/- 15.03% | 2.58% +/- 6.46% | 6 |
| 20 | Strong nonlinear | 4.0 | -0.48% +/- 12.95% | 3.80% +/- 8.13% | 6 |
| 50 | Smooth | 2.0 | 4.30% +/- 0.97% | -1.78% +/- 0.97% | 6 |
| 50 | Smooth | 3.0 | 1.35% +/- 0.78% | 0.21% +/- 1.18% | 6 |
| 50 | Smooth | 4.0 | 0.84% +/- 0.38% | -0.40% +/- 0.48% | 6 |
| 50 | Strong nonlinear | 2.0 | 1.95% +/- 1.36% | -1.48% +/- 4.27% | 6 |
| 50 | Strong nonlinear | 3.0 | 2.42% +/- 0.66% | -0.31% +/- 2.38% | 6 |
| 50 | Strong nonlinear | 4.0 | 1.65% +/- 0.73% | -0.40% +/- 0.38% | 6 |

## 图

- `figures\close_distance_highdim_dopt\close_distance_highdim_dopt_close_filter.png`
- `figures\close_distance_highdim_dopt\close_distance_highdim_dopt_gain_by_p.png`
- `figures\close_distance_highdim_dopt\close_distance_highdim_dopt_dopt_gain_heatmap_3x.png`

## 解读

这版实验比低维 p=3 sanity check 更适合放进主线，因为维度、数据生成机制、FullPR 特征阶数和候选区域都与其他高维实验对齐。需要注意的是，D-optimal 优势仍应按 close filter 后的子集解释：它检验的是条件命题，而不是声称 D-optimal 在所有高维场景下都优于随机。