# 论文级高维迭代 D-optimal 实验报告

## 实验目标

本实验检验一个实际流程：在原始高维数据集上先均匀抽取小比例初始设计集，用该设计集估计 NN 与 FullPR surrogate 的距离；随后把当前已经升级后的设计集作为下一轮 D-optimal 的条件信息，迭代加入新样本并重新拟合 surrogate。实验关注 D-optimal 升级是否稳定优于同预算随机升级。

## 实验矩阵

- 原始数据集大小：10000。
- 输入维度：10, 20, 50。
- 数据场景：highdim_poly2, highdim_smooth, highdim_strong。
- data seeds：0, 1, 2。
- NN init seeds：0, 1。
- 初始 uniform 比例：0.05, 0.1。
- 迭代轮数：5；每轮增加 500 个点。
- 随机基准：每个迭代点重复 10 次。
- 设备：cuda；GPU：NVIDIA GeForce RTX 4070 Ti SUPER。
- NN：hidden=128,64,32，activation=tanh，epochs=1000。
- FullPR：degree=2，include_special=True。

## 实际计算量

- NN oracle 训练次数：54。
- D-optimal 迭代轨迹数：108。
- D-optimal surrogate 拟合次数：648。
- 随机 surrogate 拟合次数：6480。

## 最后一轮总体结论

最后一轮全部重复上的平均 MSE gain 为 `1.15%`，正收益比例为 `80.6%`。这里的 gain 定义为 `(random median MSE - D-optimal MSE) / random median MSE`，因此正值表示 D-optimal 优于同预算随机升级。

## 最后一轮分组统计

| device | case | p | init frac | final n | reps | pilot shape | D-opt MSE | random MSE | gain mean ±95%CI | positive rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cuda | highdim_poly2 | 10 | 0.05 | 2875 | 6 | 2.185e-03 | 1.321e-03 | 1.338e-03 | 1.36% ± 0.62% | 100.0% |
| cuda | highdim_poly2 | 10 | 0.1 | 3250 | 6 | 1.545e-03 | 1.319e-03 | 1.337e-03 | 1.51% ± 0.72% | 100.0% |
| cuda | highdim_smooth | 10 | 0.05 | 2875 | 6 | 4.932e-04 | 3.583e-04 | 3.616e-04 | 0.91% ± 0.73% | 83.3% |
| cuda | highdim_smooth | 10 | 0.1 | 3250 | 6 | 4.232e-04 | 3.573e-04 | 3.596e-04 | 0.60% ± 0.94% | 66.7% |
| cuda | highdim_strong | 10 | 0.05 | 2875 | 6 | 7.781e-04 | 6.429e-04 | 5.953e-04 | -8.77% ± 5.82% | 0.0% |
| cuda | highdim_strong | 10 | 0.1 | 3250 | 6 | 7.651e-04 | 6.260e-04 | 5.924e-04 | -6.52% ± 6.48% | 16.7% |
| cuda | highdim_poly2 | 20 | 0.05 | 2875 | 6 | 2.027e-02 | 3.541e-03 | 3.612e-03 | 1.96% ± 0.58% | 100.0% |
| cuda | highdim_poly2 | 20 | 0.1 | 3250 | 6 | 3.053e-03 | 3.516e-03 | 3.576e-03 | 1.70% ± 0.34% | 100.0% |
| cuda | highdim_smooth | 20 | 0.05 | 2875 | 6 | 4.417e-03 | 8.717e-04 | 8.960e-04 | 2.58% ± 1.05% | 100.0% |
| cuda | highdim_smooth | 20 | 0.1 | 3250 | 6 | 6.661e-04 | 8.691e-04 | 8.928e-04 | 2.54% ± 1.08% | 100.0% |
| cuda | highdim_strong | 20 | 0.05 | 2875 | 6 | 7.289e-03 | 1.129e-03 | 1.219e-03 | 5.35% ± 5.33% | 83.3% |
| cuda | highdim_strong | 20 | 0.1 | 3250 | 6 | 8.210e-04 | 1.124e-03 | 1.173e-03 | 2.73% ± 4.17% | 66.7% |
| cuda | highdim_poly2 | 50 | 0.05 | 2875 | 6 | 5.937e-03 | 3.384e-02 | 3.503e-02 | 3.55% ± 1.34% | 100.0% |
| cuda | highdim_poly2 | 50 | 0.1 | 3250 | 6 | 6.584e-03 | 3.280e-02 | 3.372e-02 | 2.82% ± 0.93% | 100.0% |
| cuda | highdim_smooth | 50 | 0.05 | 2875 | 6 | 5.005e-03 | 9.685e-03 | 1.005e-02 | 3.49% ± 0.96% | 100.0% |
| cuda | highdim_smooth | 50 | 0.1 | 3250 | 6 | 4.591e-03 | 9.473e-03 | 9.763e-03 | 2.80% ± 1.47% | 83.3% |
| cuda | highdim_strong | 50 | 0.05 | 2875 | 6 | 9.188e-03 | 3.769e-03 | 3.831e-03 | 1.62% ± 1.21% | 83.3% |
| cuda | highdim_strong | 50 | 0.1 | 3250 | 6 | 1.132e-02 | 3.505e-03 | 3.530e-03 | 0.50% ± 1.80% | 66.7% |

## 论文表述建议

1. 对 p=20，初始设计集通常已经超过二阶 FullPR 的特征数，因此 D-optimal 的优势主要表现为小幅但稳定的改进。
2. 对 p=50，初始设计集低于 FullPR 特征数，早期迭代可能出现欠定不稳定；随着升级后的数据集不断被用于下一轮 D-optimal，误差会在选点数超过特征数后明显收敛。
3. 强非线性场景用于说明方法边界：当 NN oracle 的局部行为不能由二阶 FullPR 充分表达时，D-optimal 仍可能改善采样覆盖，但不应被解释为充分逼近 NN 的保证。
4. 因此论文中的核心结论应写成：pilot 距离和迭代式 D-optimal 升级提供了一个可观测、可复现的数据选择诊断流程；它在高维中可以稳定优于随机升级，但收益幅度取决于 FullPR 特征空间、初始样本比例和迭代预算。

## 输出文件

- `paper_highdim_iter_dopt_gpu_p10_20_50_raw.csv`：所有重复与迭代点的原始结果。
- `paper_highdim_iter_dopt_gpu_p10_20_50_aggregate.csv`：均值、标准差、SEM、95% CI 和正收益比例。
- `paper_highdim_iter_dopt_gpu_p10_20_50_final_summary.csv`：最后一轮分组汇总。
- `paper_highdim_iter_dopt_gpu_p10_20_50_cpu_gain_mean_ci.png` / `paper_highdim_iter_dopt_gpu_p10_20_50_cuda_gain_mean_ci.png`：gain 均值曲线与 95% CI。
- `paper_highdim_iter_dopt_gpu_p10_20_50_cpu_mse_mean_ci.png` / `paper_highdim_iter_dopt_gpu_p10_20_50_cuda_mse_mean_ci.png`：MSE 均值曲线与 95% CI。
- `paper_highdim_iter_dopt_gpu_p10_20_50_final_gain_boxplot.png`：最后一轮 gain 分布箱线图。