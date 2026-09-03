# 距离测度加权采样实验报告

## 实验目的

这次实验把 NN 与 PR/Taylor-PR 的距离测度从事后诊断量改成主动采样策略中的控制变量。核心问题是：实际应用中不知道 NN 与 FullPR/Taylor-PR 是否接近，因此先用训练集的均匀小样本训练一个初始 NN，再用该初始模型与 PR 模型之间的距离决定下一批数据中 D-optimal 点和随机点的比例。

## 流程

对每个数据集先做 75/25 训练/测试拆分。训练集内先均匀抽取 5% 作为 pilot set，并在 pilot set 内再切出一部分验证点。用 pilot fit 子集训练单隐层 tanh NN，然后在 pilot validation 子集上计算：

- FullPR 距离：FullPR 先拟合 pilot NN 的预测值，再与 NN 预测做响应曲面形状距离。
- Taylor-PR 距离：Taylor-PR 由该 NN 的权重做三阶 Taylor 展开得到，再与 NN 预测做距离。

当前主实验的组合距离使用 `min(d_FPR, d_TPR)`。由于 D-optimal 选择本身基于 FullPR 特征空间，这个设置表示：只要存在一个 PR 代理能贴近 NN，就允许更多使用旧的 PR/D-optimal 技术。距离在所有 pilot run 内用 10%/90% 分位数归一化到 `[0,1]`，再转成 D-optimal 比例：

```text
raw_dopt_weight = 1 - normalized_distance
dopt_weight = min_dopt + (max_dopt - min_dopt) * raw_dopt_weight
```

主实验使用 `min_dopt=0.30, max_dopt=0.95`。每轮固定新增 500 个训练点，其中 `round(500 * dopt_weight)` 个由 D-optimal leverage 选择，其余由随机抽样选择。对照组包括全随机、全 D-optimal、Latin hypercube candidate selection。

## 主实验结果

设置：`n=10000, p=20`，数据集为 `highdim_poly2 / highdim_smooth / highdim_strong`，5 个 data seed × 2 个初始采样 seed，共 30 个配置；每个配置训练 5 个 batch round。

| 策略 | Final MSE mean | 相对随机平均提升 | 正提升率 |
| --- | ---: | ---: | ---: |
| D-optimal | 0.003812 | 9.57% | 90.0% |
| Measure-weighted | 0.003909 | 8.44% | 90.0% |
| Latin hypercube | 0.004034 | 3.39% | 63.3% |
| Random | 0.004240 | 0.00% | 0.0% |

成对比较中，measure-weighted 相对 random 的平均提升为 8.44%，95% CI 约为 `[5.25%, 11.63%]`；相对 Latin 的平均提升为 4.75%，95% CI 约为 `[1.39%, 8.11%]`。相对全 D-optimal 的差异为 -1.49%，95% CI 约为 `[-4.54%, 1.56%]`，说明在这些原始数据集上全 D-optimal 本身已经很强，测度加权策略基本达到同一水平，但不应声称显著超过全 D-optimal。

分数据集看，measure-weighted 在 `highdim_strong` 上优于全 D-optimal：

| 数据集 | 最佳策略 | Measure-weighted final MSE | D-optimal final MSE |
| --- | --- | ---: | ---: |
| highdim_poly2 | D-optimal | 0.007466 | 0.007165 |
| highdim_smooth | D-optimal | 0.002407 | 0.002371 |
| highdim_strong | Measure-weighted | 0.001854 | 0.001901 |

图表：

- `measure_weighted_sampling_paper_p20_mse_curve.png`
- `measure_weighted_sampling_paper_p20_final_mse.png`
- `measure_weighted_sampling_paper_p20_weight_vs_gain.png`

## Bounds sensitivity

在较小的 6 配置 sensitivity run 中，不同合理 bounds 的结果如下。`30%-95%`、`30%-90%`、`40%-90%` 都让 measure-weighted 的最终 MSE 与全 D-optimal 非常接近，其中 `30%-95%` 略低于全 D-optimal。

| Bounds | Measure-weighted final MSE | Measure-weighted gain vs random | D-optimal final MSE |
| --- | ---: | ---: | ---: |
| 30%-95% | 0.004061 | 12.46% | 0.004098 |
| 40%-90% | 0.004071 | 10.93% | 0.004098 |
| 30%-90% | 0.004082 | 13.17% | 0.004098 |
| 20%-90% | 0.004179 | 11.38% | 0.004098 |
| 20%-80% | 0.004288 | 7.47% | 0.004098 |

这说明结论不是来自单个极端比例，但在当前原始数据集里，如果 D-optimal 本身普遍有效，混合策略的主要价值是稳健接近最强对照，而不是必然大幅超过它。

## Stress benchmark

为了测试“距离远时少用 D-optimal”的必要性，新增两个 stress 数据集：

- `highdim_local`：响应主要由局部内部峰值和局部 ridge 决定，检验边界高 leverage 点是否过度主导。
- `highdim_highfreq`：响应包含高频非多项式结构，检验 PR 特征空间失配时全 D-optimal 是否失效。

设置同样为 `n=10000, p=20`，5 个 data seed × 2 个初始采样 seed，共 20 个配置，bounds 为 `30%-95%`。

| 策略 | Final MSE mean | 相对随机平均提升 | 正提升率 |
| --- | ---: | ---: | ---: |
| Measure-weighted | 0.007378 | 1.33% | 65.0% |
| Random | 0.007451 | 0.00% | 0.0% |
| Latin hypercube | 0.007518 | -1.27% | 60.0% |
| D-optimal | 0.007631 | -2.48% | 50.0% |

在 stress benchmark 中，measure-weighted 相对全 D-optimal 的 win rate 为 70%，平均提升约 3.31%。这说明当 PR/D-optimal 发生失配时，距离测度确实能减少错误迁移带来的损失。

分数据集看：

| 数据集 | Measure-weighted | Random | Latin | D-optimal |
| --- | ---: | ---: | ---: | ---: |
| highdim_local | 0.004059 | 0.004227 | 0.004184 | 0.004266 |
| highdim_highfreq | 0.010697 | 0.010676 | 0.010851 | 0.010997 |

`highdim_highfreq` 中全 D-optimal 明显劣于随机，measure-weighted 虽然没有超过纯随机，但通过降低 D-optimal 比例显著减轻了全 D-optimal 的损失；`highdim_local` 中 measure-weighted 则为最佳。

图表：

- `measure_weighted_sampling_stress_p20_mse_curve.png`
- `measure_weighted_sampling_stress_p20_final_mse.png`
- `measure_weighted_sampling_stress_p20_weight_vs_gain.png`

## 结论

这组实验更准确地体现了距离测度的作用：它不是一个保证永远超过全 D-optimal 的万能采样器，而是一个迁移决策信号。距离小的时候，策略会提高 D-optimal 比例，接近甚至超过全 D-optimal；距离大的时候，策略会保留更多随机探索，避免在 PR 特征空间失配时盲目使用旧技术。

因此，论文中可以这样表述：该测度提供了一种可操作的 transfer gate，用于判断 PR/D-optimal 这类老技术是否适合迁移到新的 NN 模型上。主实验说明它在原始数据集上接近最强对照并稳定优于随机/Latin；stress benchmark 说明当 D-optimal 失效时，测度加权策略能降低错误迁移风险。

## 生成文件

- 代码：`measure_weighted_sampling_experiment.py`
- 主实验结果：`measure_weighted_sampling_paper_p20_raw.csv`
- 主实验汇总：`measure_weighted_sampling_paper_p20_summary.csv`
- 主实验报告：`measure_weighted_sampling_paper_p20_report.md`
- Stress 结果：`measure_weighted_sampling_stress_p20_raw.csv`
- Stress 汇总：`measure_weighted_sampling_stress_p20_summary.csv`
- Stress 报告：`measure_weighted_sampling_stress_p20_report.md`
