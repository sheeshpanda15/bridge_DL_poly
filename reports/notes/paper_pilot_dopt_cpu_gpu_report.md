# CPU/GPU Pilot-D-optimal 论文级实验报告

## 实验目的

真实应用中无法预先知道 NN 与 FullPR 的距离。因此这里采用实际可执行的 pilot 流程：先从完整数据集中均匀抽样得到小子集，在该子集上训练初始 NN 和 FullPR，估计二者的 shape distance，再用这个距离和同预算随机设计作为诊断，判断是否值得把基于 FullPR 特征空间的 D-optimal 采样迁移到 NN 模型上。

## 实验设置

- 数据集：paper_poly3、paper_poly4、smooth_nonlinear、smooth_nonlinear_rand。
- pilot 比例：1/10、1/100、1/200，即 0.1、0.01、0.005。
- 完整数据规模：每个数据集 10000 个点。
- NN 结构：hidden=(16, 8, 4)，训练 1500 epochs。
- D-optimal 候选池：5000 个点；评估集：2000 个点。
- 随机设计基准：每个 case/fraction/budget 重复 100 次，报告中使用随机 MSE 中位数。
- 重点预算：2 倍 FullPR 参数量，即 52 个查询点。
- 每个设备实际运行：12 个 pilot NN 训练、24 个 D-optimal surrogate 拟合、2400 个随机 surrogate 拟合。
- CPU/GPU 合计：24 个 pilot NN 训练、48 个 D-optimal surrogate 拟合、4800 个随机 surrogate 拟合。
- 设备：CPU 与 GPU 均已运行；GPU 为 NVIDIA GeForce RTX 4070 Ti SUPER。

## 设备一致性

在 12 个匹配的 case/fraction 组合上，CPU 与 GPU 的 D-optimal MSE gain 皮尔逊相关为 `0.995`。GPU 与 CPU 的 gain 绝对差平均为 `3.99` 个百分点，最大差为 `24.87` 个百分点。由于 NN 训练存在设备级数值差异，个别组合会有幅度差别，但总体符号和排序基本一致。

- CPU：pilot shape distance 与 MSE gain 的 Spearman 相关为 `0.580`；9/12 个组合中 D-optimal 优于随机中位数。
- GPU：pilot shape distance 与 MSE gain 的 Spearman 相关为 `0.629`；9/12 个组合中 D-optimal 优于随机中位数。

## 主要结果

| 数据集 | pilot比例 | CPU pilot距离 | CPU gain | GPU pilot距离 | GPU gain |
|---|---:|---:|---:|---:|---:|
| paper_poly3 | 0.1 | 1.514e-05 | 33.1% | 1.603e-05 | 32.9% |
| paper_poly3 | 0.01 | 1.704e-05 | -69.1% | 1.544e-05 | -68.6% |
| paper_poly3 | 0.005 | 4.202e-04 | -36.0% | 4.203e-04 | -39.8% |
| paper_poly4 | 0.1 | 8.808e-05 | 45.8% | 8.710e-05 | 42.0% |
| paper_poly4 | 0.01 | 1.433e-04 | 53.2% | 1.605e-04 | 48.1% |
| paper_poly4 | 0.005 | 8.213e-05 | 58.4% | 1.234e-04 | 61.0% |
| smooth_nonlinear | 0.1 | 1.477e-04 | -49.8% | 1.424e-04 | -74.6% |
| smooth_nonlinear | 0.01 | 4.869e-02 | 71.3% | 4.874e-02 | 71.0% |
| smooth_nonlinear | 0.005 | 1.519e-02 | 31.3% | 1.498e-02 | 24.9% |
| smooth_nonlinear_rand | 0.1 | 8.917e-03 | 95.0% | 8.920e-03 | 94.9% |
| smooth_nonlinear_rand | 0.01 | 2.008e-02 | 94.0% | 1.977e-02 | 94.0% |
| smooth_nonlinear_rand | 0.005 | 3.902e-01 | 95.5% | 3.884e-01 | 95.5% |

## 对低距离迁移问题的回答

实验支持一个较谨慎但有用的判断：当 pilot 阶段发现 NN 与 FullPR 的距离较小，并且随机设计尚未完全饱和时，旧的 FullPR-D-optimal 采样技术可以迁移到 NN 蒸馏任务中。正例是 paper_poly4：pilot shape distance 约为 8.7e-05 到 1.6e-04，CPU/GPU 上均得到约 42% 到 61% 的 MSE 改善；paper_poly3 在 0.1 pilot 下也有约 33% 的改善。

但低距离不是充分条件。paper_poly3 在 0.01 和 0.005 pilot 下为负收益，说明在查询预算、候选域、随机基准已经较强或 pilot 估计不稳定时，D-optimal 未必占优。因此论文中不应写成“距离小必然可迁移”，而应写成“pilot 距离加随机基准可以作为迁移诊断”。

## 结论

1. CPU/GPU 两套实验给出了同向结论，说明结果不是单一设备偶然现象。
2. D-optimal 在 smooth_nonlinear_rand 上非常稳定，所有 pilot 比例下均有约 94% 到 96% 的 MSE 改善；paper_poly4 也整体为正。
3. paper_poly3 在 pilot 比例较小时为负，smooth_nonlinear 在 0.1 pilot 下也为负，说明“NN-FullPR 距离小”本身不是充分条件。
4. 更适合论文表述的结论是：pilot 流程可以作为实际应用中的可观测诊断。它能暴露 NN-FullPR 的局部关系、随机设计的稳定性以及 D-optimal 是否有迁移价值；是否采用老的 D-optimal 技术，应由 pilot 距离和小规模随机基准共同决定。

## 输出文件

- `paper_pilot_dopt_cpu_gpu_results.csv`：CPU/GPU 全部 budget 结果。
- `paper_pilot_dopt_cpu_gpu_summary.csv`：2 倍参数预算下的汇总结果。
- `paper_pilot_dopt_cpu_gpu_device_diff.csv`：CPU 与 GPU 逐组合差异。
- `paper_pilot_dopt_cpu_gpu_distance_vs_gain.png`：pilot 距离与 D-optimal gain 的关系。
- `paper_pilot_dopt_cpu_gpu_gain_by_dataset.png`：按数据集和 pilot 比例分组的 CPU/GPU gain 对比。