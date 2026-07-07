# Grand measure-weighted sampling experiment

## Scope

This grand benchmark uses n=10000, p in {20, 50}, five dataset families, five data seeds, two initial sampling seeds, four sampling strategies, six batch updates, and a fixed batch size of 500. The main version uses the conservative combined distance mean(d_FPR, d_TYPR) to decide the D-optimal fraction.

## Main summary by dimension

| p_dim | strategy_label | final_mse_mean | gain_vs_random_mean | positive_gain_rate | runs |
| --- | --- | --- | --- | --- | --- |
| 20 | Latin hypercube | 0.004811 | 5.791% | 76.0% | 50 |
| 20 | D-optimal | 0.004986 | 4.110% | 76.0% | 50 |
| 20 | Measure-weighted | 0.004993 | 4.397% | 72.0% | 50 |
| 20 | Random | 0.005113 | 0.000% | 0.0% | 50 |
| 50 | Latin hypercube | 0.018574 | 0.168% | 46.0% | 50 |
| 50 | Random | 0.018808 | 0.000% | 0.0% | 50 |
| 50 | Measure-weighted | 0.019143 | 0.501% | 54.0% | 50 |
| 50 | D-optimal | 0.019594 | -1.846% | 48.0% | 50 |

## Measure-weighted pairwise comparisons

| p_dim | comparison | win_rate | mean_pct_gain | runs |
| --- | --- | --- | --- | --- |
| 20 | measure_weighted_vs_random | 72.0% | 4.397% | 50 |
| 50 | measure_weighted_vs_random | 54.0% | 0.501% | 50 |
| all | measure_weighted_vs_random | 63.0% | 2.449% | 100 |
| 20 | measure_weighted_vs_dopt | 56.0% | -0.243% | 50 |
| 50 | measure_weighted_vs_dopt | 52.0% | 0.413% | 50 |
| all | measure_weighted_vs_dopt | 54.0% | 0.085% | 100 |
| 20 | measure_weighted_vs_latin | 46.0% | -1.754% | 50 |
| 50 | measure_weighted_vs_latin | 40.0% | -1.384% | 50 |
| all | measure_weighted_vs_latin | 43.0% | -1.569% | 100 |

## Measure-weighted gains by case

| p_dim | case_label | final_mse_mean | gain_vs_random_mean | positive_gain_rate |
| --- | --- | --- | --- | --- |
| 20 | High-frequency | 0.010106 | -9.524% | 40.0% |
| 20 | Local interior | 0.003893 | 5.824% | 80.0% |
| 20 | Quadratic | 0.007036 | 8.241% | 90.0% |
| 20 | Smooth | 0.002244 | 8.964% | 90.0% |
| 20 | Strong nonlinear | 0.001685 | 8.482% | 60.0% |
| 50 | High-frequency | 0.022579 | 0.238% | 60.0% |
| 50 | Local interior | 0.008042 | 13.552% | 80.0% |
| 50 | Quadratic | 0.049157 | -8.046% | 30.0% |
| 50 | Smooth | 0.010861 | -2.908% | 30.0% |
| 50 | Strong nonlinear | 0.005076 | -0.331% | 70.0% |

## Weight calibration

| p_dim | case_label | fpr_distance_mean | tpr_distance_mean | dopt_weight_mean |
| --- | --- | --- | --- | --- |
| 20 | High-frequency | 0.22958 | 0.94323 | 0.323 |
| 20 | Local interior | 0.08290 | 0.90947 | 0.551 |
| 20 | Quadratic | 0.04888 | 0.92314 | 0.579 |
| 20 | Smooth | 0.01406 | 0.78233 | 0.779 |
| 20 | Strong nonlinear | 0.01903 | 0.80521 | 0.749 |
| 50 | High-frequency | 0.01115 | 0.87244 | 0.377 |
| 50 | Local interior | 0.00287 | 0.69602 | 0.545 |
| 50 | Quadratic | 0.00381 | 0.67319 | 0.578 |
| 50 | Smooth | 0.00507 | 0.53331 | 0.694 |
| 50 | Strong nonlinear | 0.00797 | 0.42922 | 0.815 |

## Sensitivity note

The earlier min(d_FPR, d_TYPR) rule was intentionally less conservative. In p=50 it caused negative transfer for the measure-weighted policy because a single small PR distance could trigger too much D-optimal sampling. The mean-distance rule reduces that over-transfer.

| p_dim | strategy_label | final_mse_mean | gain_vs_random_mean | positive_gain_rate |
| --- | --- | --- | --- | --- |
| 20 | Measure-weighted | 0.004993 | 4.231% | 74.0% |
| 50 | Measure-weighted | 0.019771 | -2.821% | 46.0% |

## Figures

- measure_weighted_grand_gain_by_p.png
- measure_weighted_grand_learning_gain.png
- measure_weighted_grand_case_gain_heatmap.png
- measure_weighted_grand_weight_heatmap.png
- measure_weighted_sampling_flowchart_cn.png

## Interpretation

The measure is most useful as a transfer gate rather than as a universal winner over every control. It improves over random in p=20 and becomes competitive in p=50 after conservative distance aggregation. In high-frequency or high-dimensional settings, the learned weight exposes when old PR/D-optimal technology should be used cautiously.
