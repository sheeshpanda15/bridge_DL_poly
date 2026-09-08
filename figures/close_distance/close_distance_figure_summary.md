# Close-distance figure summary

## FPR/NN distance close -> performance close

- Source data: `data\base_geometry\results_raw.csv`.
- Close regime: bottom quartile of `shape_NN_FPR_in`, threshold `1.426611e-04`.
- Rows after NN non-convergence cleaning: `2966`.
- Close-regime rows: `742`.
- Median `abs_rmse_gap` in close regime: `4.815480e-03`.
- Median `abs_rmse_gap` outside close regime: `2.711887e-02`.

## Design optimization under close FPR/NN distance

- Source data: `data\measure_weighted\measure_weighted_grand_mean_final_combined.csv` and `data\measure_weighted\measure_weighted_grand_mean_raw_combined.csv`.
- Close regime: lower half of `fpr_distance`, threshold `1.261037e-02`.
- Case-level close regime for the by-case figure: lower half of mean `fpr_distance`, threshold `1.260272e-02`.

| Strategy | Mean final gain vs random | 95% CI half-width | Positive rate | Runs |
|---|---:|---:|---:|---:|
| Random | 0.000% | 0.000% | 0.0% | 50 |
| D-optimal | -1.121% | 4.964% | 52.0% | 50 |
| Latin hypercube | 0.220% | 4.705% | 50.0% | 50 |
| Measure-weighted | 1.149% | 4.170% | 56.0% | 50 |

Case/p panels used in the by-case design figure:

| p | Case | Mean FPR distance |
|---:|---|---:|
| 50 | Local interior | 2.868632e-03 |
| 50 | Quadratic | 3.810911e-03 |
| 50 | Smooth | 5.070755e-03 |
| 50 | Strong nonlinear | 7.972604e-03 |
| 50 | High-frequency | 1.114569e-02 |