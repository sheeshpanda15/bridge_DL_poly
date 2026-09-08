# Close-distance D-optimal transfer experiment

## Question

When NN and FullPR are close on the same in-domain region used for candidate selection and evaluation, does FullPR-based D-optimal design approximate the NN oracle better than random sampling?

## Close filter

- Close threshold: `shape_nn_fpr <= 5.000e-04`.
- Used rows: `16` budget-level rows.
- Skipped non-close base runs: `4`.

- Median NN-FPR shape distance among used base runs: `1.307e-04`.
- Median NN/FPR RMSE gap among used base runs: `6.298e-03`.

## Aggregate gains

| Case | Budget x params | D-opt gain mean +/-95%CI | Latin gain mean +/-95%CI | Reps |
|---|---:|---:|---:|---:|
| Poly3 | 1.0 | 56.01% +/- 82.85% | 62.78% +/- 7.66% | 2 |
| Poly3 | 2.0 | 72.47% +/- 6.70% | 70.44% +/- 8.61% | 2 |
| Poly3 | 3.0 | 49.80% +/- 19.83% | 42.29% +/- 0.63% | 2 |
| Poly3 | 4.0 | 37.53% +/- 10.19% | 43.62% +/- 13.95% | 2 |
| Poly4 | 1.0 | -40.83% +/- 55.04% | -1142.19% +/- 2428.16% | 2 |
| Poly4 | 2.0 | 54.09% +/- 25.50% | 46.19% +/- 25.13% | 2 |
| Poly4 | 3.0 | 36.61% +/- 10.31% | 24.12% +/- 28.00% | 2 |
| Poly4 | 4.0 | 28.43% +/- 13.02% | 17.54% +/- 20.06% | 2 |

## Figures

- `figures\close_distance_dopt\close_distance_dopt_test_close_filter.png`
- `figures\close_distance_dopt\close_distance_dopt_test_gain_by_budget.png`
- `figures\close_distance_dopt\close_distance_dopt_test_mse_by_budget.png`

## Interpretation

A positive D-optimal gain means that the surrogate fitted from D-optimal NN queries has lower MSE against the NN oracle than the same-budget random-design median. Because candidates are drawn from the training-data region and evaluation is held-out in the same distribution, this experiment directly targets the close-distance transfer claim.