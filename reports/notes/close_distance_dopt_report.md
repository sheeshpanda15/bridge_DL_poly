# Close-distance D-optimal transfer experiment

## Question

When NN and FullPR are close on the same in-domain region used for candidate selection and evaluation, does FullPR-based D-optimal design approximate the NN oracle better than random sampling?

## Close filter

- Close threshold: `shape_nn_fpr <= 5.000e-04`.
- Used rows: `64` budget-level rows.
- Skipped non-close base runs: `4`.

- Median NN-FPR shape distance among used base runs: `3.155e-05`.
- Median NN/FPR RMSE gap among used base runs: `3.513e-03`.

## Aggregate gains

| Case | Budget x params | D-opt gain mean +/-95%CI | Latin gain mean +/-95%CI | Reps |
|---|---:|---:|---:|---:|
| Poly3 | 1.0 | -94.83% +/- 227.40% | -737563.72% +/- 1445524.77% | 8 |
| Poly3 | 2.0 | 41.89% +/- 21.77% | 37.54% +/- 10.30% | 8 |
| Poly3 | 3.0 | 39.03% +/- 14.85% | 32.96% +/- 19.72% | 8 |
| Poly3 | 4.0 | 31.94% +/- 13.25% | 22.23% +/- 21.84% | 8 |
| Poly4 | 1.0 | 8.74% +/- 127.05% | -776310.90% +/- 1521666.57% | 8 |
| Poly4 | 2.0 | 52.39% +/- 17.59% | 59.76% +/- 19.97% | 8 |
| Poly4 | 3.0 | 41.13% +/- 16.25% | 31.52% +/- 17.38% | 8 |
| Poly4 | 4.0 | 32.93% +/- 14.68% | 21.44% +/- 14.52% | 8 |

## Figures

- `figures\close_distance_dopt\close_distance_dopt_close_filter.png`
- `figures\close_distance_dopt\close_distance_dopt_gain_by_budget.png`
- `figures\close_distance_dopt\close_distance_dopt_gain_by_budget_stable.png`
- `figures\close_distance_dopt\close_distance_dopt_mse_by_budget.png`

## Interpretation

A positive D-optimal gain means that the surrogate fitted from D-optimal NN queries has lower MSE against the NN oracle than the same-budget random-design median. Because candidates are drawn from the training-data region and evaluation is held-out in the same distribution, this experiment directly targets the close-distance transfer claim.