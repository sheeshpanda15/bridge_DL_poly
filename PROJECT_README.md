# Bridge DL Poly Experiments

This repository contains experiments and reports comparing neural networks with
polynomial-regression views, Taylor-PR/TYPR approximations, and
distance-weighted D-optimal sampling.

## Current Result

The latest main experiment uses the distance between an initially trained NN and
PR/Taylor-PR models as a transfer gate:

- close to PR/TYPR: use more D-optimal points in the next batch
- far from PR/TYPR: keep more random exploration
- compare against random sampling, all-D-optimal sampling, and Latin hypercube

The grand benchmark uses `n=10000`, `p=20` and `p=50`, five dataset families,
five data seeds, two pilot seeds, four sampling strategies, and six batch
updates with 500 new points per batch.

## Directory Layout

- `reports/en/`: English Word/PDF reports suitable for sharing.
- `reports/zh/`: Chinese Word/PDF reports.
- `reports/notes/`: compact Markdown summaries and technical notes.
- `reports/archive/`: older source reports kept for reference.
- `figures/`: generated plots, grouped by experiment.
- `data/`: final and supporting CSV outputs, grouped by experiment.
- root `*.py`: experiment, analysis, and report-generation scripts.

## Reports To Share

- `reports/en/01_Main_Report_Measure_Weighted_Grand_Update.docx`
- `reports/en/01_Main_Report_Measure_Weighted_Grand_Update.pdf`
- `reports/en/02_Background_Report_NN_Polynomial_Geometry.docx`
- `reports/en/02_Background_Report_NN_Polynomial_Geometry.pdf`
- `reports/en/03_Technical_Supplement_Highdim_Doptimal.docx`
- `reports/en/03_Technical_Supplement_Highdim_Doptimal.pdf`

## Final Measure-Weighted Outputs

Reports:

- `reports/notes/measure_weighted_grand_mean_report.md`
- `reports/zh/大报告_中文_完整版_测度加权大实验更新版.docx`
- `reports/zh/大报告_中文_完整版_测度加权大实验更新版.pdf`
- `reports/en/01_Main_Report_Measure_Weighted_Grand_Update.docx`
- `reports/en/01_Main_Report_Measure_Weighted_Grand_Update.pdf`

CSV data:

- `data/measure_weighted/measure_weighted_grand_mean_raw_combined.csv`
- `data/measure_weighted/measure_weighted_grand_mean_final_combined.csv`
- `data/measure_weighted/measure_weighted_grand_mean_distances_combined.csv`
- `data/measure_weighted/measure_weighted_grand_mean_strategy_summary.csv`
- `data/measure_weighted/measure_weighted_grand_mean_case_summary.csv`
- `data/measure_weighted/measure_weighted_grand_mean_pairwise.csv`

Figures:

- `figures/measure_weighted/measure_weighted_grand_gain_by_p.png`
- `figures/measure_weighted/measure_weighted_grand_learning_gain.png`
- `figures/measure_weighted/measure_weighted_grand_case_gain_heatmap.png`
- `figures/measure_weighted/measure_weighted_grand_weight_heatmap.png`
- `figures/measure_weighted/measure_weighted_sampling_flowchart_cn.png`
- `figures/measure_weighted/measure_weighted_sampling_flowchart_en.png`

## Reproduce The Grand Experiment

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run p=20:

```powershell
python measure_weighted_sampling_experiment.py --out-prefix measure_weighted_sampling_grand_mean_p20 --n 10000 --p 20 --cases highdim_poly2 highdim_smooth highdim_strong highdim_local highdim_highfreq --data-seeds 0 1 2 3 4 --init-seeds 0 1 --initial-fraction 0.05 --rounds 6 --batch-size 500 --hidden 64 --epochs 350 --init-epochs 350 --shape-points 256 --device cuda --min-dopt-weight 0.30 --max-dopt-weight 0.95 --distance-combine mean
```

Run p=50:

```powershell
python measure_weighted_sampling_experiment.py --out-prefix measure_weighted_sampling_grand_mean_p50 --n 10000 --p 50 --cases highdim_poly2 highdim_smooth highdim_strong highdim_local highdim_highfreq --data-seeds 0 1 2 3 4 --init-seeds 0 1 --initial-fraction 0.05 --rounds 6 --batch-size 500 --hidden 64 --epochs 350 --init-epochs 350 --shape-points 256 --device cuda --min-dopt-weight 0.30 --max-dopt-weight 0.95 --distance-combine mean
```

Regenerate combined results and reports:

```powershell
python combine_measure_weighted_grand.py
python build_measure_weighted_word_reports.py
```

Report builders now export PDF copies next to the DOCX files. English reports
use English-only figures.

## Cleanup Policy

Keep source code, final reports, final combined CSV files, and final figures.
Do not keep runtime logs, pid files, smoke-test outputs, render folders, or
per-run sensitivity outputs unless a specific audit trail is needed.
