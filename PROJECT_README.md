# Bridge DL Poly Experiments

This repository contains experiments and reports for comparing neural networks
with polynomial-regression views, Taylor-PR/TYPR approximations, and
distance-weighted D-optimal sampling.

## What This Project Tests

The central idea is to use the distance between an initially trained NN and
PR/Taylor-PR models as a practical transfer gate:

- If the NN is close to PR/TYPR, use more D-optimal points in the next batch.
- If the NN is far from PR/TYPR, keep more random exploration.
- Compare this measure-weighted policy with random sampling, all-D-optimal
  sampling, and Latin hypercube sampling.

The latest grand experiment uses:

- `n=10000`
- `p=20` and `p=50`
- five dataset families
- five data seeds and two pilot seeds
- four sampling strategies
- six batch updates with 500 new points per batch

## Important Files

Core code:

- `measure_morala.py`: original NN/PR/Taylor distance and baseline tools.
- `iterative_highdim_dopt_experiment.py`: high-dimensional data generation and D-optimal helpers.
- `measure_weighted_sampling_experiment.py`: main measure-weighted sampling experiment.
- `combine_measure_weighted_grand.py`: combines p=20/p=50 grand experiment outputs and generates figures.
- `build_measure_weighted_word_reports.py`: builds the updated Chinese and English Word reports.

Final reports:

- `measure_weighted_grand_mean_report.md`
- `大报告_中文_完整版_测度加权大实验更新版.docx`
- `big_report_en_complete_measure_weighted_grand_update.docx`

Final combined outputs:

- `measure_weighted_grand_mean_raw_combined.csv`
- `measure_weighted_grand_mean_final_combined.csv`
- `measure_weighted_grand_mean_distances_combined.csv`
- `measure_weighted_grand_mean_strategy_summary.csv`
- `measure_weighted_grand_mean_case_summary.csv`
- `measure_weighted_grand_mean_pairwise.csv`

Final figures:

- `measure_weighted_grand_gain_by_p.png`
- `measure_weighted_grand_learning_gain.png`
- `measure_weighted_grand_case_gain_heatmap.png`
- `measure_weighted_grand_weight_heatmap.png`
- `measure_weighted_sampling_flowchart_cn.png`

## Setup On A New Computer

Clone the repository:

```powershell
git clone https://github.com/<your-user>/<your-repo>.git
cd bridge_DL_poly
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Check whether CUDA is available:

```powershell
@'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
'@ | python -
```

If CUDA is not available, use `--device cpu` instead of `--device cuda` in the
commands below.

## Reproduce The Grand Experiment

Run p=20:

```powershell
python measure_weighted_sampling_experiment.py --out-prefix measure_weighted_sampling_grand_mean_p20 --n 10000 --p 20 --cases highdim_poly2 highdim_smooth highdim_strong highdim_local highdim_highfreq --data-seeds 0 1 2 3 4 --init-seeds 0 1 --initial-fraction 0.05 --rounds 6 --batch-size 500 --hidden 64 --epochs 350 --init-epochs 350 --shape-points 256 --device cuda --min-dopt-weight 0.30 --max-dopt-weight 0.95 --distance-combine mean
```

Run p=50:

```powershell
python measure_weighted_sampling_experiment.py --out-prefix measure_weighted_sampling_grand_mean_p50 --n 10000 --p 50 --cases highdim_poly2 highdim_smooth highdim_strong highdim_local highdim_highfreq --data-seeds 0 1 2 3 4 --init-seeds 0 1 --initial-fraction 0.05 --rounds 6 --batch-size 500 --hidden 64 --epochs 350 --init-epochs 350 --shape-points 256 --device cuda --min-dopt-weight 0.30 --max-dopt-weight 0.95 --distance-combine mean
```

Combine results and regenerate figures:

```powershell
python combine_measure_weighted_grand.py
```

Regenerate Word reports:

```powershell
python build_measure_weighted_word_reports.py
```

## GitHub Upload Notes

Current files are small enough for GitHub. The largest project files are about
5 MB, well below GitHub's 100 MB hard file limit.

Recommended commit contents:

- source code
- `requirements.txt`
- final combined CSV files
- final figures
- final `.md` and `.docx` reports

Do not commit:

- `__pycache__/`
- `*.pyc`
- `*.log`
- `*.pid`
- `*.err`
- smoke-test outputs
- per-run intermediate sensitivity files unless you intentionally want the full history

Basic upload flow:

```powershell
git status
git add .gitattributes .gitignore PROJECT_README.md requirements.txt
git add *.py
git add measure_weighted_grand_mean_*.csv measure_weighted_grand_*.png measure_weighted_sampling_flowchart_cn.png
git add measure_weighted_grand_mean_report.md
git add "大报告_中文_完整版_测度加权大实验更新版.docx" "big_report_en_complete_measure_weighted_grand_update.docx"
git commit -m "Add measure-weighted sampling grand experiment"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

If your terminal has trouble typing the Chinese file name, use tab completion
after typing `git add "大报告` or add the file from GitHub Desktop.

If the remote already exists, skip `git remote add origin ...` and use:

```powershell
git push
```

## Current Main Result

The main experiment uses `mean(d_FPR, d_TYPR)` as a conservative combined
distance. The result supports the interpretation that the measure is a transfer
gate rather than a universal winner over every sampling design.

At p=20, measure-weighted sampling improves over random on average. At p=50,
the conservative mean-distance rule reduces over-transfer compared with the
earlier `min(...)` rule and makes the method more competitive with all-D-optimal
sampling.
