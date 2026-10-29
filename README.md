# Outcome-aligned signal simulation

Toy simulation illustrating how outcome-aligned signal strength affects predictive discrimination across machine-learning models. The simulation uses synthetic data and does not represent empirical clinical outcomes.

## Methods

- **Features:** 8 Gaussian features (2 informative, 6 noise)
- **Effect sizes (μ):** 0, 0.5, 1, 2
- **Sample sizes:** 64, 128, 256, 512, 1024
- **Class balance:** 1:1
- **Models:** LR, RF, SVM, GBM, MLP
- **Evaluation:** 80:20 stratified train-test split
- **Repetitions:** 100 per condition (`seed = 123`)
- **Metric:** Mean AUROC, SD, and 95% CI

## Required packages

`ggplot2`, `scico`, `pROC`, `randomForest`, `e1071`, `gbm`, `nnet`.

## Outputs

- `data/auroc_repetitions.csv` —individual simulation results
- `data/auroc_summary.csv` — mean AUROC, SD, and 95% CI
