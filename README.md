# Thesis Repository

# 🎓 Master’s Thesis in Data Science & Advanced Analytics Specialisation Data Science

## 👥 Author
Author: **Philippe Dutranoit** (20240518)

## Thesis Objective

This thesis benchmarks interpretability and hybrid-model techniques across multiple metrics to study the trade-off between predictive performance and interpretability.

The repository contains reusable model, grid-search, metric, and plotting utilities, plus experiment notebooks for each dataset.

## Experiments and Datasets

Main experiments are organized by dataset:

- Adult dataset: Adult_Experiment.ipynb
- COMPAS dataset: Compas_Experiment.ipynb
- Cover Type dataset: Cover_type_Experiment.ipynb

## Model Families Compared

Three model approaches are benchmarked. Note: in the code and results
tables the models are referred to using short names (`MLP`, `NAM`,
`Hybrid`) which correspond to the thesis terminology as follows:

- Post-Black-Box (`MLP` in files): Black-box MLP with post-hoc SHAP explanations
- End-to-End (`NAM` in files): Neural Additive Model with native feature contributions
- Pre-Black-Box (`Hybrid` in files): Logistic Regression gating combined with an MLP fallback

## Evaluation Framework

The evaluation is designed to be holistic and comparable across the three
approaches. To increase stability of the reported numbers we run each model
multiple times and report summary statistics:

1. Performance metrics (per dataset and model)
- Accuracy
- Log-loss
- F1 macro

For performance and interpretability metrics we run each model five times and
report the mean and standard deviation across the five runs (mean ± std). This
provides a more robust point estimate than a single-run measurement.

2. Computational cost
- Training time and prediction time are measured with repeated runs (10
	repetitions) and reported as mean and standard deviation. Time metrics are
	reported in minutes.

3. Interpretability metrics
- Faithfulness
- Stability
- Sparsity

Results are consolidated in a single comparison dataframe in the notebooks and
include mean ± std columns for the repeated-run metrics.

## Functions files

- Models_Functions.py: model architectures, training helpers, and contribution extraction
- Grid_Search_Functions.py: hyperparameter search utilities
- Metrics_Functions.py: functions helpful to benchmark models, including
	repeated-run wrappers that compute mean ± std for most metrics and a
	10-run timing benchmark for computational cost.
- Graph_Functions.py: plotting helpers for result visualization


## Typical Workflow

1. Load and preprocess a dataset.
2. Run model optimization (grid search).
3. Train final models with selected parameters.
4. Extract contributions and feature importance.
5. Compute performance, computational cost, and interpretability metrics.
6. Append all outputs to the consolidated results dataframe.
7. Visualize and compare model behavior.