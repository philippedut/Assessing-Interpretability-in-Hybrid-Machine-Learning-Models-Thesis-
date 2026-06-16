"""
Plotting and visualization helpers for the thesis experiments.

This module centralizes functions used to create, save and display figures
that summarize model performance, faithfulness, stability, sparsity and
timing across repeated runs. Keeping plotting utilities here keeps the
analysis notebooks concise and makes visual styles reusable across
experiments.

Structure
---------
- Figure saving helpers: `save_current_figure`, `_save_if_requested`.
- Global importance plotting
- Performance comparison and summary plots: `plot_performance_metric_comparison`,
  `plot_performance_summary`.
- Computational-cost visualizations: `plot_time_across_runs`,
  `plot_time_comparasion`, `plot_time_boxplot`.
- Faithfulness, stability, sparsity and metric-distribution plots.
- Training curves: `plot_train_val_accuracy`, `plot_train_val_loss`.

Usage
-----
Import the desired plotting function and pass the evaluation/result
structures produced by the metrics utilities. Example:

from Graph_Functions import plot_performance_summary
ax, saved = plot_performance_summary(mlp_metrics, nam_metrics, hybrid_metrics, save_path="figs/summary.png")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

DEFAULT_DPI = 300


def save_current_figure(path, title=None, dpi=DEFAULT_DPI, bbox_inches="tight", ext="png"):
    """
    Save the current figure at high resolution.
    """
    p = Path(path)

    safe_title = None
    if title is not None:
        safe_title = re.sub(r'[\\/*?:"<>|]+', "_", str(title)).strip()
        safe_title = safe_title or "figure"

    if p.suffix:
        filename = f"{safe_title}{p.suffix}" if safe_title else p.name
        output_path = p.with_name(filename)
    else:
        extension = ext if str(ext).startswith(".") else f".{ext}"
        filename = f"{safe_title or 'figure'}{extension}"
        output_path = p / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

    return str(output_path)


def _save_if_requested(save_path=None, save_title=None):
    """
    Save current figure only if a save path is provided.
    """
    if save_path is not None:
        return save_current_figure(save_path, title=save_title)
    return None


# =========================================================
#  Plot Global Feature Importance
# =========================================================

def plot_top_global_importance(
    global_importance,
    feature_names,
    name="Model",
    top_n=10,
    figsize=(10, 6),
    save_path=None,
    save_title=None
):
    """
    Plot the top-N most important features from contribution tensors.
    """
    feature_names = np.asarray(feature_names)

    if len(global_importance) != len(feature_names):
        raise ValueError("feature_names length must match number of features in contribs")

    n_show = min(int(top_n), len(global_importance))
    top_idx = np.argsort(global_importance)[-n_show:][::-1]

    top_features = feature_names[top_idx]
    top_scores = global_importance[top_idx]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(top_features[::-1], top_scores[::-1], color="steelblue")

    ax.set_xlabel("Global importance")
    ax.set_ylabel("Feature")

    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path


# =========================================================
#  Plot Performance Metric Comparison
# =========================================================

def plot_performance_metric_comparison(
    mlp_metrics,
    nam_metrics,
    hybrid_metrics,
    metric="accuracy",
    figsize=(8, 4),
    save_path=None,
    save_title=None
):
    """
    Plot one performance metric across models.
    """
    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]

    values = [
        mlp_metrics[metric],
        nam_metrics[metric],
        hybrid_metrics[metric],
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(models, values, color="steelblue")

    ax.set_xlabel("Model")

    if metric == "accuracy":
        ax.set_ylabel("Mean Accuracy")
        ax.set_ylim(0, 1)
    elif metric == "f1_macro":
        ax.set_ylabel("Mean Macro F1-score")
        ax.set_ylim(0, 1)
    elif metric == "logloss":
        ax.set_ylabel("Mean Log-loss")
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylabel(metric)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path


def plot_performance_summary(
    mlp_metrics,
    nam_metrics,
    hybrid_metrics,
    figsize=(8, 4),
    save_path=None,
    save_title=None
):
    """
    Plot Accuracy and Macro F1-score across models.
    """
    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]

    accuracy = [
        mlp_metrics["accuracy"],
        nam_metrics["accuracy"],
        hybrid_metrics["accuracy"],
    ]

    f1_macro = [
        mlp_metrics["f1_macro"],
        nam_metrics["f1_macro"],
        hybrid_metrics["f1_macro"],
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - width / 2, accuracy, width, color="steelblue", label="Accuracy")
    ax.bar(x + width / 2, f1_macro, width, color="orange", label="Macro F1-score")

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 1)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path


# =========================================================
#  Computational Cost Graphs
# =========================================================

def plot_time_across_runs(
    model_results,
    metric="train",
    figsize=(10, 6),
    save_path=None,
    save_title=None
):
    """
    Plot per-run timing curves for multiple models.
    """
    if metric not in {"train", "predict"}:
        raise ValueError("metric must be either 'train' or 'predict'")

    time_key = "train_times" if metric == "train" else "predict_times"
    y_label = "Time (m)"

    fig, ax = plt.subplots(figsize=figsize)

    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]
    
    color_map = {
        "MLP": "tab:blue",
        "NAM": "tab:orange",
        "Hybrid": "tab:green"
    }

    for model_name, results in model_results.items():
        if time_key not in results:
            raise KeyError(f"'{time_key}' not found in results for model '{model_name}'")
    
        times = results[time_key]
        runs = np.arange(1, len(times) + 1)

        if model_name == "MLP":
            display_name = "Post-Black_Box"
        elif model_name == "NAM":
            display_name = "End-to-End"
        elif model_name in ["Hybrid", "LogReg+MLP"]:
            display_name = "Pre-Black-box"
        else:
            display_name = model_name

        ax.plot(
            runs,
            times,
            marker="o",
            linewidth=2,
            label=display_name,
            color=color_map.get(model_name, None)
        )

    ax.set_xlabel("Run")
    ax.set_ylabel(y_label)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path


def plot_time_comparasion(
    model_results,
    metric="train",
    figsize=(10, 6),
    save_path=None,
    save_title=None
):
    """
    Plot average timing per model.
    """
    if metric not in {"train", "predict"}:
        raise ValueError("metric must be either 'train' or 'predict'")

    mean_key = "train_time_mean" if metric == "train" else "predict_time_mean"
    y_label = "Time (m)"

    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]
    means = []

    for model_name, results in model_results.items():
        if mean_key not in results:
            raise KeyError(f"'{mean_key}' not found in results for model '{model_name}'")

        means.append(results[mean_key])

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))

    ax.bar(x, means, color="steelblue")

    ax.set_xlabel("Model")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(bottom=0)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path


def plot_time_boxplot(
    model_results,
    metric="train",
    figsize=(8, 4),
    save_path=None,
    save_title=None
):
    """
    Plot distribution of timing across runs using boxplots.
    """
    if metric not in {"train", "predict"}:
        raise ValueError("metric must be either 'train' or 'predict'")

    time_key = "train_times" if metric == "train" else "predict_times"

    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]
    data = []

    for model_name, results in model_results.items():
        if time_key not in results:
            raise KeyError(f"'{time_key}' not found in results for model '{model_name}'")

        data.append(results[time_key])

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        labels=models,
        patch_artist=True,
        widths=0.5,
        showfliers=True
    )

    colors = ["steelblue", "orange", "green"]

    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Time (m)")
    ax.set_ylim(bottom=0)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path


# =========================================================
#  Faithfulness
# =========================================================

def plot_faithfulness_curves(
    results_dict,
    curve_key="mean_drop_curve",
    ylabel="Value",
    xlabel="Number of top features removed",
    figsize=(10, 6),
    linewidth=2,
    save_path=None,
    save_title=None
):
    """
    Plot a selected faithfulness curve for multiple models.
    """
    fig, ax = plt.subplots(figsize=figsize)

    color_map = {
        "MLP": "tab:blue",
        "NAM": "tab:orange",
        "Hybrid": "tab:green"
    }

    for model_name, results in results_dict.items():
        if curve_key not in results:
            raise ValueError(f"'{curve_key}' not found in results for model '{model_name}'")

        if model_name == "MLP":
            display_name = "Post-Black_Box"
        elif model_name == "NAM":
            display_name = "End-to-End"
        elif model_name in ["Hybrid", "LogReg+MLP"]:
            display_name = "Pre-Black-box"
        else:
            display_name = model_name

        x = results["k_values"]
        y = results[curve_key]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=linewidth,
            label=display_name,
            color=color_map.get(model_name, None)
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path


def plot_faithfulness_comparison(
    mlp_result,
    nam_result,
    hybrid_result,
    figsize=(10, 6),
    save_path=None,
    save_title=None
):
    """
    Plot faithfulness comparison across models.
    """
    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]

    faithfulness_scores = [
        mlp_result["faithfulness"],
        nam_result["faithfulness"],
        hybrid_result["faithfulness"],
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(models, faithfulness_scores, color="steelblue")

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Faithfulness")
    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path

def plot_faithfulness_boxplot(
    mlp_result,
    nam_result,
    hybrid_result,
    figsize=(8, 4),
    save_path=None,
    save_title=None
):
    """
    Plot the distribution of faithfulness scores across repeated runs.

    This function uses the individual run results stored inside each
    faithfulness dictionary. It shows how much the faithfulness score changes
    across the repeated model runs.
    """

    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]

    def extract_faithfulness_values(result):
        if "run_results" not in result:
            raise KeyError(
                "'run_results' not found. Make sure the repeated faithfulness "
                "wrapper returns the individual run results."
            )

        values = [
            run_result["faithfulness"]
            for run_result in result["run_results"]
        ]

        return np.asarray(values, dtype=float)

    data = [
        extract_faithfulness_values(mlp_result),
        extract_faithfulness_values(nam_result),
        extract_faithfulness_values(hybrid_result),
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        labels=models,
        patch_artist=True,
        widths=0.5,
        showfliers=True
    )

    colors = ["steelblue", "orange", "green"]

    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Faithfulness")
    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path

# =========================================================
#  Stability
# =========================================================

def plot_stability_comparison(
    mlp_result,
    nam_result,
    hybrid_result,
    figsize=(10, 6),
    save_path=None,
    save_title=None
):
    """
    Plot stability comparison across models.
    """
    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]

    stability_scores = [
        mlp_result["stability"],
        nam_result["stability"],
        hybrid_result["stability"],
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(models, stability_scores, color="steelblue")

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Stability")
    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path

def plot_stability_boxplot(
    mlp_result,
    nam_result,
    hybrid_result,
    figsize=(8, 4),
    save_path=None,
    save_title=None
):
    """
    Plot the distribution of stability scores across repeated runs.

    This function uses the individual run results stored inside each stability
    dictionary. It shows how much the stability score changes across the
    repeated model runs.
    """

    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]

    def extract_stability_values(result):
        if "run_results" not in result:
            raise KeyError(
                "'run_results' not found. Make sure the repeated stability "
                "wrapper returns the individual run results."
            )

        values = [
            run_result["stability"]
            for run_result in result["run_results"]
        ]

        return np.asarray(values, dtype=float)

    data = [
        extract_stability_values(mlp_result),
        extract_stability_values(nam_result),
        extract_stability_values(hybrid_result),
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        labels=models,
        patch_artist=True,
        widths=0.5,
        showfliers=True
    )

    colors = ["steelblue", "orange", "green"]

    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Stability")
    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path


# =========================================================
#  Sparsity
# =========================================================

def plot_sparsity_comparison(
    mlp_result,
    nam_result,
    hybrid_result,
    figsize=(8, 4),
    save_path=None,
    save_title=None
):
    """
    Plot mean and global sparsity across models.
    """
    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]

    mean_sparsity = [
        mlp_result["sparsity"],
        nam_result["sparsity"],
        hybrid_result["sparsity"],
    ]

    global_sparsity = [
        mlp_result["global_sparsity"],
        nam_result["global_sparsity"],
        hybrid_result["global_sparsity"],
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - width / 2, mean_sparsity, width, color="steelblue", label="Mean sparsity")
    ax.bar(x + width / 2, global_sparsity, width, color="orange", label="Global sparsity")

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Sparsity")
    ax.set_ylim(0, 1)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path

def plot_sparsity_boxplot(
    mlp_result,
    nam_result,
    hybrid_result,
    metric="sparsity",
    figsize=(8, 4),
    save_path=None,
    save_title=None
):
    """
    Plot the distribution of sparsity scores across repeated runs.

    This function uses the individual run results stored inside each sparsity
    dictionary. It can plot either local sparsity or global sparsity across the
    repeated model runs.
    """

    if metric not in {"sparsity", "global_sparsity"}:
        raise ValueError("metric must be either 'sparsity' or 'global_sparsity'")

    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]

    def extract_sparsity_values(result):
        if "run_results" not in result:
            raise KeyError(
                "'run_results' not found. Make sure the repeated sparsity "
                "wrapper returns the individual run results."
            )

        values = [
            run_result[metric]
            for run_result in result["run_results"]
        ]

        return np.asarray(values, dtype=float)

    data = [
        extract_sparsity_values(mlp_result),
        extract_sparsity_values(nam_result),
        extract_sparsity_values(hybrid_result),
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        labels=models,
        patch_artist=True,
        widths=0.5,
        showfliers=True
    )

    colors = ["steelblue", "orange", "green"]

    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xlabel("Model")

    if metric == "sparsity":
        ax.set_ylabel("Local sparsity")
    else:
        ax.set_ylabel("Global sparsity")

    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path

# =========================================================
#  Metric distribution boxplot
# =========================================================

def plot_metric_boxplot(
    mlp_values,
    nam_values,
    hybrid_values,
    ylabel="Value",
    figsize=(10, 6),
    save_path=None,
    save_title=None
):
    """
    Plot the distribution of a per-sample metric across models.
    """
    models = ["Post-Black-Box", "End-to-End", "Pre-Black-Box"]

    data = [
        np.asarray(mlp_values),
        np.asarray(nam_values),
        np.asarray(hybrid_values),
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        labels=models,
        patch_artist=True,
        widths=0.5,
        showfliers=True
    )

    colors = ["steelblue", "orange", "green"]

    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path


# =========================================================
#  Overfitting Curves
# =========================================================

def plot_train_val_accuracy(
    history,
    model_name="Model",
    figsize=(10, 6),
    save_path=None,
    save_title=None
):
    """
    Plot training and validation accuracy across epochs.

    For the hybrid model, this should be interpreted as the MLP component
    training history, not the full hybrid decision rule.
    """
    hist = history.history

    if "acc" in hist:
        train_key = "acc"
    elif "accuracy" in hist:
        train_key = "accuracy"
    else:
        raise KeyError("Training accuracy not found in history. Expected 'acc' or 'accuracy'.")

    if "val_acc" in hist:
        val_key = "val_acc"
    elif "val_accuracy" in hist:
        val_key = "val_accuracy"
    else:
        raise KeyError("Validation accuracy not found in history. Expected 'val_acc' or 'val_accuracy'.")

    epochs = np.arange(1, len(hist[train_key]) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        epochs,
        hist[train_key],
        marker="o",
        linewidth=2,
        label="Training accuracy",
        color="steelblue"
    )

    ax.plot(
        epochs,
        hist[val_key],
        marker="o",
        linewidth=2,
        label="Validation accuracy",
        color="orange"
    )

    if "val_loss" in hist:
        best_epoch = int(np.argmin(hist["val_loss"])) + 1
        ax.axvline(
            best_epoch,
            linestyle="--",
            linewidth=1.5,
            color="black",
            label=f"Best validation loss epoch: {best_epoch}"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path

def plot_train_val_loss(
    history,
    model_name="Model",
    figsize=(10, 6),
    save_path=None,
    save_title=None
):
    """
    Plot training and validation loss across epochs.

    For the hybrid model, this should be interpreted as the MLP component
    training history, not the full hybrid decision rule.
    """
    hist = history.history

    if "loss" not in hist or "val_loss" not in hist:
        raise KeyError("Loss history must contain 'loss' and 'val_loss'.")

    epochs = np.arange(1, len(hist["loss"]) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        epochs,
        hist["loss"],
        marker="o",
        linewidth=2,
        label="Training loss",
        color="steelblue"
    )

    ax.plot(
        epochs,
        hist["val_loss"],
        marker="o",
        linewidth=2,
        label="Validation loss",
        color="orange"
    )

    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    ax.axvline(
        best_epoch,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label=f"Best validation loss epoch: {best_epoch}"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    saved_path = _save_if_requested(save_path, save_title)
    plt.show()

    return ax, saved_path


def plot_model_scoreboard_radar_separate(
    results_df,
    model_col="Model",
    exclude_keywords=("time", "std", "sd", "transparency","overfitting"),
    figsize=(10, 8),
    save_path=None,
    save_title=None
):
    """
    Plot one radar scoreboard per model.

    The function keeps only numeric mean metrics between 0 and 1. Time-related
    metrics, standard deviations, and transparency metrics are excluded because
    they are either on a different scale or do not apply to all models.
    """

    df = results_df.copy()

    # ---------- select metric columns ----------
    metric_cols = []

    for col in df.columns:
        if col == model_col:
            continue

        col_lower = col.lower()

        if any(keyword in col_lower for keyword in exclude_keywords):
            continue

        if not col_lower.endswith("_mean"):
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if values.notna().all() and values.between(0, 1).all():
            metric_cols.append(col)

    if len(metric_cols) == 0:
        raise ValueError("No valid metric columns found for the radar plot.")

    # ---------- clean metric labels ----------
    metric_labels = [
        col.replace("_mean", "")
           .replace("_", " ")
           .title()
        for col in metric_cols
    ]

    n_metrics = len(metric_cols)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    colors = ["steelblue", "orange", "green"]

    saved_paths = []

    # ---------- one graph per model ----------
    for i, (_, row) in enumerate(df.iterrows()):

        model_name = row[model_col]

        values = row[metric_cols].astype(float).values
        values = np.concatenate([values, [values[0]]])

        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={"projection": "polar"}
        )

        ax.plot(
            angles,
            values,
            linewidth=2,
            color=colors[i % len(colors)]
        )

        ax.fill(
            angles,
            values,
            alpha=0.20,
            color=colors[i % len(colors)]
        )

        # ---------- metric labels outside plot ----------
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)

        ax.tick_params(axis="x", pad=18)

        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])

        ax.grid(True, alpha=0.3)

        # no title, thesis style
        plt.tight_layout()

        if save_path is not None:
            if save_title is None:
                title = f"scoreboard_radar_{model_name}"
            else:
                title = f"{save_title}_{model_name}"

            saved_path = _save_if_requested(save_path, title)
            saved_paths.append(saved_path)
        else:
            saved_paths.append(None)

        plt.show()

    return saved_paths