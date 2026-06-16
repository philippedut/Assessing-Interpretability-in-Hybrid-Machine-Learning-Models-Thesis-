"""
Metrics functions for evaluation and explanation analysis.

This module provides utilities to measure model performance, computational
cost, faithfulness of explanations, and stability under perturbations for
tabular classification models used in the thesis experiments.

Structure
---------
- Benchmarking: training and prediction timing functions (e.g. `benchmark_training_time`,
    `benchmark_prediction_time`).
- Performance: `compute_performance_metrics` for classification metrics.
- Overfitting: single-run and repeated overfitting-gap computations
    (`compute_overfitting_metrics`, `compute_overfitting_metrics_repeated`,
    hybrid variants).
- Faithfulness: deletion-based faithfulness measures
    (`compute_faithfulness_adup` and repeated wrapper).
- Stability: explanation stability under input perturbations
    (`compute_stability` and repeated wrapper).

Usage
-----
- Import the module or specific functions and pass model prediction/explanation
    callables so the utilities work with different model wrappers.

Example
-------
from Metrics_Functions import compute_performance_metrics
metrics = compute_performance_metrics(y_true, lambda: model.predict(X), model_name="MyModel")
"""


import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss, f1_score

from Models_Functions import get_local_importance, hybrid_predict

# =========================================================
# Computational cost metrics using time
# =========================================================
def benchmark_training_time(
    fit_fn,
    n_runs=10,
    random_seeds=None,
):
    """
    Measure how long a model takes to train over repeated runs.

    The purpose of this metric is to compare the practical cost of different
    model choices under the same setup. Repeating the measurement several
    times makes the result less sensitive to random initialization and runtime
    noise, so the reported time is a more stable estimate of training cost.

    Parameters
    ----------
    fit_fn : callable
        Zero-argument training function such as:
        - fit_blackbox_mlp
        - fit_nam
        - fit_hybrid_logreg_mlp
        In practice, pass a lambda or wrapper that already captures the
        training data and hyperparameters.
    Returns
    -------
    results : dict
        Dictionary containing mean/std training times for fair comparison.
    """
    time_divisor = 60.0
    time_suffix = "min"

    train_times = []

    if random_seeds is None:
        random_seeds = list(range(n_runs))

    for i in range(n_runs):
        seed = random_seeds[i]

        keras.backend.clear_session()

        tf.keras.utils.set_random_seed(seed)

        start = time.perf_counter()
        _ = fit_fn()
        end = time.perf_counter()

        elapsed = (end - start) / time_divisor
        train_times.append(elapsed)
        print(f"Run {i+1}/{n_runs} | train_time={elapsed:.4f}{time_suffix}")

    results = {
        "train_time_mean": float(np.mean(train_times)),
        "train_time_std": float(np.std(train_times)),
        "train_times": train_times,
    }

    return results

def benchmark_prediction_time(
    predict_fn,
    n_runs=10,
    random_seeds=None,
):
    """
    Measure how long a trained model takes to produce predictions.

    The goal is to capture inference cost, which is important when a model
    must run under latency or throughput constraints. A model can be accurate
    and still be expensive to deploy if predictions are slow, so this metric
    helps compare practical usability across architectures. Repeating the
    timing several times gives a more stable estimate.

    Parameters
    ----------
    predict_fn : callable
        Zero-argument prediction function such as:
        - lambda X: nam_model.predict(X, verbose=0)
        - lambda X: bb_model.predict(X, verbose=0)
        - lambda X: hybrid_predict(logreg_model, bb_model, X, tau=TAU)[0]
        Pass a wrapper that already captures the input data.

    n_runs : int
        Number of repeated runs.

    Returns
    -------
    results : dict
        Dictionary containing mean/std prediction times for fair comparison.
    """
    time_divisor = 60.0
    time_suffix = "min"

    pred_times = []

    if random_seeds is None:
        random_seeds = list(range(n_runs))

    for i in range(n_runs):
        seed = random_seeds[i]
        tf.keras.utils.set_random_seed(seed)

        start = time.perf_counter()
        _ = predict_fn()
        end = time.perf_counter()

        elapsed = (end - start) / time_divisor
        pred_times.append(elapsed)

        print(f"Run {i+1}/{n_runs} | predict_time={elapsed:.4f}{time_suffix}")

    results = {
        "predict_time_mean": float(np.mean(pred_times)),
        "predict_time_std": float(np.std(pred_times)),
        "predict_times": pred_times,
    }

    return results

# =========================================================
# Performance metrics
# =========================================================
def compute_performance_metrics(y_true, predict_fn, model_name="Model"):
    """
    Compute classification metrics.

    Works with:
    - one prediction output, as before
    - several prediction outputs, e.g. 5 runs

    Returns
    -------
    metrics : dict
        Same structure as before, with the addition of mean and sd metrics
        when several runs are provided.
    """

    pred_output = predict_fn()

    # If predict_fn returns a single model output, keep the old behavior
    if not isinstance(pred_output, (list, tuple)):
        pred_outputs = [pred_output]
    else:
        pred_outputs = pred_output

    all_accuracies = []
    all_f1_macros = []
    all_loglosses = []

    for output in pred_outputs:

        output = np.asarray(output)

        # Extract hard predictions and probabilities
        if output.ndim == 2:  # softmax/probability output (N, K)
            y_pred = np.argmax(output, axis=1)
            y_pred_proba = output
        else:  # hard predictions (N,)
            y_pred = output
            y_pred_proba = None

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")

        logloss = None
        if y_pred_proba is not None:
            logloss = log_loss(y_true, y_pred_proba)

        all_accuracies.append(accuracy)
        all_f1_macros.append(f1_macro)

        if logloss is not None:
            all_loglosses.append(logloss)

    # Keep the original point estimate behavior:
    # for one model, this is the only run;
    # for several models, this is the mean.
    accuracy_mean = np.mean(all_accuracies)
    f1_macro_mean = np.mean(all_f1_macros)

    accuracy_sd = np.std(all_accuracies, ddof=1) if len(all_accuracies) > 1 else 0.0
    f1_macro_sd = np.std(all_f1_macros, ddof=1) if len(all_f1_macros) > 1 else 0.0

    if len(all_loglosses) > 0:
        logloss_mean = np.mean(all_loglosses)
        logloss_sd = np.std(all_loglosses, ddof=1) if len(all_loglosses) > 1 else 0.0
    else:
        logloss_mean = None
        logloss_sd = None

    metrics = {
        "model_name": model_name,

        # Same original keys
        "accuracy": float(accuracy_mean),
        "f1_macro": float(f1_macro_mean),
        "logloss": float(logloss_mean) if logloss_mean is not None else None,

        # Added keys
        "accuracy_mean": float(accuracy_mean),
        "accuracy_sd": float(accuracy_sd),
        "f1_macro_mean": float(f1_macro_mean),
        "f1_macro_sd": float(f1_macro_sd),
        "logloss_mean": float(logloss_mean) if logloss_mean is not None else None,
        "logloss_sd": float(logloss_sd) if logloss_sd is not None else None,
    }

    print(f"{model_name} Performance Metrics:")
    print(f"  Accuracy: {accuracy_mean:.4f} ± {accuracy_sd:.4f}")
    print(f"  F1 (macro): {f1_macro_mean:.4f} ± {f1_macro_sd:.4f}")

    if logloss_mean is not None:
        print(f"  Log-loss: {logloss_mean:.4f} ± {logloss_sd:.4f}")

    return metrics

# =========================================================
# Overfitting gap
# =========================================================
def compute_overfitting_metrics(history):
    """
    Computes overfitting indicators from a Keras training history.

    Returns:
    - best_epoch: epoch with lowest validation loss
    - train_acc_best: training accuracy at best validation epoch
    - val_acc_best: validation accuracy at best validation epoch
    - accuracy_gap: train_acc_best - val_acc_best
    - train_loss_best: training loss at best validation epoch
    - val_loss_best: validation loss at best validation epoch
    - loss_gap: val_loss_best - train_loss_best
    """

    hist = history.history

    best_epoch = int(np.argmin(hist["val_loss"]))

    train_acc_best = hist["acc"][best_epoch]
    val_acc_best = hist["val_acc"][best_epoch]

    train_loss_best = hist["loss"][best_epoch]
    val_loss_best = hist["val_loss"][best_epoch]

    accuracy_gap = train_acc_best - val_acc_best
    loss_gap = val_loss_best - train_loss_best

    return {
        "best_epoch": best_epoch + 1,
        "train_accuracy": train_acc_best,
        "val_accuracy": val_acc_best,
        "accuracy_overfitting_gap": accuracy_gap,
        "train_loss": train_loss_best,
        "val_loss": val_loss_best,
        "loss_overfitting_gap": loss_gap
    }

def compute_hybrid_overfitting_gap(
    logreg_model,
    bb_model,
    X_train,
    y_train,
    X_val,
    y_val,
    tau=0.7
):
    """
    Computes overfitting indicators for the final hybrid decision rule.

    This evaluates the full Pre-Black-Box hybrid model:
    Logistic Regression is used when confidence >= tau,
    otherwise the model falls back to the MLP.
    """

    P_train, use_logreg_train, conf_train = hybrid_predict(
        logreg_model,
        bb_model,
        X_train,
        tau=tau
    )

    P_val, use_logreg_val, conf_val = hybrid_predict(
        logreg_model,
        bb_model,
        X_val,
        tau=tau
    )

    y_pred_train = np.argmax(P_train, axis=1)
    y_pred_val = np.argmax(P_val, axis=1)

    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)

    train_loss = log_loss(y_train, P_train)
    val_loss = log_loss(y_val, P_val)

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "accuracy_overfitting_gap": train_acc - val_acc,
        "train_logloss": train_loss,
        "val_logloss": val_loss,
        "logloss_overfitting_gap": val_loss - train_loss,
        "train_logreg_usage": use_logreg_train.mean(),
        "val_logreg_usage": use_logreg_val.mean(),
        "train_avg_logreg_confidence": conf_train.mean(),
        "val_avg_logreg_confidence": conf_val.mean()
    }

def compute_overfitting_metrics_repeated(history_list):
    """
    Compute overfitting metrics across repeated model runs.

    This wrapper calls compute_overfitting_metrics() once for each training
    history. Each run produces its own best epoch, train/validation accuracy,
    train/validation loss, and overfitting gaps.

    The final output keeps the same main structure as compute_overfitting_metrics(),
    but reports the average results across runs. It also adds the standard
    deviation across runs, so that the stability of the overfitting results can
    be assessed.

    Returns
    -------
    dict
        Dictionary containing the mean value of each metric across runs,
        the standard deviation of each metric across runs, and the individual
        outputs from each run.
    """

    run_results = []

    for history in history_list:
        result = compute_overfitting_metrics(history)
        run_results.append(result)

    keys = run_results[0].keys()

    summary = {}

    for key in keys:
        values = np.array([result[key] for result in run_results], dtype=float)

        summary[key] = float(np.mean(values))
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_sd"] = (
            float(np.std(values, ddof=1))
            if len(run_results) > 1
            else 0.0
        )

    summary["run_results"] = run_results

    return summary

def compute_hybrid_overfitting_gap_repeated(
    logreg_model_list,
    MLP_hybrid_list,
    X_train,
    y_train,
    X_val,
    y_val,
    tau
):
    """
    Compute hybrid overfitting metrics across repeated model runs.

    This wrapper calls compute_hybrid_overfitting_gap() once for each hybrid
    model pair. Each run produces its own train/validation performance and
    overfitting gaps.

    The final output keeps the same main structure as compute_hybrid_overfitting_gap(),
    but reports the average results across runs. It also adds the standard
    deviation across runs, so that the stability of the hybrid overfitting
    results can be assessed.

    Returns
    -------
    dict
        Dictionary containing the mean value of each metric across runs,
        the standard deviation of each metric across runs, and the individual
        outputs from each run.
    """

    if len(logreg_model_list) != len(MLP_hybrid_list):
        raise ValueError("logreg_model_list and MLP_hybrid_list must have the same length")

    run_results = []

    for logreg_model, MLP_hybrid in zip(logreg_model_list, MLP_hybrid_list):

        result = compute_hybrid_overfitting_gap(
            logreg_model,
            MLP_hybrid,
            X_train,
            y_train,
            X_val,
            y_val,
            tau=tau
        )

        run_results.append(result)

    keys = run_results[0].keys()

    summary = {}

    for key in keys:
        values = np.array([result[key] for result in run_results], dtype=float)

        summary[key] = float(np.mean(values))
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_sd"] = (
            float(np.std(values, ddof=1))
            if len(run_results) > 1
            else 0.0
        )

    summary["run_results"] = run_results

    return summary
    
# =========================================================
# Faithfulness metrics
# =========================================================
def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    elif isinstance(X, pd.DataFrame):
        return X.values
    else:
        return np.asarray(X)

def compute_faithfulness_adup(
    X,
    contribs,
    predict_fn,
    X_train,
    categorical_mask,
    max_k=10
):
    """
    Compute a deletion-based faithfulness score for tabular explanations.

    The goal is to test whether the features ranked as important by the
    explanation are truly important for the model's prediction. For each
    sample, the function ranks the features from the contribution tensor and
    then removes the top-ranked ones step by step by replacing them with a
    baseline built from the training data. Numerical features use the training
    mean, while categorical and one-hot features use the training mode.

    After each deletion step, the model is evaluated again and the confidence
    assigned to the original predicted class is tracked. This builds a
    confidence curve and a drop curve that show how quickly the model reacts
    when the explanation says important features are removed.

    The final faithfulness score is a normalized AOPC-style measure. Each
    sample's drop curve is normalized by the final drop at ``k = max_k``,
    clipped to [0, 1], averaged across deletion steps, and then averaged
    across samples. Higher faithfulness means the confidence drops earlier
    and more consistently, which indicates a better match between the
    explanation and the model's behavior.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to evaluate.
    contribs : np.ndarray of shape (n_samples, n_features, n_classes)
        Per-sample, per-feature contribution scores for each class.
    predict_fn : callable
        Function that takes an input matrix of shape (n_samples, n_features)
        and returns predicted class probabilities of shape
        (n_samples, n_classes).
    X_train : array-like of shape (n_train, n_features)
        Training data after preprocessing, used to construct the perturbation
        baseline.
    categorical_mask : array-like of bool, shape (n_features,)
        Boolean mask indicating which transformed features are categorical
        (including one-hot encoded columns). Features marked False are treated
        as numerical.
    max_k : int, default=10
        Maximum number of top-ranked features to progressively replace.

    Returns
    -------
    dict
        Dictionary containing:
        - "faithfulness": final normalized AOPC score
        - "mean_drop_curve": average confidence-drop curve across samples
        - "mean_conf_curve": average confidence curve across samples
    """
    X = _to_dense(X)
    X_train = _to_dense(X_train)
    categorical_mask = np.asarray(categorical_mask, dtype=bool)

    n_samples, n_features = X.shape
    max_k = min(max_k, n_features)

    if contribs.shape[0] != n_samples:
        raise ValueError("X and contribs must have the same number of samples")

    if len(categorical_mask) != n_features:
        raise ValueError("categorical_mask must have length equal to n_features")

    # Build baseline: mean for numeric, mode for categorical

    baseline = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        col = X_train[:, j]

        if categorical_mask[j]:
            values, counts = np.unique(col, return_counts=True)
            baseline[j] = values[np.argmax(counts)]   # mode
        else:
            baseline[j] = np.mean(col)                # mean

    # Original predictions

    P_orig = predict_fn(X)
    pred_class = np.argmax(P_orig, axis=1)
    orig_conf = P_orig[np.arange(n_samples), pred_class]

    # Rank features using local importance

    local_importance = get_local_importance(contribs)
    ranked_idx = np.argsort(local_importance, axis=1)[:, ::-1]

    # Progressive deletion curve

    X_work = X.copy()
    conf_curve = [orig_conf.copy()]

    for k in range(1, max_k + 1):
        for i in range(n_samples):
            feat = ranked_idx[i, k - 1]
            X_work[i, feat] = baseline[feat]

        P_k = predict_fn(X_work)
        conf_k = P_k[np.arange(n_samples), pred_class]
        conf_curve.append(conf_k)

    conf_curve = np.stack(conf_curve, axis=1)  # (n_samples, max_k + 1)

    # Drop curve

    drop_curve = orig_conf[:, None] - conf_curve

    # Normalized AOPC
    # Normalize by endpoint drop

    endpoint_drop = drop_curve[:, -1]
    denom = np.maximum(endpoint_drop, 1e-12)

    norm_drop_curve = drop_curve / denom[:, None]
    norm_drop_curve = np.clip(norm_drop_curve, 0, 1)

    per_sample_aopc = np.mean(norm_drop_curve[:, 1:], axis=1)  # exclude k=0
    faithfulness = float(np.mean(per_sample_aopc))

    return {
        "faithfulness": faithfulness,
        "mean_drop_curve": np.mean(drop_curve, axis=0),
        "mean_conf_curve": np.mean(conf_curve, axis=0),
        "k_values": list(range(max_k + 1))
    }

def compute_faithfulness_adup_repeated(
    X,
    contribs_list,
    predict_fn_list,
    X_train,
    categorical_mask,
    max_k=10
):
    """
    Compute faithfulness results across repeated model runs.

    This wrapper calls compute_faithfulness_adup() once for each run, using the
    corresponding contribution tensor and prediction function. Each run produces
    its own faithfulness score, confidence curve, and drop curve.

    The final output keeps the same main structure as compute_faithfulness_adup(),
    but reports the average results across runs. It also adds the standard
    deviation across runs, so that the stability of the faithfulness results can
    be assessed.

    Returns
    -------
    dict
        Dictionary containing:
        - "faithfulness": mean faithfulness score across runs
        - "mean_drop_curve": mean drop curve across runs
        - "mean_conf_curve": mean confidence curve across runs
        - "k_values": deletion steps
        - "faithfulness_sd": standard deviation of faithfulness across runs
        - "mean_drop_curve_sd": standard deviation of the drop curve across runs
        - "mean_conf_curve_sd": standard deviation of the confidence curve across runs
        - "run_results": individual outputs from each run
    """

    if len(contribs_list) != len(predict_fn_list):
        raise ValueError("contribs_list and predict_fn_list must have the same length")

    run_results = []

    for contribs, predict_fn in zip(contribs_list, predict_fn_list):

        result = compute_faithfulness_adup(
            X=X,
            contribs=contribs,
            predict_fn=predict_fn,
            X_train=X_train,
            categorical_mask=categorical_mask,
            max_k=max_k
        )

        run_results.append(result)

    faithfulness_values = np.array([
        result["faithfulness"] for result in run_results
    ])

    drop_curves = np.stack([
        result["mean_drop_curve"] for result in run_results
    ])

    conf_curves = np.stack([
        result["mean_conf_curve"] for result in run_results
    ])

    faithfulness_mean = np.mean(faithfulness_values)
    faithfulness_sd = (
        np.std(faithfulness_values, ddof=1)
        if len(run_results) > 1
        else 0.0
    )

    mean_drop_curve = np.mean(drop_curves, axis=0)
    mean_drop_curve_sd = (
        np.std(drop_curves, axis=0, ddof=1)
        if len(run_results) > 1
        else np.zeros_like(mean_drop_curve)
    )

    mean_conf_curve = np.mean(conf_curves, axis=0)
    mean_conf_curve_sd = (
        np.std(conf_curves, axis=0, ddof=1)
        if len(run_results) > 1
        else np.zeros_like(mean_conf_curve)
    )

    return {
        # same main keys as the original function
        "faithfulness": float(faithfulness_mean),
        "mean_drop_curve": mean_drop_curve,
        "mean_conf_curve": mean_conf_curve,
        "k_values": run_results[0]["k_values"],

        # added summary keys
        "faithfulness_mean": float(faithfulness_mean),
        "faithfulness_sd": float(faithfulness_sd),
        "mean_drop_curve_sd": mean_drop_curve_sd,
        "mean_conf_curve_sd": mean_conf_curve_sd,

        # optional: individual run outputs
        "run_results": run_results
    }
# =========================================================
# Stability metric 
# =========================================================
def compute_stability(
    X,
    contribs,
    explain_fn,
    numeric_mask,
    categorical_groups=None,
    n_perturbations=10,
    sigma=0.05,
    cat_flip_prob=0.05,
    random_state=42
):
    """
    Measure how consistent explanations remain when the input is perturbed.

    The goal is to check whether similar inputs receive similar explanations.
    For each sample, the function starts from the original explanation,
    generates several perturbed versions of the same input, recomputes the
    explanation on each perturbed sample, and compares the original and
    perturbed feature rankings with Spearman correlation.

    Numeric features are perturbed with small Gaussian noise, while one-hot
    categorical groups can be randomly switched with a low probability. The
    correlations are averaged over perturbations and samples to produce a
    single stability score. Higher values mean the explanation changes less
    when the input is slightly modified, which indicates a more robust and
    reliable explanation method.

    Local sparsity = per observation → “Does this prediction depend on few features?”
    Global sparsity = across dataset → “Does the model overall rely on few features?”
    """

    # ---------- convert to dense numpy ----------
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    numeric_mask = np.asarray(numeric_mask, dtype=bool)
    rng = np.random.default_rng(random_state)

    # ---------- original local importance ----------
    original_importance = get_local_importance(contribs)

    all_scores = []

    for _ in range(n_perturbations):
        X_pert = X.copy()

        # ---------- numeric perturbation ----------
        if np.any(numeric_mask):
            noise = rng.normal(0, sigma, size=(X.shape[0], numeric_mask.sum()))
            X_pert[:, numeric_mask] += noise

        # ---------- categorical perturbation ----------
        if categorical_groups is not None:
            for group in categorical_groups:
                group = np.asarray(group)

                for i in range(X.shape[0]):
                    if rng.random() < cat_flip_prob:
                        current = X_pert[i, group]
                        active = np.where(current > 0.5)[0]

                        if len(active) == 1:
                            old_pos = active[0]
                            choices = [j for j in range(len(group)) if j != old_pos]
                            new_pos = rng.choice(choices) if choices else old_pos
                        else:
                            new_pos = rng.integers(len(group))

                        X_pert[i, group] = 0.0
                        X_pert[i, group[new_pos]] = 1.0

        # ---------- perturbed explanations ----------
        pert_contribs = explain_fn(X_pert)
        pert_importance = get_local_importance(pert_contribs)

        # ---------- compare explanations ----------
        scores = []
        for i in range(X.shape[0]):
            corr, _ = spearmanr(original_importance[i], pert_importance[i])
            scores.append(0.0 if np.isnan(corr) else corr)

        all_scores.append(scores)

    all_scores = np.asarray(all_scores).T  # shape: (n_samples, n_perturbations)
    per_sample_stability = all_scores.mean(axis=1)
    stability = float(per_sample_stability.mean())

    return {
        "stability": stability,
        "per_sample_stability": per_sample_stability,
        "all_scores": all_scores
    }

def compute_stability_repeated(
    X,
    contribs_list,
    explain_fn_list,
    numeric_mask,
    categorical_groups=None,
    n_perturbations=10,
    sigma=0.05,
    cat_flip_prob=0.05,
    random_state=42
):
    """
    Compute stability results across repeated model runs.

    This wrapper calls compute_stability() once for each run, using the
    corresponding contribution tensor and explanation function. Each run
    produces its own stability score, per-sample stability values, and
    perturbation-level scores.

    The final output keeps the same main structure as compute_stability(),
    but reports the average results across runs. It also adds the standard
    deviation across runs, so that the stability of the explanation results
    can be assessed.

    Returns
    -------
    dict
        Dictionary containing:
        - "stability": mean stability score across runs
        - "per_sample_stability": mean per-sample stability across runs
        - "all_scores": mean perturbation scores across runs
        - "stability_sd": standard deviation of stability across runs
        - "per_sample_stability_sd": standard deviation of per-sample stability across runs
        - "all_scores_sd": standard deviation of perturbation scores across runs
        - "run_results": individual outputs from each run
    """

    if len(contribs_list) != len(explain_fn_list):
        raise ValueError("contribs_list and explain_fn_list must have the same length")

    run_results = []

    for run_idx, (contribs, explain_fn) in enumerate(zip(contribs_list, explain_fn_list)):

        result = compute_stability(
            X=X,
            contribs=contribs,
            explain_fn=explain_fn,
            numeric_mask=numeric_mask,
            categorical_groups=categorical_groups,
            n_perturbations=n_perturbations,
            sigma=sigma,
            cat_flip_prob=cat_flip_prob,
            random_state=random_state + run_idx
        )

        run_results.append(result)
        print("done run", run_idx + 1, "out of", len(contribs_list))

    stability_values = np.array([
        result["stability"] for result in run_results
    ])

    per_sample_values = np.stack([
        result["per_sample_stability"] for result in run_results
    ])

    all_scores_values = np.stack([
        result["all_scores"] for result in run_results
    ])

    stability_mean = np.mean(stability_values)
    stability_sd = (
        np.std(stability_values, ddof=1)
        if len(run_results) > 1
        else 0.0
    )

    per_sample_stability_mean = np.mean(per_sample_values, axis=0)
    per_sample_stability_sd = (
        np.std(per_sample_values, axis=0, ddof=1)
        if len(run_results) > 1
        else np.zeros_like(per_sample_stability_mean)
    )

    all_scores_mean = np.mean(all_scores_values, axis=0)
    all_scores_sd = (
        np.std(all_scores_values, axis=0, ddof=1)
        if len(run_results) > 1
        else np.zeros_like(all_scores_mean)
    )

    return {
        "stability": float(stability_mean),
        "per_sample_stability": per_sample_stability_mean,
        "all_scores": all_scores_mean,

        "stability_mean": float(stability_mean),
        "stability_sd": float(stability_sd),
        "per_sample_stability_sd": per_sample_stability_sd,
        "all_scores_sd": all_scores_sd,

        "run_results": run_results
    }

# =========================================================
# Stability metric 
# =========================================================

def compute_sparsity(contribs):
    """
    Measure how concentrated the explanations are across features.

    The goal is to see whether the model relies on a small set of features or
    spreads importance broadly across many features. The function first turns
    the contribution tensor into local importance scores, then computes a
    Hoyer sparsity value for each sample and for the aggregated global
    importance vector.

    Higher values mean the explanation is more concentrated, with fewer
    dominant features carrying most of the signal. Lower values mean the
    explanation is more diffuse, with importance spread across many features.

    Parameters
    ----------
    contribs : np.ndarray, shape (n_samples, n_features, n_classes)
        Contribution tensor

    Returns
    -------
    dict
        {
            "sparsity": float,
            "per_sample_sparsity": np.ndarray,
            "global_sparsity": float,
            "local_importance": np.ndarray
        }
    """

    # ---------- convert to numpy ----------
    if sparse.issparse(contribs):
        contribs = contribs.toarray()
    else:
        contribs = np.asarray(contribs)

    # ---------- local importance ----------
    local_importance = get_local_importance(contribs)
    n_samples, n_features = local_importance.shape

    # ---------- Hoyer sparsity function ----------
    def hoyer(v):
        l1 = np.sum(np.abs(v))
        l2 = np.sqrt(np.sum(v**2))

        if l2 == 0:
            return 0.0  # no importance at all

        return (np.sqrt(n_features) - (l1 / l2)) / (np.sqrt(n_features) - 1)

    # ---------- local sparsity ----------
    per_sample_sparsity = np.array([hoyer(v) for v in local_importance])
    sparsity = float(np.mean(per_sample_sparsity))

    # ---------- global sparsity ----------
    global_importance = np.mean(local_importance, axis=0)
    global_sparsity = hoyer(global_importance)

    return {
        "sparsity": sparsity,
        "per_sample_sparsity": per_sample_sparsity,
        "global_sparsity": global_sparsity,
        "local_importance": local_importance
    }

def compute_sparsity_repeated(contribs_list):
    """
    Compute sparsity results across repeated model runs.

    This wrapper calls compute_sparsity() once for each contribution tensor.
    Each run produces its own local sparsity, global sparsity, per-sample
    sparsity values, and local importance scores.

    The final output keeps the same main structure as compute_sparsity(),
    but reports the average results across runs. It also adds the standard
    deviation across runs, so that the stability of the sparsity results can
    be assessed.

    Returns
    -------
    dict
        Dictionary containing:
        - "sparsity": mean local sparsity across runs
        - "per_sample_sparsity": mean per-sample sparsity across runs
        - "global_sparsity": mean global sparsity across runs
        - "local_importance": mean local importance across runs
        - "sparsity_sd": standard deviation of local sparsity across runs
        - "global_sparsity_sd": standard deviation of global sparsity across runs
        - "per_sample_sparsity_sd": standard deviation of per-sample sparsity across runs
        - "local_importance_sd": standard deviation of local importance across runs
        - "run_results": individual outputs from each run
    """

    run_results = []

    for contribs in contribs_list:
        result = compute_sparsity(contribs)
        run_results.append(result)

    sparsity_values = np.array([
        result["sparsity"] for result in run_results
    ])

    global_sparsity_values = np.array([
        result["global_sparsity"] for result in run_results
    ])

    per_sample_values = np.stack([
        result["per_sample_sparsity"] for result in run_results
    ])

    local_importance_values = np.stack([
        result["local_importance"] for result in run_results
    ])

    sparsity_mean = np.mean(sparsity_values)
    sparsity_sd = (
        np.std(sparsity_values, ddof=1)
        if len(run_results) > 1
        else 0.0
    )

    global_sparsity_mean = np.mean(global_sparsity_values)
    global_sparsity_sd = (
        np.std(global_sparsity_values, ddof=1)
        if len(run_results) > 1
        else 0.0
    )

    per_sample_sparsity_mean = np.mean(per_sample_values, axis=0)
    per_sample_sparsity_sd = (
        np.std(per_sample_values, axis=0, ddof=1)
        if len(run_results) > 1
        else np.zeros_like(per_sample_sparsity_mean)
    )

    local_importance_mean = np.mean(local_importance_values, axis=0)
    local_importance_sd = (
        np.std(local_importance_values, axis=0, ddof=1)
        if len(run_results) > 1
        else np.zeros_like(local_importance_mean)
    )

    return {
        "sparsity": float(sparsity_mean),
        "per_sample_sparsity": per_sample_sparsity_mean,
        "global_sparsity": float(global_sparsity_mean),
        "local_importance": local_importance_mean,

        "sparsity_mean": float(sparsity_mean),
        "sparsity_sd": float(sparsity_sd),
        "global_sparsity_mean": float(global_sparsity_mean),
        "global_sparsity_sd": float(global_sparsity_sd),
        "per_sample_sparsity_sd": per_sample_sparsity_sd,
        "local_importance_sd": local_importance_sd,

        "run_results": run_results
    }