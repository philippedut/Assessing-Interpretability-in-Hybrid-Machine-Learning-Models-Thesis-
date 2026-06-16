"""
Grid-search utilities for model hyperparameter tuning.

This module contains helper routines to perform grid searches for the
different model families used in the experiments (MLP, NAM, and the
logistic-regression + MLP hybrid). Each grid-search function runs through a
cartesian product of hyperparameter configurations, trains models using a
provided `fit_fn` callable, records validation metrics, and returns the best
model(s) and a DataFrame of results.

Structure
---------
- `grid_search_MLP`: grid search for black-box MLP models (selects best by
    validation loss).
- `grid_search_nam`: grid search for NAM models.
- `grid_search_hybrid_logreg_mlp`: grid search for the hybrid Logistic
    Regression + MLP architecture; selection is based on the hybrid's final
    validation log-loss.

Usage
-----
Provide a `fit_fn(X_train, y_train, X_val, y_val, **params)` that returns the
trained model(s), history and any auxiliary outputs required by the wrapper.

Example:

best_model, results_df, best_params = grid_search_MLP(X_train, y_train, X_val, y_val, fit_fn, param_grid)
"""

import numpy as np
import keras
import itertools
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from Models_Functions import hybrid_predict

# =========================================================
# 1) Grid Search MLP
# =========================================================

# --> this search can be use for both the MLP only but also the Decison tree+ MLP model (for the MLP part of the model) 
def grid_search_MLP(X_train, y_train, X_val, y_val, fit_fn, param_grid):
    """
    Grid search for MLP models using validation loss as selection criterion.
    """

    keys = list(param_grid.keys())
    total = int(np.prod([len(param_grid[k]) for k in keys]))

    best_val_loss = np.inf
    best_acc = 0.0
    best_model = None
    best_params = None

    results = []

    for i, values in enumerate(itertools.product(*[param_grid[k] for k in keys]), start=1):
        params = dict(zip(keys, values))
        print(f"\n[{i}/{total}] Trying: {params}")

        # Always clear previous graph
        keras.backend.clear_session()

        model, hist, _ = fit_fn(X_train, y_train, X_val, y_val, **params)

        val_losses = hist.history.get("val_loss", [])
        acc = hist.history.get("val_acc", [])
        current_best_val_loss = float(np.min(val_losses)) if len(val_losses) else np.nan
        current_best_acc = float(np.max(acc)) if len(acc) else np.nan

        results.append({
            **params,
            "best_val_loss": current_best_val_loss,
            "best_val_acc": current_best_acc,
            "epochs_ran": len(hist.history.get("loss", []))
        })

        if not np.isnan(current_best_val_loss) and current_best_val_loss < best_val_loss:
            best_val_loss = current_best_val_loss
            best_acc = current_best_acc
            best_model = model
            best_params = params

        print(f"[{i}/{total}] best_val_loss (this run): {current_best_val_loss:.6f} | best so far: {best_val_loss:.6f}")
        print(f" best_val_acc (this run): {current_best_acc:.4f}")

    results_df = pd.DataFrame(results).sort_values("best_val_loss")

    print("\nTop configs:")
    print(results_df.head(10))

    print("\nBest params:", best_params)

    return best_model, results_df, best_params

# =========================================================
# 2 Grid Search NAM
# =========================================================

def grid_search_nam(X_train, y_train, X_val, y_val, fit_fn, param_grid):
    """
    Grid search for NAM models using validation loss as selection criterion.
    """

    keys = list(param_grid.keys())
    total = int(np.prod([len(param_grid[k]) for k in keys]))

    best_val_loss = np.inf
    best_acc = 0.0
    best_model = None
    best_params = None

    results = []

    for i, values in enumerate(itertools.product(*[param_grid[k] for k in keys]), start=1):
        params = dict(zip(keys, values))
        print(f"\n[{i}/{total}] Trying: {params}")

        # Always clear previous graph
        keras.backend.clear_session()

        model, hist,_ ,_ = fit_fn(X_train, y_train, X_val, y_val, **params)

        val_losses = hist.history.get("val_loss", [])
        acc = hist.history.get("val_acc", [])
        current_best_val_loss = float(np.min(val_losses)) if len(val_losses) else np.nan
        current_best_acc = float(np.max(acc)) if len(acc) else np.nan

        results.append({
            **params,
            "best_val_loss": current_best_val_loss,
            "best_val_acc": current_best_acc,
            "epochs_ran": len(hist.history.get("loss", []))
        })

        if not np.isnan(current_best_val_loss) and current_best_val_loss < best_val_loss:
            best_val_loss = current_best_val_loss
            best_acc = current_best_acc
            best_model = model
            best_params = params

        print(f"[{i}/{total}] best_val_loss (this run): {current_best_val_loss:.6f} | best so far: {best_val_loss:.6f}")
        print(f" best_val_acc (this run): {current_best_acc:.4f}")

    results_df = pd.DataFrame(results).sort_values("best_val_loss")

    print("\nTop configs:")
    print(results_df.head(10))

    print("\nBest params:", best_params)

    return best_model, results_df, best_params

# =========================================================
# 3) Grid Search Hybrid Logistic Regression + MLP
# =========================================================

def grid_search_hybrid_logreg_mlp(X_train, y_train, X_val, y_val, fit_fn, param_grid):
    """
    Grid search for the Logistic Regression + MLP hybrid model
    using validation log-loss of the FINAL HYBRID prediction
    as selection criterion.
    """

    keys = list(param_grid.keys())
    total = int(np.prod([len(param_grid[k]) for k in keys]))

    best_val_loss = np.inf
    best_acc = 0.0
    best_models = None
    best_params = None

    results = []

    for i, values in enumerate(itertools.product(*[param_grid[k] for k in keys]), start=1):
        params = dict(zip(keys, values))
        print(f"\n[{i}/{total}] Trying: {params}")

        # Always clear previous graph
        keras.backend.clear_session()

        logreg_model, bb_model, hist, _ = fit_fn(
            X_train, y_train, X_val, y_val, **params
        )

        # Evaluate FINAL hybrid on validation set
        tau = params.get("tau", 0.7)
        P_val, use_logreg, conf = hybrid_predict(logreg_model, bb_model, X_val, tau=tau)
        y_hat = np.argmax(P_val, axis=1)

        val_ll = log_loss(y_val, P_val)
        val_acc = accuracy_score(y_val, y_hat)

        # Also track the individual MLP training history
        val_losses_hist = hist.history.get("val_loss", [])
        val_acc_hist = hist.history.get("val_acc", [])
        best_mlp_val_loss = float(np.min(val_losses_hist)) if len(val_losses_hist) else np.nan
        best_mlp_val_acc = float(np.max(val_acc_hist)) if len(val_acc_hist) else np.nan

        results.append({
            **params,
            "hybrid_val_logloss": val_ll,
            "hybrid_val_acc": val_acc,
            "hybrid_transparency": float(use_logreg.mean()),
            "avg_logreg_conf": float(conf.mean()),
            "best_mlp_val_loss": best_mlp_val_loss,
            "best_mlp_val_acc": best_mlp_val_acc,
            "epochs_ran": len(hist.history.get("loss", []))
        })

        if val_ll < best_val_loss:
            best_val_loss = val_ll
            best_acc = val_acc
            best_models = (logreg_model, bb_model)
            best_params = params

        print(f"[{i}/{total}] hybrid_val_logloss (this run): {val_ll:.6f} | best so far: {best_val_loss:.6f}")
        print(f"hybrid_val_acc (this run): {val_acc:.4f}")
        print(f"hybrid_transparency: {use_logreg.mean():.4f}")
        print(f"avg_logreg_conf: {conf.mean():.4f}")

    results_df = pd.DataFrame(results).sort_values("hybrid_val_logloss").reset_index(drop=True)

    print("\nTop configs:")
    print(results_df.head(10))

    print("\nBest params:", best_params)

    return best_models, results_df, best_params
