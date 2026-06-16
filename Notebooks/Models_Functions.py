"""
Model utilities and architectures used throughout the thesis experiments.

This module centralizes model-building, training helpers, and explanation
utilities used across experiments. It provides reusable components for the
black-box MLP, Neural Additive Model (NAM), and hybrid "pre–black-box"
approach (Logistic Regression gating + MLP fallback), together with functions
to extract SHAP contributions and compute local/global feature importance.

Structure
---------
1. Global hyperparameters and defaults for training.
2. Helper functions for local/global importance extraction.
3. Shared neural-network building blocks and callbacks.
4. Black-box MLP: build, fit, and SHAP contribution extraction.
5. NAM model components and training utilities.
6. Hybrid PRE–BLACK-BOX helpers (logistic gating and hybrid prediction).

Usage
-----
Import the module or specific functions and pass model/data callables so
utilities remain agnostic to training/evaluation loops. Example:

from Models_Functions import build_blackbox_mlp, get_mlp_shap_contributions
model = build_blackbox_mlp(n_features, n_classes)
contribs = get_mlp_shap_contributions(model, X_train, X_explain)
"""

import pandas as pd
import numpy as np
import shap
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Shared hyperparams just for the test, but will be fined tuned for each model during training
HIDDEN = (128, 64)
DROPOUT = 0.2
L2 = 1e-4
LR = 3e-4
BATCH = 256
EPOCHS = 400
TAU = 0.7 # confidence threshold for using the tree's prediction in the hybrid model
backround = 200
seed = 42

# =========================================================
# 0 Helper functions to extract local and global importance from contributions
# =========================================================
def get_local_importance(contribs):
    """
    contribs: (n_samples, n_features, n_classes)
    returns:  (n_samples, n_features)
    """
    Local_importance = np.mean(np.abs(contribs), axis=2)
    return Local_importance

# --------------------------------------------------------
def get_global_importance(contribs):
    """
    contribs: (n_samples, n_features, n_classes)
    returns:  (n_features,)
    """
    global_importance = np.mean(np.abs(contribs), axis=(0, 2))
    return global_importance


# =========================================================
# 1 SHARED NEURAL BUILDING BLOCK
# =========================================================
"""
Those function builds a reusable MLP block to be use in all three of the models (black-box, NAM, residual).
the only difference between mlp_block and mlp_block_nam is that the latter does not include batch normalization,
as it can interfere with the interpretability of the NAM model. 

    - the first dense layer is used to learn a linear transformation of the input features, 
      and the L2 regularization helps prevent overfitting by adding a penalty to large weights.
    - the batch normalization layer normalizes the output of the dense layer, which can help stabilize 
      training and improve convergence.
    - the activation layer applies the ReLU activation function, introducing non-linearity to the model, 
      which allows it to learn more complex patterns in the data.
    - the dropout layer randomly sets a fraction of the input units to zero during training, 
      which helps prevent overfitting by reducing reliance.

    we use does layers as they are standard and provide a stable architecture.
"""
def mlp_block(x, units, dropout=0.2, l2=1e-4, name_prefix=""):

    x = layers.Dense(
        units,
        kernel_regularizer=regularizers.l2(l2),
        name=f"{name_prefix}dense_{units}"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}bn_{units}")(x)
    x = layers.Activation("relu", name=f"{name_prefix}relu_{units}")(x)
    x = layers.Dropout(dropout, name=f"{name_prefix}drop_{units}")(x)
    return x

# --------------------------------------------------------
def mlp_block_nam(x, units, dropout=0.2, l2=1e-4, name_prefix=""):

    x = layers.Dense(
        units,
        kernel_regularizer=regularizers.l2(l2),
        name=f"{name_prefix}dense_{units}"
    )(x)
    x = layers.Activation("relu", name=f"{name_prefix}relu_{units}")(x)
    x = layers.Dropout(dropout, name=f"{name_prefix}drop_{units}")(x)
    return x

# --------------------------------------------------------
def common_callbacks():
    """
    This function defines common callbacks for all models, including early stopping and learning rate reduction on plateau.
    """
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

# =========================================================
# 2 POST–BLACK-BOX: Black-box MLP (probabilities only)
# =========================================================

def get_mlp_shap_contributions(bb_model, X_train, X_explain, n_background=200, random_state=42):
    """
    This function computes SHAP contributions for a trained tf.keras MLP and
    returns them in the same NAM-like format used by the rest of the file:
    (n_samples, n_features, n_classes).

    - the trained black-box model is the network we want to explain;
    - X_train is used to draw a small background sample;
    - X_explain is the data we want to interpret;
    - n_background controls how many training points are used as the SHAP
      reference set;
    - random_state makes the background sampling reproducible.

    The background matters because SHAP explains predictions relative to a
    reference distribution. In practice, using a small sample of training data
    gives the explainer a realistic baseline for what "typical" inputs look
    like, while keeping the computation manageable.

    Returns
    -------
    contribs : np.ndarray
        SHAP contributions with shape (n_samples, n_features, n_classes).
    """

    def to_dense(X):
        if sparse.issparse(X):
            return X.toarray()
        elif isinstance(X, pd.DataFrame):
            return X.values
        else:
            return np.asarray(X)

    X_train_dense = to_dense(X_train)
    X_explain_dense = to_dense(X_explain)

    # sample background from training data
    rng = np.random.default_rng(random_state)
    n_background = min(n_background, X_train_dense.shape[0])
    bg_idx = rng.choice(X_train_dense.shape[0], size=n_background, replace=False)
    X_background = X_train_dense[bg_idx]

    # build SHAP explainer
    explainer = shap.GradientExplainer(bb_model, X_background)
    shap_values = explainer.shap_values(X_explain_dense)

    # convert SHAP output to common format
    if isinstance(shap_values, list):
        # multiclass: list of (n_samples, n_features)
        contribs = np.stack(shap_values, axis=-1)   # (n_samples, n_features, n_classes)
    else:
        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 2:
            # binary case -> force 2-class format
            contribs = np.stack([-shap_values, shap_values], axis=-1)
        elif shap_values.ndim == 3:
            contribs = shap_values
        else:
            raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")

    return contribs

def build_blackbox_mlp(n_features, n_classes, hidden=HIDDEN, dropout=DROPOUT, l2=L2):
    """ this functions builds a standard MLP architecture for the black-box model, 
    which can be more transparent with the help of black-box explainers.
    - the input layer takes in the features of the dataset.
    - the hidden layers are built using the reusable mlp_block function.
    - the output layer uses a softmax activation function to produce class probabilities."""

    x_in = keras.Input(shape=(n_features,), name="X")
    x = x_in
    for i, u in enumerate(hidden):
        x = mlp_block(x, u, dropout=dropout, l2=l2, name_prefix=f"bb_{i}_")
    probs = layers.Dense(n_classes, activation="softmax", name="bb_probs")(x)
    return keras.Model(x_in, probs, name="BlackBoxMLP")

# --------------------------------------------------------
def fit_blackbox_mlp(
        X_train, 
        y_train,
        X_val,
        y_val,
        hidden=HIDDEN, 
        dropout=DROPOUT, 
        l2=L2, 
        lr=LR, 
        epochs=EPOCHS, 
        batch_size=BATCH
        ):
    """ 
    this function fits the black-box MLP model to the training data, using the specified hyperparameters.
    - it first builds the model using the build_blackbox_mlp function, then compiles it with the Adam optimizer and sparse categorical crossentropy loss.
    - it then fits the model to the training data, using a validation split for early stopping and learning rate reduction.
    - it returns the fitted model, which can be used to make predictions and evaluate performance on the test set. 
    """
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    bb_model = build_blackbox_mlp(n_features, n_classes, hidden=hidden, dropout=dropout, l2=l2)

    bb_model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    history =bb_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=common_callbacks(),
        verbose=1
    )

    contribs = get_mlp_shap_contributions(bb_model, X_train, X_val, n_background= backround, random_state= seed)


    return bb_model, history , contribs

# =========================================================
# 3 NAM: Intrinsic interpretable model (additive per-feature nets)
# =========================================================

def build_feature_net(feature_index, n_classes, hidden=HIDDEN, dropout=DROPOUT, l2=L2):
    """
    This functions builds small neural networks for each feature, 
    the output of each network is the contribution of that feature to the final prediction.

    - the input layer takes in a single feature (hence shape=(1,)).
    - the hidden layers are built using the reusable mlp_block function.
    - the output layer produces the logits contribution of that feature to each class,
      without applying softmax, as we will sum these contributions across features and then apply 
      softmax at the end to get probabilities.
    """
    x_in = keras.Input(shape=(1,), name=f"f{feature_index}_in")
    x = x_in
    for i, u in enumerate(hidden):
        x = mlp_block_nam(x, u, dropout=dropout, l2=l2, name_prefix=f"nam_f{feature_index}_{i}_")
    out = layers.Dense(n_classes, activation=None, name=f"f{feature_index}_out")(x)  # internal logits contributions
    return keras.Model(x_in, out, name=f"FeatureNet_{feature_index}")

# --------------------------------------------------------
def build_nam(n_features, n_classes, hidden=HIDDEN, dropout=DROPOUT, l2=L2):
    """
    This function builds the NAM model by creating a separate feature net for each input feature and then summing their contributions.
    - the input layer takes in all features of the dataset.
    - for each feature, we slice it out and pass it through its corresponding feature net to get its contribution to the logits.
    - we then sum the contributions from all features and add a bias term to get the final logits.
    - finally, we apply a softmax activation to get the class probabilities.
    - we also create a separate model that outputs the individual feature contributions, which can be used for plotting and importance analysis.
    """
    x_in = keras.Input(shape=(n_features,), name="X")

    # Keras-safe per-feature slicing
    feature_tensors = [
        layers.Lambda(lambda t, i=i: t[:, i:i+1], name=f"slice_f{i}")(x_in)
        for i in range(n_features)
    ]

    feature_nets = [
        build_feature_net(i, n_classes, hidden=hidden, dropout=dropout, l2=l2)
        for i in range(n_features)
    ]

    contribs = [feature_nets[i](feature_tensors[i]) for i in range(n_features)]  # each (batch, K)
    
    logits = layers.Add(name="nam_logits_sum")(contribs)
    logits = layers.Dense(n_classes,use_bias=True, activation=None, name="nam_bias")(logits)

    probs = layers.Activation("softmax", name="nam_probs")(logits)

    model = keras.Model(x_in, probs, name="NAM")
    contrib_model = keras.Model(x_in, contribs, name="NAM_Contribs")  # for plotting/importance
    return model, contrib_model, feature_nets

# --------------------------------------------------------
def fit_nam(
        X_train, 
        y_train,
        X_val,
        y_val, 
        hidden=HIDDEN, 
        dropout=DROPOUT, 
        l2=L2, lr=LR, 
        epochs=EPOCHS, 
        batch_size=BATCH
        ):
    """ this function fits the NAM model to the training data, using the specified hyperparameters.
    - it first builds the model using the build_nam function, then compiles it with the Adam optimizer and sparse categorical crossentropy loss.
    - it then fits the model to the training data, using a validation split for early stopping and learning rate reduction.
    - it returns the fitted model, the contribution model (for plotting/importance), and the list of feature nets (for plotting/importance). 
    """

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    nam_model, nam_contribs, nam_feature_nets = build_nam(n_features, n_classes, hidden=hidden, dropout=dropout, l2=l2)

    nam_model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    history = nam_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=common_callbacks(),
        verbose=1
    )

    return nam_model, history, nam_contribs, nam_feature_nets

# --------------------------------------------------------
def get_nam_contributions(contrib_model, X):
    """
    Returns NAM contributions in shape:
    (n_samples, n_features, n_classes)
    """
    contribs = contrib_model.predict(X, verbose=0)

    # Keras model returns a list: length n_features,
    # each element has shape (n_samples, n_classes)
    contribs = np.stack(contribs, axis=1)

    return contribs

# =========================================================
# 4 PRE–BLACK-BOX: Logistic Regression + MLP based thresholding
# =========================================================
"""
The idea of this model is to use logistic regression as the interpretable model.
If logistic regression is confident enough, we use its prediction.
Otherwise, we defer to the black-box MLP.
"""
def eval_probs(y_true, P, name="Model"):
    """
    Evaluates the predicted probabilities P against the true labels y_true using accuracy and log-loss.
    It prints the results and returns the accuracy and log-loss values.
    """
    y_pred = np.argmax(P, axis=1)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, P)
    print(f"[{name}] accuracy={acc:.4f} | log-loss={ll:.4f}")
    return acc, ll

# --------------------------------------------------------
def hybrid_predict(logreg_model, bb_model, X, tau=0.8):
    """
    This function combines the interpretable logistic regression model with
    the black-box MLP using a confidence threshold.

    - the logistic regression model is checked first;
    - its confidence is defined as the maximum predicted class probability;
    - if that confidence is at least tau, we trust the logistic regression
      prediction;
    - otherwise, we defer to the black-box MLP prediction.

    The idea is to keep the simpler model when it is sufficiently certain,
    while falling back to the more flexible MLP when the linear model is less
    confident.
    """
    P_lr = logreg_model.predict_proba(X)          # (n, K)
    P_bb = bb_model.predict(X, verbose=0)         # (n, K)

    conf = P_lr.max(axis=1)
    use_logreg = conf >= tau

    P_final = P_bb.copy()
    P_final[use_logreg] = P_lr[use_logreg]

    return P_final, use_logreg, conf

# --------------------------------------------------------
def get_logreg_contributions(logreg_model, X):
    """
    Returns logistic regression contributions in shape:
    (n_samples, n_features, n_classes)

    For binary classification, output is expanded to 2 classes.
    """

    if sparse.issparse(X):
        X_np = X.toarray()
    elif isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = np.asarray(X)

    coef = logreg_model.coef_          # (K, p) or (1, p)

    # Binary case
    if coef.shape[0] == 1:
        pos_contrib = X_np * coef[0]             # (n, p)
        neg_contrib = -pos_contrib               # (n, p)
        contribs = np.stack([neg_contrib, pos_contrib], axis=-1)  # (n, p, 2)

    # Multiclass case
    else:
        contribs = X_np[:, None, :] * coef[None, :, :]   # (n, K, p)
        contribs = np.transpose(contribs, (0, 2, 1))     # (n, p, K)

    return contribs

# --------------------------------------------------------
def fit_hybrid_logreg_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        tau=0.8,
        logreg_C=1.0,
        logreg_penalty="l2",
        logreg_solver="lbfgs",
        logreg_class_weight=None,
        hidden=HIDDEN,
        dropout=DROPOUT,
        l2=L2,
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH
        ):
    """
    This function trains the hybrid model that combines logistic regression
    with a black-box MLP.

        - first, logistic regression is fitted on the training data;
        - then, the black-box MLP is trained on the same data;
        - on validation data, the model uses logistic regression when its
        confidence is at least tau, and falls back to the MLP otherwise.

    This setup keeps the interpretable model when it is sufficiently certain,
    while still relying on the MLP in harder cases.

    """
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Logistic Regression
    # Handles binary and multiclass automatically.
    logreg_model = LogisticRegression(
        C=logreg_C,
        penalty=logreg_penalty,
        solver=logreg_solver,
        class_weight=logreg_class_weight,
        max_iter=2000,
        
    )
    logreg_model.fit(X_train, y_train)

    # Black-box MLP
    bb_model = build_blackbox_mlp(
        n_features=n_features,
        n_classes=n_classes,
        hidden=hidden,
        dropout=dropout,
        l2=l2
    )

    bb_model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    history = bb_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=common_callbacks(),
        verbose=1
    )

    # Validation hybrid evaluation
    P_hybrid_val, use_logreg, conf = hybrid_predict(
        logreg_model, bb_model, X_val, tau=tau
    )

    contribs = get_logreg_contributions(logreg_model, X_val)
    eval_probs(y_val, P_hybrid_val, name=f"Hybrid LogReg+MLP (τ={tau})")

    print(f"Fraction using Logistic Regression on validation set: {use_logreg.mean():.4f}")
    print(f"Average Logistic Regression confidence on validation set: {conf.mean():.4f}")

    return logreg_model, bb_model, history, contribs 

