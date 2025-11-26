# ============================================================
# 0. IMPORT PREPROCESSING + LIBRARIES

# ===========================================================
# --- Neural Network Search Complete ---
# Best AUC Score: 0.8139pytho
# Best Hyperparameters:
# {'mlp__solver': 'adam', 'mlp__learning_rate_init': 0.01, 'mlp__hidden_layer_sizes': (128,), 'mlp__alpha': 0.01, 'mlp__activation': 'tanh'}
from P1_preprocess import *
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from scipy.optimize import minimize


# ============================================================
# 1. DEFINE PIPELINE + PARAMETER SPACE
# ============================================================
print("\n--- Starting MLP Neural Network Training ---")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(max_iter=400, random_state=42))
])

param_dist = {
    "mlp__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (128, 128)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "mlp__learning_rate_init": [0.0005, 0.001, 0.003, 0.01],
    "mlp__solver": ["adam"],
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 2. RANDOMIZED SEARCH
# ============================================================
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,
    scoring="roc_auc",
    cv=skf,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_processed, y)

best_params = random_search.best_params_
best_auc = random_search.best_score_

print("\n--- Neural Network Search Complete ---")
print(f"Best AUC Score: {best_auc:.4f}")
print("Best Hyperparameters:")
print(best_params)

best_mlp = random_search.best_estimator_


# ============================================================
# 3. OOF PROBAS + THRESHOLD OPTIMIZATION
# ============================================================
print("\n--- Performing OOF prediction for F1 threshold optimization ---")

oof_probas = cross_val_predict(
    best_mlp,
    X_processed,
    y,
    cv=skf,
    method="predict_proba",
    n_jobs=-1
)[:, 1]


def objective_f1(threshold):
    return -f1_score(y, (oof_probas >= threshold).astype(int))


result = minimize(
    objective_f1,
    x0=0.5,
    method="L-BFGS-B",
    bounds=[(0.01, 0.99)],
    tol=1e-5
)

optimal_threshold = float(result.x)
optimal_f1 = -result.fun

print("\n--- MLP Threshold Optimization Results ---")
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"OOF F1 Score:     {optimal_f1:.4f}")


# ============================================================
# 4. TEST PREDICTION + SUBMISSION
# ============================================================
test_pred_probas = best_mlp.predict_proba(X_test_processed)[:, 1]

hard_labels = np.where(test_pred_probas >= optimal_threshold, "Left", "Stayed")

submission = pd.DataFrame({
    "founder_id": test_ids,
    "retention_status": hard_labels
})

submission.to_csv("mlp_f1_submission.csv", index=False)

print("\nSubmission Created: mlp_f1_submission.csv")
print(submission.head())
