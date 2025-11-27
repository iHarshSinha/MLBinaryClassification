
# 0. IMPORT PREPROCESSING + REQUIRED LIBRARIES

from P1_preprocess import *

import catboost as cb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from scipy.optimize import minimize
import numpy as np
import pandas as pd


# 1. TRAIN CATBOOST MODEL
print("--- Training CatBoost Model ---")

cb_clf = cb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    verbose=0,
    random_state=42
)

cb_clf.fit(X_processed, y)
cb_test_probas = cb_clf.predict_proba(X_test_processed)[:, 1]

print("CatBoost training + test prediction complete.\n")


# 2. CROSS-VALIDATION (OOF PROBAS) + F1 THRESHOLD OPTIMIZATION
print("Performing 5-Fold CV for CatBoost threshold optimization...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cb_oof_probas = cross_val_predict(
    cb_clf,
    X_processed,
    y,
    cv=skf,
    method='predict_proba',
    n_jobs=-1
)[:, 1]

def objective_f1_cb(threshold):
    """Returns negative F1-score for minimization."""
    y_pred_binary = (cb_oof_probas >= threshold).astype(int)
    return -f1_score(y, y_pred_binary)

result_cb = minimize(
    objective_f1_cb,
    x0=0.5,
    method='L-BFGS-B',
    bounds=[(0.01, 0.99)],
    tol=1e-5
)

optimal_threshold_cb = result_cb.x[0]
optimal_f1_cb = -result_cb.fun

print("\n--- CatBoost F1 Optimization Results ---")
print(f"Optimal CatBoost Threshold: {optimal_threshold_cb:.4f}")
print(f"Maximized CatBoost OOF F1 Score: {optimal_f1_cb:.4f}\n")


# 3. FINAL SUBMISSION FILE
hard_labels_cb = np.where(
    cb_test_probas >= optimal_threshold_cb,
    "Left",
    "Stayed"
)

submission_df_cb = pd.DataFrame({
    "founder_id": test_ids,
    "retention_status": hard_labels_cb
})

output_path = "../output/catboost_f1_submission.csv"

submission_df_cb.to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)

print("--- CATBOOST SUBMISSION CREATED ---")
print("Saved as: catboost_f1_submission.csv\n")
print("Preview:")
print(submission_df_cb.head())
