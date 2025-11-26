# ============================================================
# 0. IMPORT PREPROCESSING + LIBRARIES
# ============================================================
from P1_preprocess import *
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from scipy.optimize import minimize


# ============================================================
# 1. PIPELINE + PARAM SPACE
# ============================================================
print("\n--- Starting SVM Model Training ---")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True, random_state=42))
])

param_dist = {
    "svm__C": np.logspace(-2, 3, 10),
    "svm__gamma": np.logspace(-4, 0, 10),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 2. RANDOM SEARCH
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

print("\n--- SVM Search Complete ---")
print(f"Best AUC Score: {best_auc:.4f}")
print("Best Grid:")
print(best_params)

best_svm = random_search.best_estimator_


# ============================================================
# 3. OOF + F1 THRESHOLD TUNING
# ============================================================
print("\n--- Performing OOF predictions for threshold optimization ---")

oof_probas = cross_val_predict(
    best_svm,
    X_processed,
    y,
    cv=skf,
    method="predict_proba",
    n_jobs=-1
)[:, 1]


def objective_f1(t):
    return -f1_score(y, (oof_probas >= t).astype(int))


result = minimize(
    objective_f1,
    x0=0.5,
    method="L-BFGS-B",
    bounds=[(0.01, 0.99)]
)

optimal_threshold = float(result.x)
optimal_f1 = -result.fun

print("\n--- SVM F1 Optimization Results ---")
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"OOF F1 Score:     {optimal_f1:.4f}")


# ============================================================
# 4. TEST PREDICTIONS + SUBMISSION
# ============================================================
test_pred_probas = best_svm.predict_proba(X_test_processed)[:, 1]

hard_labels = np.where(test_pred_probas >= optimal_threshold, "Left", "Stayed")

submission = pd.DataFrame({
    "founder_id": test_ids,
    "retention_status": hard_labels
})

submission.to_csv("svm_f1_submission.csv", index=False)

print("\nSubmission Created: svm_f1_submission.csv")
print(submission.head())
