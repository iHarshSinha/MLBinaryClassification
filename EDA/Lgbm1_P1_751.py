import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix
from scipy.optimize import minimize_scalar
# Ensure output folder exists
os.makedirs("../output", exist_ok=True)


from P1_preprocess import X_processed, X_test_processed, y, test_ids


# 1. Define LGBM Model
lgbm_clf = lgb.LGBMClassifier(
    objective='binary',
    metric='None',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)


# 2. Cross-validation for threshold optimization

print("Starting threshold optimization...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_probas = cross_val_predict(
    lgbm_clf,
    X_processed,
    y,
    cv=skf,
    method='predict_proba',
    n_jobs=-1
)[:, 1]


def objective_f1(th):
    return -f1_score(y, (oof_probas >= th).astype(int))


result = minimize_scalar(objective_f1, bounds=(0.01, 0.99), method='bounded')
optimal_threshold = float(result.x)
optimal_f1 = -result.fun

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Best F1 score: {optimal_f1:.4f}")

print("\nConfusion matrix (OOF):")
print(confusion_matrix(y, (oof_probas >= optimal_threshold).astype(int)))


# 3. Train final model
print("\nTraining final LightGBM model...")
final_model = lgbm_clf.fit(X_processed, y)

test_probas = final_model.predict_proba(X_test_processed)[:, 1]

# 4. Prepare submission
hard_labels = np.where(test_probas >= optimal_threshold, "Left", "Stayed")

submission = pd.DataFrame({
    "founder_id": test_ids,
    "retention_status": hard_labels
})

output_path = "../output/lgbm_f1_hard_labels_submission_final.csv"

submission.to_csv(output_path, index=False)

print("\nSaved submission as:", output_path)
print(submission.head())
