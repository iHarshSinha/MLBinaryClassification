# Best Parameters:
# {'subsample': 0.8, 'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 20, 'min_child_samples': 10, 'max_depth': 6, 'learning_rate': 0.01, 'colsample_bytree': 0.6}

# --- Final Model Training and F1 Optimization ---
# --- Tuned LGBM F1 Results ---
# Optimal Threshold: 0.5000
# Maximized F1 Score (OOF): 0.7348


# 0. IMPORT PREPROCESSING + LIBRARIES
from P1_preprocess import *    
import numpy as np
import pandas as pd
import lightgbm as lgb
import os


from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.metrics import f1_score
from scipy.optimize import minimize

os.makedirs("../output", exist_ok=True)

# 1 Define models and hyperparameter spaces
print("\n--- Starting Randomized Hyperparameter Search (LGBM) ---")

base_lgbm = lgb.LGBMClassifier(
    objective='binary',
    metric='None',
    n_estimators=2000,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

param_dist = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'num_leaves': [20, 31, 50, 70],
    'max_depth': [-1, 6, 8, 10],
    'min_child_samples': [10, 20, 50, 100],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



# 2. RANDOMIZED SEARCH
random_search = RandomizedSearchCV(
    estimator=base_lgbm,
    param_distributions=param_dist,
    n_iter=50,
    scoring='roc_auc',
    cv=skf,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    return_train_score=True
)

random_search.fit(X_processed, y)

best_params = random_search.best_params_
best_auc = random_search.best_score_

print("\n--- Search Complete ---")
print(f"Best AUC Score: {best_auc:.4f}")
print("Best Parameters:")
print(best_params)



# 3. TRAIN BEST MODEL + OOF PROBAS FOR F1 THRESHOLD
print("\n--- Final Model Training and F1 Optimization ---")

best_lgbm = random_search.best_estimator_
best_lgbm.fit(X_processed, y)

oof_probas_tuned = cross_val_predict(
    best_lgbm,
    X_processed,
    y,
    cv=skf,
    method='predict_proba',
    n_jobs=-1
)[:, 1]



# 4. F1 THRESHOLD OPTIMIZATION
def objective_f1_tuned(threshold):
    y_pred = (oof_probas_tuned >= threshold).astype(int)
    return -f1_score(y, y_pred)

result_tuned = minimize(
    objective_f1_tuned,
    x0=0.5,
    method='L-BFGS-B',
    bounds=[(0.01, 0.99)],
    tol=1e-5
)

optimal_threshold_tuned = float(result_tuned.x)
optimal_f1_tuned = -result_tuned.fun

print("\n--- Tuned LGBM F1 Results ---")
print(f"Optimal Threshold: {optimal_threshold_tuned:.4f}")
print(f"Maximized F1 Score (OOF): {optimal_f1_tuned:.4f}")


# 5. TEST PREDICTION + SUBMISSION FILE
test_probas_tuned = best_lgbm.predict_proba(X_test_processed)[:, 1]

hard_labels_tuned = np.where(
    test_probas_tuned >= optimal_threshold_tuned,
    "Left",
    "Stayed"
)

submission_df = pd.DataFrame({
    "founder_id": test_ids,
    "retention_status": hard_labels_tuned
})

output_path = "../output/lgbm_hyperparameter_tuned_submission.csv"

submission_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)

print("\nSubmission File Created:")
print(f" -> {output_path}\n")
print("Preview:")
print(submission_df.head())
