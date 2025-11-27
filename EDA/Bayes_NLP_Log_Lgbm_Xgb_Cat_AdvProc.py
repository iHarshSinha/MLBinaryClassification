import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from scipy.optimize import minimize



# 1. LOAD DATA
print("Loading data…")
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

train["is_train"] = 1
test["is_train"] = 0
test["retention_status"] = "Unknown"

combined = pd.concat([train, test], axis=0)



# 2. ADVANCED PREPROCESSING
print("Preprocessing…")

cols_to_drop = ["founder_id", "founder_visibility", "innovation_support"]
combined = combined.drop(columns=cols_to_drop)

combined["monthly_revenue_generated"] = combined["monthly_revenue_generated"].fillna(
    combined["monthly_revenue_generated"].median()
)
combined["years_since_founding"] = combined["years_since_founding"].fillna(
    combined["years_since_founding"].median()
)
combined["num_dependents"] = combined["num_dependents"].fillna(
    combined["num_dependents"].mode()[0]
)
combined["work_life_balance_rating"] = combined["work_life_balance_rating"].fillna("Unknown")
combined["venture_satisfaction"] = combined["venture_satisfaction"].fillna("Unknown")

combined["revenue_efficiency"] = np.log1p(combined["monthly_revenue_generated"]) / (
    combined["years_since_founding"] + 1
)
combined["prior_experience"] = combined["founder_age"] - combined["years_with_startup"]

binary_map = {"No": 0, "Yes": 1}
combined["working_overtime"] = combined["working_overtime"].map(binary_map)
combined["remote_operations"] = combined["remote_operations"].map(binary_map)

sat_map = {
    "Unknown": 2,
    "Low": 0,
    "Poor": 0,
    "Below Average": 1,
    "Fair": 2,
    "Medium": 2,
    "Average": 2,
    "Good": 3,
    "High": 3,
    "Very High": 4,
    "Excellent": 4,
}

combined["satisfaction_score"] = combined["venture_satisfaction"].map(sat_map)
combined["burnout_index"] = combined["working_overtime"] / (combined["satisfaction_score"] + 1)

ordinal_cols = [
    "work_life_balance_rating",
    "venture_satisfaction",
    "startup_performance_rating",
    "startup_reputation",
]
for c in ordinal_cols:
    combined[c] = combined[c].map(sat_map).fillna(2)

stage_map = {"Entry": 1, "Mid": 2, "Senior": 3, "Growth": 3, "Established": 4}
combined["startup_stage"] = combined["startup_stage"].map(stage_map).fillna(1)

combined = pd.get_dummies(
    combined,
    columns=[
        "founder_gender",
        "education_background",
        "personal_status",
        "founder_role",
        "team_size_category",
        "leadership_scope",
    ],
    drop_first=True,
)



# 3. PREPARE MATRICES
train_final = combined[combined["is_train"] == 1].drop(columns=["is_train"])
test_final = combined[combined["is_train"] == 0].drop(columns=["is_train", "retention_status"])

y = train_final["retention_status"].map({"Stayed": 0, "Left": 1})
X = train_final.drop(columns=["retention_status"])
X_submit = test_final[X.columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_submit_scaled = scaler.transform(X_submit)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)



# 4. MODEL CONFIGS
model_configs = {
    "logistic": {
        "model": LogisticRegression(max_iter=2000),
        "params": {"C": np.logspace(-3, 3, 20)},
    },
    "mlp": {
        "model": MLPClassifier(max_iter=600, random_state=42),
        "params": {
            "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "learning_rate_init": [0.001, 0.003, 0.01],
            "alpha": [1e-5, 1e-4, 1e-3],
        },
    },
    "nbayes": {
        "model": GaussianNB(),
        "params": {"var_smoothing": np.logspace(-9, -1, 10)},
    },
    "lgbm": {
        "model": lgb.LGBMClassifier(n_estimators=500),
        "params": {
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [31, 50, 70],
            "min_child_samples": [10, 20, 50],
        },
    },
    "xgb": {
        "model": xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=500,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "params": {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
        },
    },
    "cat": {
        "model": cb.CatBoostClassifier(iterations=500, verbose=0),
        "params": {
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5],
        },
    },
}



# 5. TRAIN MODELS
os.makedirs("../output", exist_ok=True)

model_scores = {}
test_predictions_all = {}

for name, cfg in model_configs.items():
    print(f"\n=== Training {name.upper()} ===")

    search = RandomizedSearchCV(
        estimator=cfg["model"],
        param_distributions=cfg["params"],
        n_iter=15,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # VALIDATION SCORE
    val_pred = best_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    model_scores[name] = auc

    # TRAIN ON FULL DATA
    best_model.fit(X_scaled, y)
    test_pred = best_model.predict_proba(X_submit_scaled)[:, 1]
    test_predictions_all[name] = test_pred

    # SAVE CSV
    out_df = pd.DataFrame({
        "founder_id": test["founder_id"],
        "retention_status": np.where(test_pred >= 0.5, "Left", "Stayed")
    })
    out_df.to_csv(f"../output/{name}_submission.csv", index=False)
    print(f"Saved {name}_submission.csv")


# 6. TOP 3 MODELS FOR ENSEMBLE
top3 = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
top3_names = [m for m, _ in top3]

print("\nTop 3 models:", top3)

# Validation predictions for threshold optimization
val_top3_preds = np.column_stack([val_predictions_all[m] for m in top3_names])

# Test predictions for final submission
test_top3_preds = np.column_stack([test_predictions_all[m] for m in top3_names])


# 7. SIMPLE AVERAGE ENSEMBLE
avg_val = np.mean(val_top3_preds, axis=1)
avg_test = np.mean(test_top3_preds, axis=1)

df_simple = pd.DataFrame({
    "founder_id": test["founder_id"],
    "retention_status": np.where(avg_test >= 0.5, "Left", "Stayed")
})
df_simple.to_csv("../output/ensemble_simple.csv", index=False)
print("Saved ensemble_simple.csv")



# 8. THRESHOLD OPTIMIZED ENSEMBLE  (using validation only)
def loss_threshold(threshold):
    preds = (avg_val >= threshold).astype(int)
    return -(roc_auc_score(y_val, preds))

res_th = minimize(loss_threshold, x0=0.5, bounds=[(0.01, 0.99)], method="L-BFGS-B")
best_th = float(res_th.x)

df_th = pd.DataFrame({
    "founder_id": test["founder_id"],
    "retention_status": np.where(avg_test >= best_th, "Left", "Stayed")
})
df_th.to_csv("../output/ensemble_threshold.csv", index=False)
print("Saved ensemble_threshold.csv")



# 9. WEIGHTED ENSEMBLE (using validation only)
def weight_loss(w):
    w = np.array(w)
    w /= w.sum()
    blended_val = np.dot(val_top3_preds, w)
    preds = (blended_val >= 0.5).astype(int)
    return -(roc_auc_score(y_val, preds))

bounds = [(0, 1)] * 3
cons = {"type": "eq", "fun": lambda w: w.sum() - 1}

res_w = minimize(weight_loss, x0=[1/3, 1/3, 1/3], bounds=bounds, constraints=cons)
best_w = res_w.x / np.sum(res_w.x)

weighted_test = np.dot(test_top3_preds, best_w)

df_weighted = pd.DataFrame({
    "founder_id": test["founder_id"],
    "retention_status": np.where(weighted_test >= 0.5, "Left", "Stayed")
})
df_weighted.to_csv("../output/ensemble_weighted.csv", index=False)
print("Saved ensemble_weighted.csv")

