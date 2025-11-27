import pandas as pd
import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import time



# 1. LOAD DATA
print("Loading Data...")
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

train['is_train'] = 1
test['is_train'] = 0
test['retention_status'] = "Unknown"

df = pd.concat([train, test], axis=0)



# 2. CLEANING + ENCODING (FULLY FIXED)
print("Preprocessing...")

# Drop truly irrelevant columns
drop_cols = ["founder_id", "founder_visibility", "innovation_support"]
df = df.drop(columns=drop_cols)

# NUMERIC IMPUTATION
df["monthly_revenue_generated"] = df["monthly_revenue_generated"].fillna(df["monthly_revenue_generated"].median())
df["years_since_founding"] = df["years_since_founding"].fillna(df["years_since_founding"].median())
df["num_dependents"] = df["num_dependents"].fillna(df["num_dependents"].mode()[0])

# Target-related ordinals
rating_map = {
    'Unknown': 2, 'Low': 0, 'Poor': 0, 'Below Average': 1, 'Fair': 2,
    'Medium': 2, 'Average': 2, 'Good': 3, 'High': 3, 'Very High': 4, 'Excellent': 4
}

for col in ["work_life_balance_rating", "venture_satisfaction", "startup_performance_rating", "startup_reputation"]:
    df[col] = df[col].map(rating_map).fillna(2)

# Stage encoding
stage_map = {'Entry': 1, 'Mid': 2, 'Senior': 3, 'Growth': 3, 'Established': 4}
df["startup_stage"] = df["startup_stage"].map(stage_map).fillna(1)

# Binary encoding
binary_map = {"No": 0, "Yes": 1}
for col in ["working_overtime", "remote_operations"]:
    df[col] = df[col].map(binary_map).fillna(0)

# Log transform for skew
df["monthly_revenue_generated"] = np.log1p(df["monthly_revenue_generated"])

# One-hot encode remaining categoricals
df = pd.get_dummies(
    df,
    columns=["founder_gender", "education_background", "personal_status", 
             "founder_role", "team_size_category", "leadership_scope"],
    drop_first=True
)



# 3. FORM TRAIN / TEST MATRICES
train_df = df[df['is_train'] == 1].drop(columns=["is_train"])
test_df  = df[df['is_train'] == 0].drop(columns=["is_train", "retention_status"])

y = train_df["retention_status"].map({"Stayed": 0, "Left": 1})
X = train_df.drop(columns=["retention_status"])
X_submit = test_df[X.columns]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_submit_scaled = scaler.transform(X_submit)

inverse_map = {0: "Stayed", 1: "Left"}



# 4. MODEL CONFIGS — ONLY MLP + LGBM
models = {
    "MLP": {
        "estimator": MLPClassifier(max_iter=800, random_state=42),
        "params": {
            "hidden_layer_sizes": [
                (64,), (128,), (256,),
                (64, 32), (128, 64), (256, 128),
                (128, 128)
            ],
            "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "activation": ["relu", "tanh"],
            "solver": ["adam"]
        }
    },

    "LGBM": {
        "estimator": lgb.LGBMClassifier(n_estimators=1200, random_state=42),
        "params": {
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [31, 50, 70, 100],
            "max_depth": [-1, 6, 8, 10],
            "min_child_samples": [10, 20, 50, 100],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5]
        }
    }
}



# 5. TRAIN BOTH MODELS IN LOOP
os.makedirs("../output", exist_ok=True)

for name, cfg in models.items():

  
    print(f" TRAINING {name}")
  
    search = RandomizedSearchCV(
        estimator=cfg["estimator"],
        param_distributions=cfg["params"],
        n_iter=40,              # strong overnight search
        scoring="accuracy",
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    start = time.time()
    search.fit(X_scaled, y)

    print("\nBest Params:", search.best_params_)
    print("Best CV Score:", search.best_score_)

    model = search.best_estimator_
    model.fit(X_scaled, y)

    preds = model.predict(X_submit_scaled)
    labels = [inverse_map[p] for p in preds]

    df_out = pd.DataFrame({
        "founder_id": test["founder_id"],
        "retention_status": labels
    })

    file = f"../output/submission_{name}.csv"
    df_out.to_csv(file, index=False)

    print(f"Saved: {file}")
    print(f"Time: {time.time() - start:.1f}s")



print("     FINISHED TRAINING MLP + LGBM")

