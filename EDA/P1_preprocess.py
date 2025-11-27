import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


# 1. Load Data
TRAIN_PATH = os.path.join('..', 'data', 'train.csv')
TEST_PATH = os.path.join('..', 'data', 'test.csv')

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


test_ids = test_df["founder_id"].copy()

# 2. Feature Engineering
def feature_engineering(df):
    df = df.copy()

    df['age_at_founding'] = df['founder_age'] - df['years_since_founding']

    epsilon = 1e-6
    df['tenure_ratio'] = df['years_with_startup'] / (df['years_since_founding'] + epsilon)

    df['unhappy_overtime'] = (
        (df['working_overtime'] == 'Yes').astype(int)
        * df['venture_satisfaction'].map(
            {'Low': 1, 'Medium': 0.5, 'High': 0, 'Very High': 0}
        )
    )

    return df.drop('founder_id', axis=1, errors='ignore')


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)



# 3. Target
train_df['retention_status'] = train_df['retention_status'].map({'Left': 1, 'Stayed': 0})

X = train_df.drop('retention_status', axis=1)
y = train_df['retention_status']
X_test = test_df


# 4. Preprocessing
numerical_features = [
    'founder_age', 'years_with_startup', 'monthly_revenue_generated',
    'funding_rounds_led', 'distance_from_investor_hub',
    'num_dependents', 'years_since_founding', 'age_at_founding',
    'tenure_ratio', 'unhappy_overtime'
]

ordinal_mappings = [
    ('work_life_balance_rating', ['Poor', 'Fair', 'Good', 'Excellent', 'Missing']),
    ('venture_satisfaction', ['Low', 'Medium', 'High', 'Very High', 'Missing']),
    ('startup_performance_rating', ['Below Average', 'Low', 'Average', 'High', 'Excellent']),
    ('startup_reputation', ['Low', 'Moderate', 'High', 'Excellent']),
    ('founder_visibility', ['Low', 'Medium', 'High', 'Very High'])
]

ordinal_cols = [col for col, _ in ordinal_mappings]
ordinal_categories = [cats for _, cats in ordinal_mappings]

nominal_cols = [
    'founder_gender', 'founder_role', 'working_overtime',
    'education_background', 'personal_status', 'startup_stage',
    'team_size_category', 'remote_operations', 'leadership_scope',
    'innovation_support'
]

# Pipelines
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('ordinal', OrdinalEncoder(categories=ordinal_categories,
                               handle_unknown='use_encoded_value',
                               unknown_value=-1))
])

nominal_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols)
    ],
    remainder='drop',
    n_jobs=-1
)

# Transform
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

print("P1_preprocess.py executed successfully.")
print("Train shape:", X_processed.shape)
print("Test shape:", X_test_processed.shape)


__all__ = [
    "X_processed",
    "X_test_processed",
    "y",
    "test_ids"
]
