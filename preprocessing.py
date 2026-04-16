"""
preprocessing.py
----------------
Defines feature lists, builds the ColumnTransformer preprocessor,
and handles raw CSV loading and cleaning for the bank churn dataset.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature definitions
# These lists define which columns are treated as numerical vs categorical.
# They must match exactly the columns present in the dataset after cleaning.
# ---------------------------------------------------------------------------

NUMERICAL_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

CATEGORICAL_FEATURES = [
    "Geography",
    "Gender",
]

# Target column name in the dataset
TARGET_COLUMN = "Exited"

# Columns to drop during cleaning (identifiers, not predictive features)
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw bank churn CSV, apply mandatory cleaning steps, and
    return a DataFrame ready for training.

    Cleaning steps:
      1. Drop identifier columns: RowNumber, CustomerId, Surname.
      2. The target column Exited is already binary (0/1) — no re-encoding needed.
      3. Handle any missing values downstream through the Scikit-learn Pipeline.
    """
    # Load raw CSV
    df = pd.read_csv(filepath)

    # Drop non-predictive identifier columns
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    # Ensure target is integer (already is, but be explicit for safety)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Construct and return a ColumnTransformer that applies:
      - Numerical pipeline: median imputation → StandardScaler
      - Categorical pipeline: most-frequent imputation → OneHotEncoder

    This preprocessor is intended to be the first step in a Scikit-learn
    Pipeline so that the exact same transformations are applied at both
    training and prediction time.
    """
    # Numerical feature transformation: impute then scale
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical feature transformation: impute then one-hot encode
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # Combine both sub-pipelines into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor
