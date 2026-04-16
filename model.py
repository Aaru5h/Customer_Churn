"""
model.py
--------
All ML logic lives here: pipeline construction, training, evaluation,
prediction, and feature-importance extraction.
No Streamlit or UI code belongs in this file.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    load_and_clean_data,
)

# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def build_pipeline(model_type: str, preprocessor) -> Pipeline:
    """
    Assemble and return a full Scikit-learn Pipeline consisting of:
      1. A ColumnTransformer preprocessor (passed in)
      2. A classifier — either Logistic Regression or Decision Tree

    Parameters
    ----------
    model_type : str
        "Logistic Regression" or "Decision Tree"
    preprocessor : ColumnTransformer
        The preprocessor from build_preprocessor()

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if model_type == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "Decision Tree":
        classifier = DecisionTreeClassifier(max_depth=6, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_models(filepath: str) -> dict:
    """
    Load data, split into train/test, train both pipelines, evaluate them,
    and return a results dictionary.

    Parameters
    ----------
    filepath : str
        Path to the raw bank churn CSV file.

    Returns
    -------
    dict with keys:
        "Logistic Regression" → {"pipeline": ..., "metrics": ...,
                                  "X_test": ..., "y_test": ...}
        "Decision Tree"       → same structure
        "feature_names"       → list[str] of encoded feature names
    """
    # Load and clean raw data
    df = load_and_clean_data(filepath)

    # Separate features from target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 80/20 stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    for model_type in ["Logistic Regression", "Decision Tree"]:
        # Build a fresh preprocessor and pipeline for each model
        preprocessor = build_preprocessor()
        pipeline = build_pipeline(model_type, preprocessor)

        # Fit on training data
        pipeline.fit(X_train, y_train)

        # Evaluate on held-out test data
        metrics = evaluate(pipeline, X_test, y_test)

        results[model_type] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
        }

    # Derive encoded feature names from the first trained pipeline's preprocessor
    results["feature_names"] = get_feature_names(
        results["Logistic Regression"]["pipeline"]
    )

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Generate all evaluation metrics for a trained pipeline against test data.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix,
                    classification_report (as DataFrame)
    """
    y_pred = pipeline.predict(X_test)

    # Compute scalar metrics with macro averaging for balanced representation
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Full confusion matrix (2×2 for binary classification)
    cm = confusion_matrix(y_test, y_pred)

    # Classification report as a tidy DataFrame
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Not Churned", "Churned"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T.round(3)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report_df,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(pipeline: Pipeline, input_df: pd.DataFrame) -> tuple:
    """
    Run the trained pipeline on a single-row input DataFrame.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fully trained churn prediction pipeline.
    input_df : pd.DataFrame
        One-row DataFrame with the same column names as the training features.

    Returns
    -------
    (label, probability) where label is "Churned" or "Not Churned"
    and probability is the float churn probability for class 1.
    """
    # Predict probability scores for both classes
    proba = pipeline.predict_proba(input_df)[0]
    churn_prob = float(proba[1])

    label = "Churned" if churn_prob > 0.5 else "Not Churned"
    return label, churn_prob


# ---------------------------------------------------------------------------
# Feature name extraction
# ---------------------------------------------------------------------------


def get_feature_names(pipeline: Pipeline) -> list:
    """
    Extract the full list of feature names produced by the ColumnTransformer
    after one-hot encoding, in the order they appear in the transformed matrix.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted pipeline whose first step is a ColumnTransformer.

    Returns
    -------
    list[str] of feature names
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    # Numerical features pass through unchanged (after impute + scale)
    num_feature_names = NUMERICAL_FEATURES.copy()

    # Categorical features expand via OneHotEncoder — get the generated names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = list(
        cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    )

    return num_feature_names + cat_feature_names


# ---------------------------------------------------------------------------
# Feature importance helpers
# ---------------------------------------------------------------------------


def get_top_features(
    pipeline: Pipeline,
    model_type: str,
    feature_names: list,
    n: int = 5,
) -> tuple:
    """
    Extract the top-N most important features for the selected model type.

    For Decision Tree       → uses feature_importances_ (Gini importance)
    For Logistic Regression → uses absolute values of coefficients for class 1

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted pipeline.
    model_type : str
        "Logistic Regression" or "Decision Tree"
    feature_names : list[str]
        Decoded feature names from get_feature_names().
    n : int
        Number of top features to return.

    Returns
    -------
    (names, values) — both lists of length n, sorted descending by value
    """
    classifier = pipeline.named_steps["classifier"]

    if model_type == "Decision Tree":
        importances = classifier.feature_importances_
    elif model_type == "Logistic Regression":
        # coef_ shape is (1, n_features) for binary classification
        importances = np.abs(classifier.coef_[0])
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # Rank features by importance descending
    indices = np.argsort(importances)[::-1][:n]
    top_names  = [feature_names[i] for i in indices]
    top_values = [float(importances[i]) for i in indices]

    return top_names, top_values


def get_all_top_features(
    pipeline: Pipeline,
    model_type: str,
    feature_names: list,
    n: int = 15,
) -> tuple:
    """
    Same as get_top_features but defaults to top 15 for the performance tab.
    Returns all available features if fewer than n exist.
    """
    return get_top_features(
        pipeline, model_type, feature_names, n=min(n, len(feature_names))
    )
