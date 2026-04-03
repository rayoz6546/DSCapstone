import os
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CODE_DIR = os.path.join(PROJECT_ROOT, "Code")
sys.path.insert(0, CODE_DIR)

from model_training import (
    select_features_and_target,
    drop_all_missing_train_columns,
    evaluate_model,
)

# allow imports from the Code/ folder
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CODE_DIR = os.path.join(PROJECT_ROOT, "Code")
sys.path.insert(0, CODE_DIR)

from model_training import (
    select_features_and_target,
    drop_all_missing_train_columns,
)


def test_select_features_and_target_filters_missing_probability_and_excludes_columns():
    df = pd.DataFrame({
        "Probability": [0.2, np.nan, 0.8],
        "SOC": ["11-1011", "11-1021", "11-1031"],
        "Occupation": ["A", "B", "C"],
        "feature_num_1": [1.0, 2.0, 3.0],
        "feature_num_2": [10.0, 20.0, 30.0],
        "feature_all_nan": [np.nan, np.nan, np.nan],
        "non_numeric": ["x", "y", "z"],
    })

    X, y, feature_cols, filtered_df = select_features_and_target(df)

    # row with missing Probability should be removed
    assert len(filtered_df) == 2
    assert y.shape[0] == 2

    # excluded identifier columns should not appear in features
    assert "Probability" not in feature_cols
    assert "SOC" not in feature_cols
    assert "Occupation" not in feature_cols

    # non-numeric and all-NaN columns should not appear
    assert "non_numeric" not in feature_cols
    assert "feature_all_nan" not in feature_cols

    # valid numeric features should remain
    assert "feature_num_1" in feature_cols
    assert "feature_num_2" in feature_cols
    assert X.shape[1] == 2


def test_drop_all_missing_train_columns_removes_only_invalid_training_columns():
    X_train = pd.DataFrame({
        "good_col": [1.0, 2.0, 3.0],
        "all_missing_train": [np.nan, np.nan, np.nan],
        "partially_missing": [1.0, np.nan, 2.0],
    })

    X_test = pd.DataFrame({
        "good_col": [4.0, 5.0],
        "all_missing_train": [7.0, 8.0],
        "partially_missing": [np.nan, 3.0],
    })

    X_train_new, X_test_new, valid_cols = drop_all_missing_train_columns(X_train, X_test)

    assert "all_missing_train" not in valid_cols
    assert "all_missing_train" not in X_train_new.columns
    assert "all_missing_train" not in X_test_new.columns

    assert "good_col" in valid_cols
    assert "partially_missing" in valid_cols
    assert list(X_train_new.columns) == list(X_test_new.columns)


def test_random_forest_pipeline_fits_and_predicts_on_small_sample():
    X_train = pd.DataFrame({
        "feature_1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "feature_2": [10.0, 20.0, 30.0, np.nan, 50.0],
    })
    y_train = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

    X_test = pd.DataFrame({
        "feature_1": [1.5, np.nan],
        "feature_2": [15.0, 35.0],
    })

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == 2
    assert np.isfinite(preds).all()

def test_evaluate_model_returns_predictions_and_metrics():
    X_train = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0],
        "feature_2": [10.0, 20.0, 30.0, 40.0],
    })
    y_train = pd.Series([0.1, 0.2, 0.3, 0.4])

    X_test = pd.DataFrame({
        "feature_1": [1.5, 3.5],
        "feature_2": [15.0, 35.0],
    })
    y_test = pd.Series([0.15, 0.35])

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    fitted_model, preds, metrics = evaluate_model(
        "Random Forest Test",
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    assert len(preds) == len(y_test)
    assert np.isfinite(preds).all()
    assert {"MAE", "RMSE", "R2"} <= set(metrics.keys())
    assert isinstance(fitted_model, Pipeline)