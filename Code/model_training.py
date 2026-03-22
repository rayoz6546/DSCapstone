import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_training_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def select_features_and_target(df: pd.DataFrame):
    df = df.copy()
    df = df[df["Probability"].notna()].copy()

    exclude_cols = {
        "Probability",
        "SOC",
        "Occupation",
        "soc_clean",
        "occupation_clean",
        "occupation_clean_target",
        "2024_national_employment_matrix_title",
        "automation_risk_score",
        "automation_risk_label",
        "raw_risk_score",
        "lower_risk_score",
        "higher_risk_score",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if df[col].notna().sum() > 0]

    X = df[feature_cols].copy()
    y = df["Probability"].copy()

    return X, y, feature_cols, df


def drop_all_missing_train_columns(X_train, X_test):
    valid_cols = [col for col in X_train.columns if X_train[col].notna().sum() > 0]
    X_train = X_train[valid_cols].copy()
    X_test = X_test[valid_cols].copy()
    return X_train, X_test, valid_cols


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n{name}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")

    return model, preds, {"MAE": mae, "RMSE": rmse, "R2": r2}


def get_random_forest_importance_df(rf_model, X_train):
    rf = rf_model.named_steps["regressor"]

    feature_names = X_train.columns.tolist()
    importances_array = rf.feature_importances_

    min_len = min(len(feature_names), len(importances_array))

    importances = pd.DataFrame({
        "feature": feature_names[:min_len],
        "importance": importances_array[:min_len]
    }).sort_values("importance", ascending=False)

    return importances


if __name__ == "__main__":
    input_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/training_dataset.csv"
    model_output_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/automation_rf_model.pkl"
    importance_output_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/rf_feature_importances.csv"
    predictions_output_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/model_predictions.csv"

    df = load_training_data(input_path)

    X, y, feature_cols, filtered_df = select_features_and_target(df)

    print("Training rows:", len(filtered_df))
    print("Number of features before train filtering:", len(feature_cols))
    print("Target summary:")
    print(y.describe())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Keep matching row info for later inspection
    train_indices = X_train.index
    test_indices = X_test.index

    X_train, X_test, feature_cols = drop_all_missing_train_columns(X_train, X_test)

    print("Number of features used in training:", len(feature_cols))

    linear_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression())
    ])

    rf_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    linear_model, linear_preds, linear_metrics = evaluate_model(
        "Linear Regression",
        linear_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    rf_model, rf_preds, rf_metrics = evaluate_model(
        "Random Forest Regressor",
        rf_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    importances_df = get_random_forest_importance_df(rf_model, X_train)

    print("\nTop Random Forest Features:")
    print(importances_df.head(15).to_string(index=False))

    test_rows = filtered_df.loc[test_indices].copy()

    results_df = pd.DataFrame({
        "occupation_title": test_rows["2024_national_employment_matrix_title"].values,
        "actual": y_test.values,
        "linear_pred": linear_preds,
        "rf_pred": rf_preds
    })

    print("\nSample predictions:")
    print(results_df.head(10).to_string(index=False))

    # Save trained random forest pipeline
    with open(model_output_path, "wb") as f:
        pickle.dump({
            "model": rf_model,
            "feature_columns": feature_cols
        }, f)

    # Save feature importances
    importances_df.to_csv(importance_output_path, index=False)

    # Save predictions
    results_df.to_csv(predictions_output_path, index=False)

    print(f"\nSaved model to: {model_output_path}")
    print(f"Saved feature importances to: {importance_output_path}")
    print(f"Saved predictions to: {predictions_output_path}")