import pickle
import pandas as pd


def load_saved_model(model_path: str):
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["feature_columns"]


def load_job_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def predict_for_job_title(job_title: str, model, feature_columns, job_features_df: pd.DataFrame):
    title_col = "2024_national_employment_matrix_title"

    match = job_features_df[
        job_features_df[title_col].astype(str).str.strip().str.lower()
        == job_title.strip().lower()
    ]

    if match.empty:
        raise ValueError(f"No occupation found for: {job_title}")

    row = match.iloc[[0]].copy()

    missing_cols = [col for col in feature_columns if col not in row.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X = row[feature_columns].copy()
    pred = model.predict(X)[0]

    result = {
        "occupation_title": row[title_col].iloc[0],
        "predicted_automation_probability": float(pred),
        "employment_2024": None if pd.isna(row["employment_2024"].iloc[0]) else float(row["employment_2024"].iloc[0]),
        "employment_2034": None if pd.isna(row["employment_2034"].iloc[0]) else float(row["employment_2034"].iloc[0]),
        "employment_change_numeric_2024_34": None if pd.isna(row["employment_change_numeric_2024_34"].iloc[0]) else float(row["employment_change_numeric_2024_34"].iloc[0]),
        "employment_change_percent_2024_34": None if pd.isna(row["employment_change_percent_2024_34"].iloc[0]) else float(row["employment_change_percent_2024_34"].iloc[0]),
    }

    return result


if __name__ == "__main__":
    model_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/automation_rf_model.pkl"
    job_features_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/job_features.csv"

    model, feature_columns = load_saved_model(model_path)
    job_features_df = load_job_features(job_features_path)

    # change this to test different occupations
    job_title = "Data scientists"

    result = predict_for_job_title(job_title, model, feature_columns, job_features_df)

    print("\nPrediction result:")
    print(f"Occupation: {result['occupation_title']}")
    print(f"Predicted automation probability: {result['predicted_automation_probability']:.4f}")
    print(f"Employment 2024: {result['employment_2024']}")
    print(f"Employment 2034: {result['employment_2034']}")
    print(f"Employment change (numeric): {result['employment_change_numeric_2024_34']}")
    print(f"Employment change (%): {result['employment_change_percent_2024_34']}")