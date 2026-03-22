import pandas as pd


LOWER_RISK_FEATURES = [
    # Creative Intelligence
    "element_originality",
    "element_thinking_creatively",
    "skill_creativity_and_innovation",
    # Social Intelligence
    "element_social_perceptiveness",
    "element_assisting_and_caring_for_others",
    "element_coaching_and_developing_others",
    "element_selling_or_influencing_others",
    # Unstructured Problem Solving
    "element_repairing_and_maintaining_mechanical_equipment",
    "element_complex_problem_solving",
]

HIGHER_RISK_FEATURES = [
    # Routine Cognitive / Clerical
    "element_documenting_recording_information",
    "element_importance_of_being_exact_or_accurate",
    "element_attention_to_detail",
    # Structured Manual / Control
    "element_operation_and_control",
    "element_controlling_machines_and_processes",
]


def min_max_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series([0.5] * len(series), index=series.index)

    return (series - min_val) / (max_val - min_val)


def build_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize lower-risk features
    for col in LOWER_RISK_FEATURES:
        if col in df.columns:
            df[f"{col}_norm"] = min_max_normalize(df[col])

    # normalize higher-risk features
    for col in HIGHER_RISK_FEATURES:
        if col in df.columns:
            df[f"{col}_norm"] = min_max_normalize(df[col])

    lower_risk_norm_cols = [f"{col}_norm" for col in LOWER_RISK_FEATURES if col in df.columns]
    higher_risk_norm_cols = [f"{col}_norm" for col in HIGHER_RISK_FEATURES if col in df.columns]

    df["lower_risk_score"] = df[lower_risk_norm_cols].mean(axis=1)
    df["higher_risk_score"] = df[higher_risk_norm_cols].mean(axis=1)

    # Base risk score: weight bottlenecks (lower risk) more heavily (0.6) than routine signals (0.4)
    # Higher value = more automatable
    df["raw_risk_score"] = (0.4 * df["higher_risk_score"]) + (0.6 * (1 - df["lower_risk_score"]))

    # final automation risk score (0-100 percentile rank)
    df["automation_risk_score"] = df["raw_risk_score"].rank(pct=True) * 100

    return df


def assign_risk_label(score: float) -> str:
    if score < 33:
        return "Low"
    elif score < 66:
        return "Medium"
    return "High"


def add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["automation_risk_label"] = df["automation_risk_score"].apply(assign_risk_label)
    return df


if __name__ == "__main__":
    input_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/job_features.csv"
    output_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/job_features_scored.csv"

    df = pd.read_csv(input_path)
    df = build_risk_scores(df)
    df = add_risk_labels(df)

    print("\nSample Occupations:")
    print(df[[
        "2024_national_employment_matrix_title",
        "automation_risk_score",
        "automation_risk_label"
    ]].head(10))

    print("\nAutomation Risk Label Distribution:")
    print(df["automation_risk_label"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
    print()

    df.to_csv(output_path, index=False)
    print(f"Saved scored features to: {output_path}")