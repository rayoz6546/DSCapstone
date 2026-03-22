import pandas as pd
import re


def clean_soc_code(value):
    """
    Clean SOC codes so both datasets can be merged more easily.
    Keeps digits and hyphen only.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()
    value = re.sub(r"[^0-9\-]", "", value)

    return value if value else None


def load_job_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_automation_target(path: str) -> pd.DataFrame:
    """
    Load the Kaggle automation dataset as Excel.
    """
    return pd.read_csv(path, encoding="latin1")


def prepare_automation_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns needed for supervised learning.
    """
    df = df.copy()

    needed_cols = ["SOC", "Occupation", "Probability"]
    df = df[needed_cols]

    df["soc_clean"] = df["SOC"].apply(clean_soc_code)
    df["occupation_clean"] = df["Occupation"].astype(str).str.strip().str.lower()

    return df


def prepare_job_features_for_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the engineered O*NET features for merging.
    """
    df = df.copy()

    title_col = "2024_national_employment_matrix_title"
    if title_col in df.columns:
        df["occupation_clean"] = df[title_col].astype(str).str.strip().str.lower()

    # Look for a SOC-like column if it exists
    soc_candidates = [col for col in df.columns if "soc" in col.lower()]
    if soc_candidates:
        soc_col = soc_candidates[0]
        df["soc_clean"] = df[soc_col].apply(clean_soc_code)
    else:
        df["soc_clean"] = None

    return df


def merge_training_data(job_features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    First try merging on SOC code.
    For rows that do not match, fall back to occupation title.
    """
    merged_soc = job_features_df.merge(
        target_df,
        on="soc_clean",
        how="left",
        suffixes=("", "_target")
    )

    no_match_mask = merged_soc["Probability"].isna()

    unmatched_job_features = merged_soc.loc[no_match_mask, job_features_df.columns].copy()

    fallback_merge = unmatched_job_features.merge(
        target_df,
        on="occupation_clean",
        how="left",
        suffixes=("", "_target")
    )

    matched_soc = merged_soc.loc[~no_match_mask].copy()

    final_df = pd.concat([matched_soc, fallback_merge], ignore_index=True)

    return final_df


if __name__ == "__main__":
    job_features_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/job_features.csv"
    automation_target_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/automation_data_by_state.csv"
    output_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/training_dataset.csv"

    job_features_df = load_job_features(job_features_path)
    automation_df = load_automation_target(automation_target_path)

    job_features_df = prepare_job_features_for_merge(job_features_df)
    automation_df = prepare_automation_target(automation_df)

    merged_df = merge_training_data(job_features_df, automation_df)

    print("Merged dataset shape:", merged_df.shape)
    print("Rows with target Probability:", merged_df["Probability"].notna().sum())
    print(
        merged_df[
            ["2024_national_employment_matrix_title", "Occupation", "Probability"]
        ].head(10)
    )

    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged training dataset to: {output_path}")