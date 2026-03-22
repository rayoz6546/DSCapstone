import pandas as pd


def load_skills_data(csv_path: str) -> pd.DataFrame:
    """
    Load the public skills dataset from CSV.
    """
    df = pd.read_csv(csv_path)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names so they are easier to work with in code.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df

def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df

def build_job_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per occupation with:
    - employment summary columns
    - aggregate statistics on O*NET data values
    - pivoted EP skill features
    - pivoted O*NET element features
    """
    df = clean_column_names(df)

    occupation_col = "2024_national_employment_matrix_title"
    employment_2024_col = "employment_2024"
    employment_2034_col = "employment_2034"
    emp_change_num_col = "employment_change_numeric_2024_34"
    emp_change_pct_col = "employment_change_percent_2024_34"
    ep_skill_col = "ep_skills_title"
    onet_element_col = "o_net_element_name"
    onet_value_col = "o_net_data_value"

    # Keep only rows with occupation and score
    df = df.dropna(subset=[occupation_col, onet_value_col])

    # Base occupation-level summary
    base_features = (
        df.groupby(occupation_col)
        .agg(
            employment_2024=(employment_2024_col, "first"),
            employment_2034=(employment_2034_col, "first"),
            employment_change_numeric_2024_34=(emp_change_num_col, "first"),
            employment_change_percent_2024_34=(emp_change_pct_col, "first"),
            mean_onet_value=(onet_value_col, "mean"),
            max_onet_value=(onet_value_col, "max"),
            min_onet_value=(onet_value_col, "min"),
            std_onet_value=(onet_value_col, "std"),
            num_records=(onet_value_col, "count"),
            num_unique_skills=(ep_skill_col, "nunique"),
            num_unique_elements=(onet_element_col, "nunique"),
        )
        .reset_index()
    )

    # Average O*NET score by EP skill title
    skill_pivot = (
        df.pivot_table(
            index=occupation_col,
            columns=ep_skill_col,
            values=onet_value_col,
            aggfunc="mean"
        )
        .add_prefix("skill_")
        .reset_index()
    )

    # Average O*NET score by O*NET element name
    element_pivot = (
        df.pivot_table(
            index=occupation_col,
            columns=onet_element_col,
            values=onet_value_col,
            aggfunc="mean"
        )
        .add_prefix("element_")
        .reset_index()
    )

    # Merge everything
    job_features = base_features.merge(skill_pivot, on=occupation_col, how="left")
    job_features = job_features.merge(element_pivot, on=occupation_col, how="left")
    job_features = clean_feature_names(job_features)
    return job_features


def get_features_for_job(job_title: str, job_features_df: pd.DataFrame) -> pd.Series:
    """
    Return the feature row for a given occupation title.
    """
    match = job_features_df[
        job_features_df["2024_national_employment_matrix_title"] == job_title
    ]

    if match.empty:
        raise ValueError(f"No features found for job title: {job_title}")

    return match.iloc[0]


if __name__ == "__main__":
    csv_path = "../Datasets/public_skills_data.csv"

    raw_df = load_skills_data(csv_path)
    job_features_df = build_job_feature_table(raw_df)

    print("Job feature table shape:", job_features_df.shape)
    print(job_features_df.head())

    output_path = "/Users/selmayilmaz/Desktop/Capstone/DSCapstone/Datasets/job_features.csv"
    job_features_df.to_csv(output_path, index=False)
    print(f"Saved job features to: {output_path}")