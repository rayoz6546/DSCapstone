import json
import boto3
from openai import OpenAI
import uuid
from datetime import datetime
import pandas as pd
from io import StringIO

dynamodb = boto3.resource("dynamodb")
secrets_client = boto3.client("secretsmanager")
s3 = boto3.client("s3")

TABLE_NAME = "job-automation-capstone"
SECRET_NAME = "job-automation-openai-key"
BUCKET_NAME = "job-automation-datasets"

SKILLS_KEY = "public_skills_data.csv"
SCORED_FEATURES_KEY = "job_features_scored.csv"

table = dynamodb.Table(TABLE_NAME)

response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
secret_dict = json.loads(response["SecretString"])
api_key = secret_dict["api_key"]
client = OpenAI(api_key=api_key)


def load_datasets_from_s3(bucket_name: str):
    """
    Load the raw skills dataset and the scored job-level feature dataset from S3.
    """
    skills_obj = s3.get_object(Bucket=bucket_name, Key=SKILLS_KEY)
    skills_df = pd.read_csv(StringIO(skills_obj["Body"].read().decode("utf-8")))

    scored_obj = s3.get_object(Bucket=bucket_name, Key=SCORED_FEATURES_KEY)
    scored_df = pd.read_csv(StringIO(scored_obj["Body"].read().decode("utf-8")))

    return skills_df, scored_df


def llm_find_matches(input_title: str, skills_df: pd.DataFrame):
    """
    Use the LLM to map a user-entered job title to 1-5 official occupations.
    """
    job_titles = (
        skills_df["2024 National Employment Matrix title"]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )

    prompt = f"""
You are an expert in job classification.

A user entered this job title:

"{input_title}"

Below is a list of official occupations.

Your task is to select the occupations that best match the user's job.

Rules:
- Return between 1 and 5 matches.
- Only choose occupations that truly represent the user's job.
- Ignore occupations that are clearly unrelated.

Return ONLY valid JSON in this format:

{{
  "matches": ["job title 1", "job title 2"]
}}

Occupations:
{json.dumps(job_titles)}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    text = response.output_text.strip()
    parsed = json.loads(text)
    return parsed["matches"]


def build_job_info(input_title: str, matches, skills_df: pd.DataFrame):
    """
    Build a detailed job info structure from the raw skills dataset.
    Expects matches as a list of dicts:
    [{"job_title": "...", "similarity": 1.0}, ...]
    """
    result = {
        "input_job_title": input_title,
        "matches": []
    }

    for match in matches:
        title = match["job_title"]

        job_rows = skills_df[
            skills_df["2024 National Employment Matrix title"] == title
        ]

        if job_rows.empty:
            continue

        employment_2024 = job_rows["Employment, 2024"].iloc[0]
        employment_2034 = job_rows["Employment, 2034"].iloc[0]

        skills = {}

        for _, row in job_rows.iterrows():
            skill = row["EP skills title"]
            element = row["O*NET element name"]
            score = row["O*NET data value"]

            if skill not in skills:
                skills[skill] = {}

            skills[skill][element] = score

        result["matches"].append({
            "matched_job_title": title,
            "similarity": match["similarity"],
            "employment": {
                "2024": None if pd.isna(employment_2024) else float(employment_2024),
                "2034": None if pd.isna(employment_2034) else float(employment_2034)
            },
            "skills": skills
        })

    return result


def get_scored_match(best_match_title: str, scored_df: pd.DataFrame):
    """
    Retrieve the structured automation-risk result for the matched occupation.
    """
    row = scored_df[
        scored_df["2024_national_employment_matrix_title"] == best_match_title
    ]

    if row.empty:
        return None

    row = row.iloc[0]

    return {
        "matched_job_title": best_match_title,
        "automation_risk_score": None if pd.isna(row.get("automation_risk_score")) else round(float(row["automation_risk_score"]), 2),
        "automation_risk_label": None if pd.isna(row.get("automation_risk_label")) else str(row["automation_risk_label"]),
        "employment_2024": None if pd.isna(row.get("employment_2024")) else float(row["employment_2024"]),
        "employment_2034": None if pd.isna(row.get("employment_2034")) else float(row["employment_2034"]),
        "employment_change_numeric_2024_34": None if pd.isna(row.get("employment_change_numeric_2024_34")) else float(row["employment_change_numeric_2024_34"]),
        "employment_change_percent_2024_34": None if pd.isna(row.get("employment_change_percent_2024_34")) else float(row["employment_change_percent_2024_34"]),
    }


def lambda_handler(event, context):
    method = event["requestContext"]["http"]["method"]

    if method == "POST":
        try:
            body = json.loads(event["body"])
            job_title = body["jobTitle"]

            skills_df, scored_df = load_datasets_from_s3(BUCKET_NAME)

            matched_titles = llm_find_matches(job_title, skills_df)
            matches = [{"job_title": title, "similarity": 1.0} for title in matched_titles]

            print("Closest matches:", matched_titles)

            job_info = build_job_info(job_title, matches, skills_df)

            if not matched_titles:
                return {
                    "statusCode": 404,
                    "body": json.dumps({"message": "No matching occupations found"})
                }

            best_match_title = matched_titles[0]
            scored_result = get_scored_match(best_match_title, scored_df)

            if scored_result is None:
                return {
                    "statusCode": 404,
                    "body": json.dumps({
                        "message": f"No scored result found for matched occupation: {best_match_title}"
                    })
                }

            job_id = str(uuid.uuid4())

            item = {
                "jobId": job_id,
                "jobTitle": job_title,
                "matchedJobTitle": scored_result["matched_job_title"],
                "automationRiskScore": scored_result["automation_risk_score"],
                "automationRiskLabel": scored_result["automation_risk_label"],
                "employment2024": scored_result["employment_2024"],
                "employment2034": scored_result["employment_2034"],
                "employmentChangeNumeric2024_34": scored_result["employment_change_numeric_2024_34"],
                "employmentChangePercent2024_34": scored_result["employment_change_percent_2024_34"],
                "createdAt": datetime.utcnow().isoformat()
            }

            table.put_item(Item=item)

            response_body = {
                "jobId": job_id,
                "inputJobTitle": job_title,
                "matchedTitles": matched_titles,
                "bestMatch": scored_result,
                "jobInfo": job_info
            }

            return {
                "statusCode": 200,
                "body": json.dumps(response_body)
            }

        except Exception as e:
            print("POST error:", str(e))
            return {
                "statusCode": 500,
                "body": json.dumps({"message": "Internal server error", "error": str(e)})
            }

    elif method == "GET":
        try:
            job_id = event["queryStringParameters"]["id"]
            response = table.get_item(Key={"jobId": job_id})

            if "Item" in response:
                return {
                    "statusCode": 200,
                    "body": json.dumps(response["Item"])
                }
            else:
                return {
                    "statusCode": 404,
                    "body": json.dumps({"message": "Not found"})
                }

        except Exception as e:
            print("GET error:", str(e))
            return {
                "statusCode": 500,
                "body": json.dumps({"message": "Internal server error", "error": str(e)})
            }

    return {
        "statusCode": 400,
        "body": json.dumps({"message": "Unsupported method"})
    }
