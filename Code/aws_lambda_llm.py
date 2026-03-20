import json
import boto3
import os
from openai import OpenAI
import uuid
from datetime import datetime
import pandas as pd
from io import StringIO 
import numpy as np

dynamodb = boto3.resource("dynamodb")
secrets_client = boto3.client("secretsmanager")

TABLE_NAME = "job-automation-capstone"
SECRET_NAME = "job-automation-capstone-openai-key"
ASSISTANT_ID = "asst_8XcFT2UwFXX4dDyTTfq8b3pP"

table = dynamodb.Table(TABLE_NAME)

response = secrets_client.get_secret_value(SecretId=SECRET_NAME)

secret_dict = json.loads(response["SecretString"])
api_key = secret_dict["api_key"]
client = OpenAI(api_key=api_key)


def load_datasets_from_s3(bucket_name):

    s3 = boto3.client("s3")
 
    ai_obj = s3.get_object(
        Bucket=bucket_name,
        Key="ai_job_trends_dataset.csv"
    )
    ai_jobs_df = pd.read_csv(
        StringIO(ai_obj["Body"].read().decode("utf-8"))
    )


    skills_obj = s3.get_object(
        Bucket=bucket_name,
        Key="public_skills_data.csv"
    )
    skills_df = pd.read_csv(
        StringIO(skills_obj["Body"].read().decode("utf-8"))
    )

    return ai_jobs_df, skills_df


def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


def llm_find_matches(input_title, skills_df):

    job_titles = skills_df["2024 National Employment Matrix title"].astype(str).unique().tolist()

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

Return ONLY JSON in this format:

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

    text = response.output_text

    return json.loads(text)["matches"]

def build_job_info(input_title, matches, skills_df):

    result = {
        "input_job_title": input_title,
        "matches": []
    }

    for match in matches:

        title = match["job_title"]

        job_rows = skills_df[
            skills_df["2024 National Employment Matrix title"] == title
        ]

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
                "2024": employment_2024,
                "2034": employment_2034
            },
            "skills": skills
        })

    return result

def generate_prediction(job_title):
   

    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=job_title
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )

    messages = client.beta.threads.messages.list(thread_id=thread.id)

    return messages.data[0].content[0].text.value.strip()

def lambda_handler(event, context):

    method = event["requestContext"]["http"]["method"]

    if method == "POST":
        body = json.loads(event["body"])
        job_title = body["jobTitle"]

        automation_df, skills_df = load_datasets_from_s3("job-automation-capstone")


        matched_titles = llm_find_matches(job_title, skills_df)

        matches = [{"job_title": t, "similarity": 1.0} for t in matched_titles]

        print("closest matches are:", matched_titles)

        job_info = build_job_info(job_title, matches, skills_df)


        prediction = generate_prediction(job_title)

        print("Prediction is:",prediction)



        job_id = str(uuid.uuid4())

        table.put_item(
            Item={
                "jobId": job_id,
                "jobTitle": job_title,
                "prediction": prediction,
                "createdAt": datetime.utcnow().isoformat()
            }
        )

        return {
            "statusCode": 200,
            "body": json.dumps({"jobId": job_id})
        }

    elif method == "GET":
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

    return {
        "statusCode": 400,
        "body": json.dumps({"message": "Unsupported method"})
    }