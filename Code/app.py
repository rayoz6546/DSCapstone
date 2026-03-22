import os
import json
import pickle
import requests
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

api_base_url_secret = st.secrets.get("API_BASE_URL", None)
api_key_secret = st.secrets.get("API_KEY", None)

if api_base_url_secret and api_key_secret:
    API_BASE_URL = api_base_url_secret
    API_KEY = api_key_secret
else:
    env = dotenv_values(ENV_PATH)
    API_BASE_URL = env.get("API_BASE_URL")
    API_KEY = env.get("API_KEY")

MODEL_PATH = BASE_DIR / "Datasets" / "automation_rf_model.pkl"
JOB_FEATURES_PATH = BASE_DIR / "Datasets" / "job_features.csv"

@st.cache_resource
def load_saved_model(model_path: str):
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["feature_columns"]


@st.cache_data
def load_job_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def predict_local_model(job_title: str, model, feature_columns, job_features_df: pd.DataFrame):
    title_col = "2024_national_employment_matrix_title"

    match = job_features_df[
        job_features_df[title_col].astype(str).str.strip().str.lower()
        == job_title.strip().lower()
    ]

    if match.empty:
        return None

    row = match.iloc[[0]].copy()

    missing_cols = [col for col in feature_columns if col not in row.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X = row[feature_columns].copy()
    pred = model.predict(X)[0]

    return {
        "occupation_title": row[title_col].iloc[0],
        "predicted_automation_probability": float(pred),
        "employment_2024": None if pd.isna(row["employment_2024"].iloc[0]) else float(row["employment_2024"].iloc[0]),
        "employment_2034": None if pd.isna(row["employment_2034"].iloc[0]) else float(row["employment_2034"].iloc[0]),
        "employment_change_numeric_2024_34": None if pd.isna(row["employment_change_numeric_2024_34"].iloc[0]) else float(row["employment_change_numeric_2024_34"].iloc[0]),
        "employment_change_percent_2024_34": None if pd.isna(row["employment_change_percent_2024_34"].iloc[0]) else float(row["employment_change_percent_2024_34"].iloc[0]),
    }


def call_aws_api(job_title: str):
    if not API_BASE_URL or not API_KEY:
        raise ValueError("Missing API_BASE_URL or API_KEY in .env")

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {"jobTitle": job_title}

    response = requests.post(API_BASE_URL, json=payload, headers=headers, timeout=30)

    try:
        data = response.json()
    except Exception:
        data = {"raw_response": response.text}

    if response.status_code != 200:
        raise ValueError(f"AWS API error ({response.status_code}): {data}")

    if "body" in data:
        body = data["body"]
        if isinstance(body, str):
            try:
                return json.loads(body)
            except Exception:
                return {"raw_body": body}
        return body

    return data


def probability_label(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Medium"
    return "High"


def format_job_count_in_thousands(value):
    if value is None:
        return "N/A"
    return f"{int(round(value * 1000)):,}"


def risk_badge(label: str):
    if not label:
        return ""
    colors = {
        "Low": "#16a34a",
        "Medium": "#d97706",
        "High": "#dc2626"
    }
    color = colors.get(label, "#64748b")
    return f"""
    <span style="
        display:inline-block;
        padding:0.25rem 0.65rem;
        border-radius:999px;
        background:{color}22;
        color:{color};
        font-weight:600;
        font-size:0.9rem;
        border:1px solid {color}55;
    ">{label}</span>
    """


def metric_card(title: str, value: str, subtitle: str = "", accent: str = "#3b82f6"):
    return f"""
    <div style="
        border:1px solid rgba(148,163,184,0.25);
        border-radius:18px;
        padding:1.1rem 1rem 1rem 1rem;
        background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        box-shadow:0 1px 2px rgba(0,0,0,0.2);
        height:100%;
    ">
        <div style="
            width:42px;
            height:4px;
            border-radius:999px;
            background:{accent};
            margin-bottom:0.8rem;
        "></div>
        <div style="font-size:0.95rem; color:#cbd5e1; margin-bottom:0.5rem;">{title}</div>
        <div style="font-size:2.1rem; font-weight:700; margin-bottom:0.7rem;">{value}</div>
        <div style="font-size:0.92rem; color:#94a3b8;">{subtitle}</div>
    </div>
    """


def build_combined_takeaway(local_result, llm_prediction, best_match):
    parts = []

    model_prob = None
    model_label = None
    if local_result and local_result.get("predicted_automation_probability") is not None:
        model_prob = local_result["predicted_automation_probability"]
        model_label = probability_label(model_prob)

    llm_label = llm_prediction.get("automation_risk_label") if llm_prediction else None
    rule_label = best_match.get("automation_risk_label") if best_match else None
    matched_title = best_match.get("matched_job_title", "This occupation") if best_match else "This occupation"

    if model_label and llm_label:
        if model_label == llm_label:
            parts.append(
                f"Both the trained model and the LLM estimate suggest a {model_label.lower()} level of automation risk for {matched_title.lower()}."
            )
        else:
            parts.append(
                f"The trained model suggests a {model_label.lower()} level of automation risk, while the LLM estimate is {llm_label.lower()}, so the signals are somewhat mixed."
            )
    elif model_label:
        parts.append(
            f"The trained model suggests a {model_label.lower()} level of automation risk for {matched_title.lower()}."
        )

    if rule_label:
        parts.append(
            f"The rule-based baseline rates the role as {rule_label.lower()}, which gives an additional structured reference point."
        )

    growth_pct = best_match.get("employment_change_percent_2024_34") if best_match else None

    if growth_pct is not None:
        if growth_pct > 10:
            parts.append(
                f"Employment is projected to grow strongly through 2034, with about {growth_pct:.1f}% growth, which suggests positive demand for the role."
            )
        elif growth_pct > 0:
            parts.append(
                f"Employment is projected to grow modestly through 2034, with about {growth_pct:.1f}% growth."
            )
        else:
            parts.append(
                f"Employment is projected to decline by about {abs(growth_pct):.1f}% through 2034, which may signal a weaker long-term outlook."
            )

    return " ".join(parts)


st.set_page_config(page_title="Job Automation Insights", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
h1, h2, h3 {
    letter-spacing: -0.02em;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(148,163,184,0.18);
    padding: 1rem;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("Job Automation Insights")
st.write(
    "This tool compares two automation estimates for a job: a trained machine learning model and a pure LLM-only estimate. "
    "It also shows a simple rule-based baseline built from selected onet job traits, along with projected US employment trends for the matched occupation over the next 10 years."
)
st.caption("Built on US occupation data and employment projections.")

job_title_input = st.text_input("Enter a job title", value="Data Scientist")

model, feature_columns = load_saved_model(MODEL_PATH)
job_features_df = load_job_features(JOB_FEATURES_PATH)

if "aws_result" not in st.session_state:
    st.session_state.aws_result = None
if "local_result" not in st.session_state:
    st.session_state.local_result = None
if "llm_prediction" not in st.session_state:
    st.session_state.llm_prediction = {}
if "explanation" not in st.session_state:
    st.session_state.explanation = None


st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #7c5295;
    color: white;
    border: none;
}
div.stButton > button:first-child:hover {
    background-color: #5c3b70;
    color: white;
}
</style>
""", unsafe_allow_html=True)

if st.button("Analyze Job"):
    if not job_title_input.strip():
        st.warning("Please enter a job title.")
    else:
        with st.spinner("Running analysis..."):
            try:
                aws_result = call_aws_api(job_title_input)

                best_match = aws_result.get("bestMatch", {})
                matched_title = best_match.get("matched_job_title") if best_match else None
                explanation = aws_result.get("explanation")
                llm_prediction = aws_result.get("llmPrediction", {})

                local_result = None
                if matched_title:
                    local_result = predict_local_model(
                        matched_title,
                        model,
                        feature_columns,
                        job_features_df
                    )

                st.session_state.aws_result = aws_result
                st.session_state.local_result = local_result
                st.session_state.llm_prediction = llm_prediction
                st.session_state.explanation = explanation

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.aws_result = None
                st.session_state.local_result = None
                st.session_state.llm_prediction = {}
                st.session_state.explanation = None

aws_result = st.session_state.aws_result
local_result = st.session_state.local_result
llm_prediction = st.session_state.llm_prediction
explanation = st.session_state.explanation

if aws_result:
    matched_titles = aws_result.get("matchedTitles", [])
    best_match = aws_result.get("bestMatch", {})

    st.markdown("## Matched Occupation")
    st.markdown(
        f"<div style='font-size:1.5rem; font-weight:600; margin-bottom:0.3rem;'>{best_match.get('matched_job_title', 'N/A')}</div>",
        unsafe_allow_html=True
    )

    st.markdown("## Automation Estimates")
    col1, col2 = st.columns(2)

    with col1:
        if local_result:
            model_prob = local_result["predicted_automation_probability"]
            model_label = probability_label(model_prob)
            st.markdown(
                metric_card(
                    "ML Automation Probability",
                    f"{model_prob:.2%}",
                    "Learned from occupation-level features",
                    "#2563eb"
                ),
                unsafe_allow_html=True
            )
            st.markdown(f"Model Risk Level: {risk_badge(model_label)}", unsafe_allow_html=True)
        else:
            st.info("No local model prediction found for the matched occupation.")

    with col2:
        llm_score = llm_prediction.get("automation_risk_score")
        llm_label = llm_prediction.get("automation_risk_label")
        llm_reasoning = llm_prediction.get("reasoning")

        if llm_score is not None:
            st.markdown(
                metric_card(
                    "LLM Automation Estimate",
                    f"{llm_score:.0f}%",
                    "Standalone text-based estimate",
                    "#7c3aed"
                ),
                unsafe_allow_html=True
            )
        if llm_label:
            st.markdown(f"LLM Risk Level: {risk_badge(llm_label)}", unsafe_allow_html=True)
        if llm_reasoning:
            st.caption(llm_reasoning)

    st.markdown("## Structured Baseline")
    aws_score = best_match.get("automation_risk_score")
    aws_label = best_match.get("automation_risk_label")

    baseline_col1, baseline_col2 = st.columns([1.2, 1])

    with baseline_col1:
        if aws_score is not None:
            st.markdown(
                metric_card(
                    "Rule-Based Risk Index",
                    f"{aws_score:.1f} / 100",
                    "A simple score built from job traits we manually selected from onet",
                    "#f59e0b"
                ),
                unsafe_allow_html=True
            )
        if aws_label:
            st.markdown(f"Rule-Based Risk Level: {risk_badge(aws_label)}", unsafe_allow_html=True)

    with baseline_col2:
        st.markdown("""
        <div style="
            border:1px solid rgba(148,163,184,0.18);
            border-radius:18px;
            padding:1rem 1.1rem;
            background:rgba(245,158,11,0.06);
            line-height:1.65;
            font-size:0.98rem;
            color:#cbd5e1;
        ">
            This is not a learned prediction like the ML model. It is a simple baseline score built by manually choosing a small set of onet job traits that seemed more or less automatable, then combining them into a single index.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## US Employment Outlook (2024–2034)")
    col4, col5, col6, col7 = st.columns(4)

    with col4:
        st.metric("Employment 2024", format_job_count_in_thousands(best_match.get("employment_2024")))
    with col5:
        st.metric("Employment 2034", format_job_count_in_thousands(best_match.get("employment_2034")))
    with col6:
        change_num = best_match.get("employment_change_numeric_2024_34")
        st.metric("Change (Jobs)", format_job_count_in_thousands(change_num) if change_num is not None else "N/A")
    with col7:
        change_pct = best_match.get("employment_change_percent_2024_34")
        st.metric("Change (%)", f"{change_pct:.1f}%" if change_pct is not None else "N/A")

    combined_takeaway = build_combined_takeaway(local_result, llm_prediction, best_match)

    if combined_takeaway:
        st.markdown("## Career Takeaway")
        st.markdown(f"""
        <div style="
            border:1px solid rgba(148,163,184,0.22);
            border-radius:18px;
            padding:1rem 1.1rem;
            background:rgba(59,130,246,0.07);
            line-height:1.7;
            font-size:1.03rem;
        ">
            {combined_takeaway}
        </div>
        """, unsafe_allow_html=True)

    with st.expander("See matched titles"):
        st.write(matched_titles)

    job_info = aws_result.get("jobInfo", {})
    matches = job_info.get("matches", [])

    if matches:
        st.markdown("## Skills Snapshot")
        first_match = matches[0]
        skills = first_match.get("skills", {})

        if skills:
            skill_rows = []
            for skill_name, elements in skills.items():
                avg_score = sum(elements.values()) / len(elements) if elements else None
                skill_rows.append({
                    "skill_group": skill_name,
                    "num_elements": len(elements),
                    "avg_score": round(avg_score, 2) if avg_score is not None else None
                })

            skill_df = pd.DataFrame(skill_rows).sort_values("avg_score", ascending=False)

            st.markdown("### Top Skill Groups")
            chart_df = skill_df.head(10).copy()

            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("avg_score:Q", title="Average Score"),
                    y=alt.Y("skill_group:N", sort="-x", title=None),
                    tooltip=["skill_group", "avg_score", "num_elements"]
                )
                .properties(height=400)
            )

            st.altair_chart(chart, use_container_width=True)

            st.markdown("### Skill Group Table")
            st.dataframe(skill_df, use_container_width=True)

    