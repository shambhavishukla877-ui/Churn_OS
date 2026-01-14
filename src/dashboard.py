import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from config import Config

API_URL = "http://127.0.0.1:8000"

try:
    SECURE_KEY = Config.API_KEY
except Exception:
    SECURE_KEY = "sk-proj-churn-secure-2026-v1"

st.set_page_config(page_title="Churn Intelligence Engine", page_icon="🧠", layout="wide")


def check_api_status():
    try:
        return requests.get(f"{API_URL}/").status_code == 200
    except Exception:
        return False


def make_prediction(features):
    try:
        r = requests.post(
            f"{API_URL}/predict_churn",
            json={"features": features},
            headers={"X-API-Key": SECURE_KEY}
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def analyze_text(text):
    try:
        r = requests.post(
            f"{API_URL}/analyze_complaint",
            json={"text": text},
            headers={"X-API-Key": SECURE_KEY}
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8637/8637106.png", width=80)
st.sidebar.title("Churn OS 2.0 (Enterprise)")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Navigation", ["🔍 Live Prediction", "📂 Batch Analysis", "🗣️ Complaint Decoder"])
st.sidebar.markdown("---")

api_status = check_api_status()
if api_status:
    st.sidebar.success("🟢 Brain Online (Secured)")
else:
    st.sidebar.error("🔴 Brain Offline")


# ---------------- LIVE PREDICTION ----------------
if app_mode == "🔍 Live Prediction":
    st.title("👤 Customer Risk Scanner")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    with col3:
        tech_support = st.selectbox("Tech Support?", ["Yes", "No", "No internet service"])
        online_security = st.selectbox("Online Security?", ["Yes", "No", "No internet service"])
        paperless = st.selectbox("Paperless Billing?", ["Yes", "No"])

    if st.button("🚀 Analyze Risk") and api_status:
        features = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet_service,
            "PaymentMethod": payment_method,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
            "PaperlessBilling": paperless,
            "gender": "Male",
            "Partner": "No",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
        }

        result = make_prediction(features)
        if result:
            risk_score = result["churn_risk_score"]
            risk_level = result["risk_level"]

            m1, m2 = st.columns([1, 2])
            with m1:
                st.metric("Churn Probability", f"{risk_score:.2%}", delta_color="inverse")

                if risk_level == "CRITICAL":
                    st.error(f"⚠️ STATUS: {risk_level}")
                elif risk_level == "HIGH":
                    st.warning(f"⚠️ STATUS: {risk_level}")
                else:
                    st.success(f"✅ STATUS: {risk_level}")

            with m2:
                drivers = result.get("key_drivers", [])
                if drivers:
                    df = pd.DataFrame(drivers)
                    fig = px.bar(df, x="impact_score", y="feature", orientation="h", color="type")
                    st.plotly_chart(fig, use_container_width=True)


# ---------------- BATCH ANALYSIS ----------------
elif app_mode == "📂 Batch Analysis":
    st.title("📊 Bulk Risk Processor")
    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

    if uploaded_file and api_status:
        df = pd.read_csv(uploaded_file)

        if st.button("Run Batch Prediction"):
            results = []
            limit = min(20, len(df))

            for i in range(limit):
                row = df.iloc[i].to_dict()
                pred = make_prediction(row)
                if pred:
                    r = row.copy()
                    r["Churn_Prob"] = pred["churn_risk_score"]
                    r["Risk_Level"] = pred["risk_level"]
                    results.append(r)

            res_df = pd.DataFrame(results)
            if not res_df.empty:
                st.metric("Critical Risk Customers Identified", len(res_df[res_df["Risk_Level"] == "CRITICAL"]))
                st.dataframe(res_df)


# ---------------- COMPLAINT DECODER ----------------
elif app_mode == "🗣️ Complaint Decoder":
    st.title("🧠 Cognitive NLP Engine")
    txt_input = st.text_area("Customer Complaint Text", height=150)

    if st.button("Decode Sentiment") and api_status and txt_input:
        nlp_res = analyze_text(txt_input)
        if nlp_res:
            sev = nlp_res["severity_score"]
            label = nlp_res["sentiment_label"]

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=sev, gauge={"axis": {"range": [None, 100]}}))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader(label)
                for k in nlp_res.get("keywords", []):
                    st.write(k)
