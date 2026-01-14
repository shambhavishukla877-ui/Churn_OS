import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from intelligence_engine import IntelligenceEngine
from config import Config

config = Config()
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == config.API_KEY:
        return api_key_header
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


app = FastAPI(
    title="Churn Intelligence Engine (Enterprise)",
    version="2.1.0",
    description="Secured Enterprise AI API with Feature Engineering",
)

try:
    brain = IntelligenceEngine()
except Exception:
    brain = None


class CustomerData(BaseModel):
    features: dict


class ComplaintData(BaseModel):
    text: str


@app.get("/")
def health_check():
    if brain and brain.model:
        return {"status": "active", "version": "2.1.0", "security": "enabled"}
    return {"status": "critical"}


@app.post("/predict_churn")
def predict_churn(data: CustomerData, api_key: str = Security(get_api_key)):
    if not brain:
        raise HTTPException(503)

    input_df = pd.DataFrame([data.features])

    if "tenure" in input_df.columns and "MonthlyCharges" in input_df.columns:
        input_df["Tenure_Monthly_Interaction"] = input_df["tenure"] * input_df["MonthlyCharges"]
    else:
        input_df["Tenure_Monthly_Interaction"] = 0

    contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    c_val = input_df["Contract"].iloc[0]
    c_weight = contract_map.get(c_val, 1)
    input_df["Contract_Tenure_Interaction"] = c_weight * input_df["tenure"]

    if brain.preprocessor:
        processed_array = brain.preprocessor.transform(input_df)
    else:
        processed_array = input_df.values

    if hasattr(brain.model, "predict_proba"):
        risk_prob = brain.model.predict_proba(processed_array)[0][1]
    else:
        risk_prob = brain.model.predict(processed_array).flatten()[0]

    processed_row = processed_array[0].reshape(1, -1)
    top_factors = brain.generate_explanation(processed_row)

    reasons = []
    for feat, impact in top_factors:
        impact_val = float(np.mean(impact))
        clean_name = feat.replace("num__", "").replace("cat__", "")
        reasons.append({
            "feature": clean_name,
            "impact_score": round(impact_val, 4),
            "type": "Risk Increaser" if impact_val > 0 else "Risk Reducer"
        })

    return {
        "churn_risk_score": round(float(risk_prob), 4),
        "risk_level": "CRITICAL" if risk_prob > 0.7 else "HIGH" if risk_prob > 0.5 else "SAFE",
        "key_drivers": reasons,
    }


@app.post("/analyze_complaint")
def analyze_complaint(data: ComplaintData, api_key: str = Security(get_api_key)):
    if not brain:
        raise HTTPException(503)
    return brain.analyze_complaint(data.text)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
