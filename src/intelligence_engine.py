import pandas as pd
import shap
import joblib
import os
from textblob import TextBlob
from config import Config
import numpy as np



class IntelligenceEngine:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.X_train = None
        self.feature_names = []
        self.nlp_model = None
        self._load_resources()

    def _load_resources(self):
        self.model = joblib.load(
            os.path.join(self.config.PROCESSED_DATA_DIR, "final_churn_model.pkl")
        )
        self.preprocessor = joblib.load(
            os.path.join(self.config.PROCESSED_DATA_DIR, "preprocessor.pkl")
        )
        self.X_train = joblib.load(
            os.path.join(self.config.PROCESSED_DATA_DIR, "X_train.pkl")
        )

        nlp_path = os.path.join(self.config.PROCESSED_DATA_DIR, "nlp_severity_model.pkl")
        if os.path.exists(nlp_path):
            self.nlp_model = joblib.load(nlp_path)

        try:
            self.feature_names = self.preprocessor.get_feature_names_out()
        except Exception:
            self.feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]

    def generate_explanation(self, customer_data_row):
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)

        shap_values = self.explainer.shap_values(customer_data_row)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        names = (
            self.feature_names
            if len(self.feature_names) == len(sv)
            else [f"Feature_{i}" for i in range(len(sv))]
        )

        impact_dict = dict(zip(names, sv))
        return sorted(
    impact_dict.items(),
    key=lambda x: abs(float(np.mean(x[1]))),
    reverse=True
)[:3]


    def analyze_complaint(self, text_complaint):
        if not text_complaint or pd.isna(text_complaint):
            return {"sentiment_label": "Neutral", "severity_score": 0.0, "keywords": []}

        blob = TextBlob(text_complaint)
        keywords = blob.noun_phrases

        if self.nlp_model:
            pred_class = self.nlp_model.predict([text_complaint])[0]
            pred_prob = self.nlp_model.predict_proba([text_complaint])[0][1]
            severity_score = round(pred_prob * 100, 2)
            label = "CRITICAL" if pred_class == 1 else "NORMAL"
        else:
            sentiment = blob.sentiment.polarity
            severity_score = round((1 - sentiment) * 50, 2)
            label = "CRITICAL" if severity_score > 75 else "NORMAL"

        return {
            "sentiment_label": label,
            "severity_score": severity_score,
            "keywords": keywords,
        }
