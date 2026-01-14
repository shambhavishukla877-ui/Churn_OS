import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from config import Config


class NLPTrainer:
    def __init__(self):
        self.config = Config()
        self.data = pd.DataFrame(
            {
                "text": [
                    "I am cancelling my service immediately.",
                    "Hidden fees on my bill are unacceptable.",
                    "Internet has been down for 3 days.",
                    "Technician never showed up, I am furious.",
                    "This is a scam, I want a refund.",
                    "Your customer service is useless.",
                    "I am switching to a competitor.",
                    "Speed is terrible, nothing loads.",
                    "Stop calling me or I will sue.",
                    "My account was charged twice!",
                    "How do I upgrade my plan?",
                    "I want to change my payment method.",
                    "When does my contract expire?",
                    "Is there a discount for seniors?",
                    "Thank you for fixing the issue.",
                    "The service is okay but could be faster.",
                    "Just checking my balance.",
                    "I need to move my service to a new address.",
                    "Good morning, I have a question.",
                    "Can you explain this charge?",
                    "I love the new speed, thanks.",
                    "No issues so far.",
                    "How do I reset my router?",
                    "Please update my email address.",
                    "Is fiber available in my area?",
                ]
                * 10,
                "severity": [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
                * 10,
            }
        )

    def train(self):
        X = self.data["text"]
        y = self.data["severity"]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english")),
                ("clf", LogisticRegression(random_state=42)),
            ]
        )

        pipeline.fit(X_train, y_train)

        joblib.dump(
            pipeline,
            os.path.join(self.config.PROCESSED_DATA_DIR, "nlp_severity_model.pkl"),
        )


if __name__ == "__main__":
    NLPTrainer().train()
