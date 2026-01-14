import pandas as pd
import joblib
import os
from config import Config


class ErrorAnalyzer:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.X_test = None
        self.y_test = None

    def load_resources(self):
        model_path = os.path.join(self.config.PROCESSED_DATA_DIR, "final_churn_model.pkl")
        self.model = joblib.load(model_path)

        self.X_test = joblib.load(
            os.path.join(self.config.PROCESSED_DATA_DIR, "X_test.pkl")
        )
        self.y_test = joblib.load(
            os.path.join(self.config.PROCESSED_DATA_DIR, "y_test.pkl")
        )

    def analyze_errors(self):
        if self.model is None:
            return

        y_pred = self.model.predict(self.X_test)

        feature_cols = self.config.NUMERICAL_COLS + list(
            range(self.X_test.shape[1] - len(self.config.NUMERICAL_COLS))
        )

        errors_df = pd.DataFrame(self.X_test, columns=feature_cols)
        errors_df["True_Label"] = self.y_test
        errors_df["Predicted_Label"] = y_pred

        mistakes = errors_df[
            errors_df["True_Label"] != errors_df["Predicted_Label"]
        ]

        save_path = os.path.join(
            self.config.PROCESSED_DATA_DIR, "error_cases.csv"
        )
        mistakes.to_csv(save_path, index=False)


if __name__ == "__main__":
    analyzer = ErrorAnalyzer()
    analyzer.load_resources()
    analyzer.analyze_errors()
