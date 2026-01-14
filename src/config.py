import os


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH_CHURN = os.path.join(
        BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

    API_KEY = "sk-proj-churn-secure-2026-v1"

    TARGET_COLUMN = "Churn"
    ID_COLUMN = "customerID"
    MISSING_VAL_STRATEGY = "median"
    OUTLIER_THRESHOLD = 3.0
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    NUMERICAL_COLS = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Tenure_Monthly_Interaction",
        "Contract_Tenure_Interaction",
    ]

    CATEGORICAL_COLS = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
