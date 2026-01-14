import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from config import Config


class DataEngineer:
    def __init__(self):
        self.config = Config()
        self.preprocessor = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        df = pd.read_csv(self.config.DATA_PATH_CHURN)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if self.config.ID_COLUMN in df.columns:
            df = df.drop(columns=[self.config.ID_COLUMN])
        return df

    def augment_features(self, df):
        df = df.copy()
        df["Tenure_Monthly_Interaction"] = df["tenure"] * df["MonthlyCharges"]
        contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
        contract_weight = df["Contract"].map(contract_map).fillna(1)
        df["Contract_Tenure_Interaction"] = contract_weight * df["tenure"]
        return df

    def handle_outliers(self, df):
        for col in self.config.NUMERICAL_COLS:
            if col in df.columns:
                m = df[col].mean()
                s = df[col].std()
                up = m + self.config.OUTLIER_THRESHOLD * s
                low = m - self.config.OUTLIER_THRESHOLD * s
                df[col] = np.where(df[col] > up, up, df[col])
                df[col] = np.where(df[col] < low, low, df[col])
        return df

    def feature_engineering_pipeline(self):
        num = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.config.MISSING_VAL_STRATEGY)),
                ("scaler", StandardScaler()),
            ]
        )
        cat = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", num, self.config.NUMERICAL_COLS),
                ("cat", cat, self.config.CATEGORICAL_COLS),
            ]
        )

    def process_data(self):
        df = self.load_data()
        df = self.augment_features(df)
        df = self.handle_outliers(df)

        X = df.drop(columns=[self.config.TARGET_COLUMN])
        y = self.label_encoder.fit_transform(df[self.config.TARGET_COLUMN])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y,
        )

        self.preprocessor = self.feature_engineering_pipeline()
        X_train_p = self.preprocessor.fit_transform(X_train)
        X_test_p = self.preprocessor.transform(X_test)

        X_train_r, y_train_r = SMOTE(random_state=self.config.RANDOM_STATE).fit_resample(
            X_train_p, y_train
        )

        self.save_artifacts(X_train_r, X_test_p, y_train_r, y_test)
        return X_train_r, y_train_r

    def save_artifacts(self, X_train, X_test, y_train, y_test):
        joblib.dump(X_train, os.path.join(self.config.PROCESSED_DATA_DIR, "X_train.pkl"))
        joblib.dump(X_test, os.path.join(self.config.PROCESSED_DATA_DIR, "X_test.pkl"))
        joblib.dump(y_train, os.path.join(self.config.PROCESSED_DATA_DIR, "y_train.pkl"))
        joblib.dump(y_test, os.path.join(self.config.PROCESSED_DATA_DIR, "y_test.pkl"))
        joblib.dump(
            self.preprocessor,
            os.path.join(self.config.PROCESSED_DATA_DIR, "preprocessor.pkl"),
        )


if __name__ == "__main__":
    DataEngineer().process_data()
