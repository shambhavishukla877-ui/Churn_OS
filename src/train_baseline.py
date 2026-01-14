import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from config import Config


class BaselineTrainer:
    def __init__(self):
        self.config = Config()
        self.model = LogisticRegression(
            random_state=self.config.RANDOM_STATE, max_iter=1000
        )
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_artifacts(self):
        self.X_train = joblib.load(
            os.path.join(self.config.PROCESSED_DATA_DIR, "X_train.pkl")
        )
        self.X_test = joblib.load(
            os.path.join(self.config.PROCESSED_DATA_DIR, "X_test.pkl")
        )
        self.y_train = joblib.load(
            os.path.join(self.config.PROCESSED_DATA_DIR, "y_train.pkl")
        )
        self.y_test = joblib.load(
            os.path.join(self.config.PROCESSED_DATA_DIR, "y_test.pkl")
        )

    def perform_cross_validation(self):
        kfold = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.config.RANDOM_STATE
        )
        cross_val_score(self.model, self.X_train, self.y_train, cv=kfold, scoring="roc_auc")

    def train_and_evaluate(self):
        self.model.fit(self.X_train, self.y_train)
        joblib.dump(
            self.model,
            os.path.join(self.config.PROCESSED_DATA_DIR, "baseline_logistic_model.pkl"),
        )


if __name__ == "__main__":
    trainer = BaselineTrainer()
    trainer.load_artifacts()
    trainer.perform_cross_validation()
    trainer.train_and_evaluate()
