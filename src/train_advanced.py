import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from config import Config


class AdvancedTrainer:
    def __init__(self):
        self.config = Config()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.best_model = None
        self.best_score = 0
        self.best_model_name = ""

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

    def tune_random_forest(self):
        rf = RandomForestClassifier(random_state=self.config.RANDOM_STATE)

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [1, 2, 4],
        }

        grid = GridSearchCV(rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
        grid.fit(self.X_train, self.y_train)

        return grid.best_estimator_, grid.best_score_

    def tune_xgboost(self):
        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=self.config.RANDOM_STATE,
        )

        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5, 7],
            "scale_pos_weight": [1, 3],
        }

        grid = GridSearchCV(xgb, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
        grid.fit(self.X_train, self.y_train)

        return grid.best_estimator_, grid.best_score_

    def evaluate_and_compare(self, model, name):
        y_prob = model.predict_proba(self.X_test)[:, 1]
        roc = roc_auc_score(self.y_test, y_prob)

        if roc > self.best_score:
            self.best_score = roc
            self.best_model = model
            self.best_model_name = name

    def execute_phase_3(self):
        self.load_artifacts()

        rf_model, _ = self.tune_random_forest()
        self.evaluate_and_compare(rf_model, "Random Forest")

        xgb_model, _ = self.tune_xgboost()
        self.evaluate_and_compare(xgb_model, "XGBoost")

        joblib.dump(
            self.best_model,
            os.path.join(self.config.PROCESSED_DATA_DIR, "final_churn_model.pkl"),
        )


if __name__ == "__main__":
    AdvancedTrainer().execute_phase_3()
