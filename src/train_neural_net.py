import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score
from config import Config


class NeuralNetTrainer:
    def __init__(self):
        self.config = Config()
        self.model = None
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

    def build_model(self, input_dim):
        model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(input_dim,)),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])
        return model

    def train(self):
        self.load_artifacts()

        input_dim = self.X_train.shape[1]
        self.model = self.build_model(input_dim)

        stopper = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[stopper],
            verbose=1,
        )

        self.evaluate()

    def evaluate(self):
        y_prob = self.model.predict(self.X_test).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        roc = roc_auc_score(self.y_test, y_prob)

        if roc > 0.8261:
            self.model.save(
                os.path.join(self.config.PROCESSED_DATA_DIR, "final_churn_model.h5")
            )


if __name__ == "__main__":
    NeuralNetTrainer().train()
