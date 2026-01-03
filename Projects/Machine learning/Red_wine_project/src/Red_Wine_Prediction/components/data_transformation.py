# components/data_transformation.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from Red_Wine_Prediction import logger
from Red_Wine_Prediction.entity.config_entity import DataTransformationConfig

# component

import os
import numpy as np
import pandas as pd
import joblib

from Red_Wine_Prediction import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataTransformation:
    """
    DataTransformation handles:
    - Feature engineering
    - Feature scaling
    - Train-test split
    - Saving transformed artifacts
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.scaler = StandardScaler()

    def train_test_splitting(self) -> None:
        # -----------------------------
        # Load data
        # -----------------------------
        data = pd.read_csv(self.config.data_path)
        logger.info("Data loaded successfully")

        # -----------------------------
        # 🔧 FEATURE ENGINEERING
        # -----------------------------
        data["log_residual_sugar"] = np.log1p(data["residual sugar"])
        data["log_chlorides"] = np.log1p(data["chlorides"])
        data["log_free_sulfur_dioxide"] = np.log1p(data["free sulfur dioxide"])
        data["log_total_sulfur_dioxide"] = np.log1p(data["total sulfur dioxide"])
        data["log_sulphates"] = np.log1p(data["sulphates"])
        data["alcohol_density_ratio"] = data["alcohol"] / (data["density"] + 1e-6)

        logger.info("Feature engineering completed")

        # -----------------------------
        # Separate features & target
        # -----------------------------
        X = data.drop(columns=["quality"])
        y = data["quality"]

        # -----------------------------
        # Train-Test Split (75/25)
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42
        )

        # -----------------------------
        # 🔥 FEATURE SCALING (NO LEAKAGE)
        # -----------------------------
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

        # Reattach target
        train = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
        test = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)

        # -----------------------------
        # Save artifacts (UNCHANGED STYLE)
        # -----------------------------
        os.makedirs(self.config.root_dir, exist_ok=True)

        train_path = os.path.join(self.config.root_dir, "train.csv")
        test_path = os.path.join(self.config.root_dir, "test.csv")
        scaler_path = os.path.join(self.config.root_dir, "scaler.joblib")

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        joblib.dump(self.scaler, scaler_path)

        # -----------------------------
        # Logging
        # -----------------------------
        logger.info("Train-test split completed")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")
        logger.info(f"Scaler saved at: {scaler_path}")

        print(train.shape)
        print(test.shape)
