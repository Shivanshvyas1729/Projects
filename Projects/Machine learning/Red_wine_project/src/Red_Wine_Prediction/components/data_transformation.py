# components/data_transformation.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from Red_Wine_Prediction import logger
from Red_Wine_Prediction.entity.config_entity import DataTransformationConfig


class DataTransformation:
    """
    DataTransformation handles:
    - Reading validated dataset
    - Performing train-test split
    - Saving transformed artifacts for downstream stages
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Initialize DataTransformation with configuration.

        Args:
            config (DataTransformationConfig): paths and parameters
        """
        self.config = config

    def train_test_splitting(self) -> None:
        """
        Splits the dataset into train and test sets
        and saves them as CSV artifacts.
        """

        # Read dataset from previous stage
        data = pd.read_csv(self.config.data_path)

        # Ensure transformation artifact directory exists
        os.makedirs(self.config.root_dir, exist_ok=True)

        # Perform reproducible train-test split (75% train, 25% test)
        train, test = train_test_split(
            data,
            test_size=0.25,
            random_state=42
        )

        # Save transformed datasets
        train_path = os.path.join(self.config.root_dir, "train.csv")
        test_path = os.path.join(self.config.root_dir, "test.csv")

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        # Logging
        logger.info("Train-test split completed")
        logger.info(f"Train data saved at: {train_path} | Shape: {train.shape}")
        logger.info(f"Test data saved at: {test_path} | Shape: {test.shape}")
