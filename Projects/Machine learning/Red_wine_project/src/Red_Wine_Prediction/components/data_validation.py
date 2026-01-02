'''The DataValidation class reads the ingested dataset, 
verifies that all dataset columns match the defined schema,
and records the validation result in a status file.'''

from Red_Wine_Prediction import logger
from Red_Wine_Prediction.entity.config_entity import DataValidationConfig
import pandas as pd 
import os


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                else:
                    validation_status = True

                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e
