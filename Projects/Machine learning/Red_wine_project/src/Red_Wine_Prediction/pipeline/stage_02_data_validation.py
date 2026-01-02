
from Red_Wine_Prediction import logger
from Red_Wine_Prediction.config.configuration import ConfigurationManager
from Red_Wine_Prediction.components.data_validation import DataValidation


STAGE_NAME="Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()






