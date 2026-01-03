
from Red_Wine_Prediction import logger
from Red_Wine_Prediction.config.configuration import ConfigurationManager
from Red_Wine_Prediction.components.model_trainer import ModelTrainer


STAGE_NAME="Model Trainer stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()



