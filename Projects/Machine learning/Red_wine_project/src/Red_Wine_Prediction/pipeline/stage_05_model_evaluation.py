
from Red_Wine_Prediction import logger
from Red_Wine_Prediction.config.configuration import ConfigurationManager
from Red_Wine_Prediction.components.model_evaluation import ModelEvaluation


STAGE_NAME="Model evaluation stage"

class ModelEvaluationPipeline:
    
    def __init__(self):
        pass


    def main(self):
        
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.save_results()



