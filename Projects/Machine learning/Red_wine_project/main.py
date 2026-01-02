from Red_Wine_Prediction import logger

from Red_Wine_Prediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Red_Wine_Prediction.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from Red_Wine_Prediction.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx")
except Exception as e:
    logger.exception(f"Error occurred in stage {STAGE_NAME}.\n{e}")
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx")
except Exception as e:
    logger.exception(f"Error occurred in stage {STAGE_NAME}.\n{e}")
    raise e



STAGE_NAME = "Data Transformation stage"


try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_transformation= DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx")
except Exception as e:
    logger.exception(f"Error occurred in stage {STAGE_NAME}.\n{e}")
    raise e