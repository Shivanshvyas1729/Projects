# Import all constant values such as file paths (CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH)
from Red_Wine_Prediction.constants import *

# Import common utility functions
# - read_yaml: reads YAML configuration files
# - create_directories: creates required directories if they do not exist
from Red_Wine_Prediction.utils.common import read_yaml, create_directories     

# Import configuration entity classes
# These classes define the structure of configuration objects
from Red_Wine_Prediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
    
)


class ConfigurationManager:
    """
    ConfigurationManager is responsible for:
    - Reading configuration, parameters, and schema files
    - Creating required directories
    - Providing configuration objects for different pipeline stages
    """

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH,
    ) -> None:
        # Read main configuration YAML file
        self.config = read_yaml(config_filepath)

        # Read parameters YAML file (model parameters, training params, etc.)
        self.params = read_yaml(params_filepath)

        # Read schema YAML file (column names, data types, validation rules)
        self.schema = read_yaml(schema_filepath)

        # Create the main artifacts directory defined in config
        create_directories([Path(self.config.artifacts_root)])
        
        
        
        
        

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Creates and returns the DataIngestionConfig object.

        This method:
        - Extracts data ingestion configuration from the main config
        - Creates required directories for data ingestion
        - Returns a structured DataIngestionConfig object
        """

        # Extract data ingestion section from the configuration
        config = self.config.data_ingestion

        # Create root directory for data ingestion artifacts
        create_directories([Path(config.root_dir)])

        # Create and return DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config





    def get_data_validation_config(self) -> DataValidationConfig:


            # Access the data_validation section from config.yaml
            config = self.config.data_validation

            # Access expected columns/schema from schema.yaml
            # Example: schema.yaml → COLUMNS:
            #              fixed_acidity: float
            #              alcohol: float
            schema = self.schema.COLUMNS

            # Create directory for data validation artifacts
            create_directories([config.root_dir])

            # Create DataValidationConfig object
            # This converts raw YAML values into a structured, immutable config
            data_validation_config = DataValidationConfig(
                root_dir=config.root_dir,
                STATUS_FILE=config.STATUS_FILE,
                unzip_data_dir=config.unzip_data_dir,
                all_schema=schema,
            )

            # Return the prepared configuration to the DataValidation component
            return data_validation_config
    
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config =self.config.data_transformation 
        
        create_directories([config.root_dir])
        
        Data_transformation_config =  DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            
            
        )
        
        return Data_transformation_config
    
    
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.ElasticNet
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            alpha = params.alpha,
            l1_ratio = params.l1_ratio,
            target_column = schema.name
            
        )

        return model_trainer_config
    
    
    
    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
        config=self.config.model_evaluation
        params=self.params.ElasticNet
        schema=self.schema.TARGET_COLUMN
        
        
        create_directories([config.root_dir])
        model_evaluation_config= ModelEvaluationConfig(
        
        
        root_dir = config.root_dir,
        test_data_path=config.test_data_path,
        model_path= config.model_path,
        metric_file_name = config.metric_file_name,
        all_params=params,
        target_column=schema.name 
        
        )
        
        
        return model_evaluation_config