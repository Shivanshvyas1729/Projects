from dataclasses import dataclass 
from pathlib import Path    

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration class for the data ingestion stage.

    This class stores all paths and source information required
    to download, store, and extract the dataset.
    The `frozen=True` parameter makes the configuration immutable.
    """

    root_dir: Path
    # Root directory where all data ingestion artifacts will be stored

    source_URL: str
    # URL of the source from which the data will be downloaded

    local_data_file: Path
    # Path to the local file where the downloaded data will be saved

    unzip_dir: Path
    # Directory where the downloaded data will be extracted



@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration entity for the data validation stage.
    Holds all paths and schema information required to validate the dataset.
    """
    # Root directory where all data validation artifacts will be stored
    # Example: artifacts/data_validation/
    root_dir: Path

    # File path where validation status is written (success / failure)
    # Used to track whether data validation passed
    STATUS_FILE: str

    # Path to the dataset file that needs to be validated
    # Comes from the data ingestion stage
    unzip_data_dir: Path

    # Schema definition loaded from schema.yaml
    # Contains expected columns, data types, etc.
    all_schema: dict
    
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir:Path 
    data_path:Path
