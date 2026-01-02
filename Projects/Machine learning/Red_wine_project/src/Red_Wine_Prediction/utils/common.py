# 📌 PURPOSE OF THIS FILE (utils/common.py)

# This file contains reusable utility/helper functions that are used across multiple ML pipeline stages such as:

# configuration loading

# directory creation

# saving/loading artifacts (JSON, models)

# logging

# file size inspection

# 👉 This avoids code duplication and keeps the project clean & modular.

import os
import json
import yaml
import joblib

from pathlib import Path
from typing import Any

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations #@ensure_annotations = runtime type checker for function annotations

from Red_Wine_Prediction import logger


# --------------------------------------------------
# read_yaml
# --------------------------------------------------
# PURPOSE:
# - Reads YAML configuration files (config.yaml, params.yaml, schema.yaml)
# - Returns data in ConfigBox format for dot-notation access
#
# WHY ConfigBox?
# Instead of:
#   config["data_ingestion"]["root_dir"]
# You can write:
#   config.data_ingestion.root_dir
# --------------------------------------------------
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as ConfigBox.-> direct access to yaml file data using dot notation.

    Args:
        path_to_yaml (Path): Path to YAML file

    Raises:
        ValueError: If YAML file is empty

    Returns:
        ConfigBox: YAML content as attribute-accessible object
    """
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:

            content = yaml.safe_load(yaml_file)

            # Log successful YAML loading
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")

            return ConfigBox(content)
#What is BoxValueError?

# BoxValueError is an exception raised by the python-box library when it cannot convert data into a ConfigBox object.

# In simple words:

# It means the data you are trying to convert into ConfigBox is invalid (most commonly EMPTY).
    except BoxValueError:
        # Raised when YAML file is empty
        raise ValueError("YAML file is empty")

    except Exception as e:
        # Catch any other exception
        raise e


# --------------------------------------------------
# create_directories
# --------------------------------------------------
# PURPOSE:
# - Creates multiple directories safely
# - Used during artifact creation in pipelines
#
# WHY exist_ok=True?
# Prevents error if directory already exists
# --------------------------------------------------
@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Creates a list of directories.

    Args:
        path_to_directories (list): List of directory paths
        verbose (bool): Whether to log directory creation
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)

        if verbose:
            logger.info(f"Directory created at: {path}")


# save_json
# --------------------------------------------------
# PURPOSE:
# - Saves dictionaries as JSON files
# - Commonly used for metrics, reports, configs
# --------------------------------------------------
@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves dictionary data to a JSON file.

    Args:
        path (Path): Path where JSON file will be saved
        data (dict): Data to save
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file saved at: {path}")


# --------------------------------------------------
# load_json
# --------------------------------------------------
# PURPOSE:
# - Loads JSON file and returns ConfigBox
# - Allows attribute-style access to JSON content
# --------------------------------------------------
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads JSON data from file.

    Args:
        path (Path): Path to JSON file

    Returns:
        ConfigBox: JSON data as ConfigBox
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


# --------------------------------------------------
# save_bin
# --------------------------------------------------
# PURPOSE:
# - Saves Python objects in binary format
# - Typically used for trained ML models
# --------------------------------------------------
@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves any Python object as a binary file.

    Args:
        data (Any): Object to save (model, transformer, etc.)
        path (Path): Path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


# --------------------------------------------------
# load_bin
# --------------------------------------------------
# PURPOSE:
# - Loads binary objects (models, encoders)
# --------------------------------------------------
@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads a binary file.

    Args:
        path (Path): Path to binary file

    Returns:
        Any: Loaded Python object
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


# --------------------------------------------------
# get_size
# --------------------------------------------------
# PURPOSE:
# - Returns file size in KB
# - Useful for logging & monitoring artifacts
# --------------------------------------------------
@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns file size in KB.

    Args:
        path (Path): Path to file

    Returns:
        str: File size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"



# common.py contains reusable utility functions for configuration loading, directory creation, artifact saving/loading, and logging.
# It helps maintain clean, modular, and reusable ML pipelines.