import os
from pathlib import Path
import logging

# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s'
)

# ----------------------------
# Project name
# ----------------------------
project_name = "Red_Wine_Prediction"

# ----------------------------
# Future Enhancements (Roadmap)
# ----------------------------
# DVC            → data & model versioning
# MLflow         → experiment tracking
# Feature store  → centralized feature management
# Model monitoring → drift & performance tracking
# CI/CD pipelines → automated testing & deployment
# Airflow / Prefect → pipeline orchestration
# Kubernetes     → scaling
# Real-time inference → low-latency predictions
# Agentic AI layer → autonomous decision-making

# ----------------------------
# List of files to be created
# ----------------------------
list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
    
]

# ----------------------------
# Create directories and files
# ----------------------------
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent
    filename = filepath.name

    if filedir != Path('.'):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Directory ensured: {filedir}")

    if not filepath.exists():
        filepath.touch()
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filename}")

