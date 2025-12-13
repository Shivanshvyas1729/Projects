# Standard library imports
import os            # Provides functions for interacting with the operating system
import sys           # Access to system-specific parameters and functions
import warnings      # Used to ignore or control warnings in the script

# Data manipulation and numerical computing
import pandas as pd  # Pandas for data loading, cleaning, and manipulation
import numpy as np   # NumPy for numerical operations and arrays

# Scikit-learn modules for data splitting and evaluation metrics
from sklearn.model_selection import train_test_split   # Splits data into train and test sets
from sklearn.metrics import (                        # Model performance metrics
    mean_absolute_error, 
    mean_squared_error,
    r2_score
)

# Scikit-learn regression models
from sklearn.linear_model import LinearRegression    # Linear Regression model
from sklearn.linear_model import ElasticNet          # ElasticNet Regression (L1 + L2 regularization)

# Utilities
from urllib.parse import urlparse                    # Helps extract components from URLs

# MLflow for experiment tracking
import mlflow                                        # Main MLflow library
from mlflow.models.signature import infer_signature  # Used to capture model input/output schema
import mlflow.sklearn                                # MLflow support for sklearn models

# DagsHub integration for remote MLflow tracking
import dagshub
dagshub.init(repo_owner='shivanshvyas29', repo_name='mlflow', mlflow=True)                                      # Connect MLflow tracking to DagsHub

# Logging module for structured application logs
import logging                                        # Used to write logs (info, warning, error)


 



# Configure logging to display only warnings, errors, and critical messages
logging.basicConfig(level=logging.WARN)

# Create a logger for this file; __name__ represents the module's name
logger = logging.getLogger(__name__)

# If you run the file directly,
# __name__ = "__main__"

# If the file is imported,
# __name__ = "filename"


def eval_metrics(actual, pred):
    """Evaluate and return regression metrics: MAE, MSE, RMSE, R2."""
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, pred)
    return mae, mse, rmse, r2


if __name__ == "__main__": #--->dought #
    warnings.filterwarnings("ignore")  # Ignore all warnings
    np.random.seed(40)                 # Set random seed for reproducibility
    
    # Read the wine-quality csv file from the URL 
    csv_url = r"C:\Users\DELL\OneDrive\Desktop\Projects\Projects\Temp files\Pratices folder\data\winequality-red.csv"


    try:
        data = pd.read_csv(csv_url)
    
    except Exception as e:
      logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e
    )
      
      
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, test_size=0.25) 
    
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    



# ✅ Memory Lines for sys.argv

# sys.argv is always a list.

# sys.argv[0] is always the script name — even with no arguments.

# User inputs start from sys.argv[1].

# Use len(sys.argv) > 1 to check if the first user argument exists.

# Use len(sys.argv) > 2 to check if the second user argument exists.

# If the argument doesn't exist → use a default value.

# ⭐ One-line formula to remember

# Index 0 = file name,
# Index 1+ = real arguments.
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5 
    
    
    
    with mlflow.start_run():
      lr=ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
      
      lr.fit(train_x,train_y) 
      
      predicted_qualities = lr.predict(test_x) 
      
      (mae, mse , rmse, r2) = eval_metrics(test_y, predicted_qualities)
      
      print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f})")
      
      print("  MAE: %s" % mae)
      print("  MSE: %s" % mse)
      print("  RMSE: %s" % rmse)
      print("  R2: %s" % r2) 
      
      mlflow.log_param("alpha", alpha)
      mlflow.log_param("l1_ratio", l1_ratio)
      mlflow.log_metric("mae", mae)
      mlflow.log_metric("mse", mse)
      mlflow.log_metric("rmse", rmse)
      mlflow.log_metric("r2", r2)
      
      
      #for remote tracking url of dagshub
      remote_server_uri = "https://dagshub.com/shivanshvyas29/mlflow.mlflow"
      mlflow.set_tracking_uri(remote_server_uri) 
      
      tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
      
      #model registry does not work with file store
      if tracking_url_type_store != "file":
          #register the model
          #there are other ways to use the model registry, which depends on the use case,
          #please refer to the doc for more information:
          #https://mlflow.org/docs/latest/model-registry.html#api-workflow
          mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
          
      else:
          mlflow.sklearn.log_model(lr, "model")
          
# ✅ Explanation of Remote Model Registration with MLflow and DagsHub
       
#     🌟 Simple Explanation (Like you’re 10 years old)
# ✔️ You are telling MLflow:

# “Save my experiment on DAGsHub (online), not on my laptop.”

# ✔️ Then MLflow checks:

# “Am I saving online or just in a local file?”

# ✔️ If saving ONLINE →

# MLflow allows Model Registry (official model storage)

# ✔️ If saving LOCALLY →

# MLflow cannot use the Model Registry
# so it only saves the model normally.

# 🌈 Line-by-line SIMPLE meaning
# ✅ 1. Tell MLflow where to store tracking data
# remote_server_uri = "https://dagshub.com/shivanshvyas29/mlflow.mlflow"
# mlflow.set_tracking_uri(remote_server_uri)


# 👉 “MLflow, store everything on my DAGsHub project online.”

# ✅ 2. Check the storage type
# tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


# 👉 “Let me check if the storage is online or local.”
# Examples of scheme:

# "https" → online

# "file" → local folder

# ✅ 3. If storage is NOT a file → means ONLINE
# if tracking_url_type_store != "file":


# 👉 “Yes! We are saving online. So we can use the Model Registry.”

# ✅ 4. Register the model online
# mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")


# 👉 “Save and register the model under the name ElasticnetWineModel.”

# This makes it appear in DAGsHub under Model Registry.

# ❌ Else (if storage is a file)

# (local MLflow folder)

# else:
#     mlflow.sklearn.log_model(lr, "model")


# 👉 “Can't register because you’re storing locally. Just save the model normally.”