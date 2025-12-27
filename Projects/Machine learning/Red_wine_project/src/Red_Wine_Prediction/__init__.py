import os
import sys
import logging

# ----------------------------
# Logging Level Configuration
# ----------------------------
# Available levels:
# DEBUG    → Detailed debugging information
# INFO     → General application flow
# WARNING  → Unexpected behavior, but program continues
# ERROR    → A failure in execution
# CRITICAL → Severe error, program must stop
#
# Default level is INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ----------------------------
# Log Directory & File Path
# ----------------------------
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create logs directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# ----------------------------
# Log Format
# ----------------------------
# asctime   → Timestamp
# levelname → Log level (INFO, ERROR, etc.)
# module    → Python module name
# message   → Log message
logging_format = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format=logging_format,
    handlers=[
        # 1️⃣ Write logs to a file
        logging.FileHandler(log_filepath),

        # 2️⃣ Print logs to console (stdout)
        logging.StreamHandler(sys.stdout)
    ]
)

# ----------------------------
# Project-Level Logger
# ----------------------------
# Use this logger across the entire ML project
logger = logging.getLogger("mlProjectLogger")
