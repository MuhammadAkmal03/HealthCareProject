# import logging
# import os
# from datetime import datetime

# # --- Set the path for the log file ---
# # This creates a log file in a 'logs' folder within the current module's directory.
# # This assumes the logger.py file is in DeepLearning_module/notebook/
# MODULE_DIR = os.path.dirname(__file__)
# LOGS_PATH = os.path.join(MODULE_DIR, "logs")
# os.makedirs(LOGS_PATH, exist_ok=True)

# # Create a unique log file name with a timestamp
# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# LOG_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE)

# # --- Configure the logger ---
# # Create a logger instance with a custom name
# logger = logging.getLogger("deeplearning_logger")
# logger.setLevel(logging.INFO)

# # Create a file handler to write log messages to a file
# file_handler = logging.FileHandler(LOG_FILE_PATH)

# # Create a formatter for the log messages
# formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
# file_handler.setFormatter(formatter)

# # Add the file handler to the logger
# logger.addHandler(file_handler)

# # --- If you also want to see logs in the console ---
# # Create a console handler
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)
# DeepLearning_module/notebook/logger.py

import logging
import os
from datetime import datetime

# Log file named by timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# NOTE: This path is now relative to where the notebook is run
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the root logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Create a specific logger instance to use in your notebook
logger = logging.getLogger("chest_xray_logger")
