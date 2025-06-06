"""
"""
import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Create timestamped log file name
log_filename = datetime.now().strftime("logs/app_%Y%m%d_%H%M%S.log")

# Configure root logger
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG if needed
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
