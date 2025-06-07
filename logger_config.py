import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Create timestamped log file name
log_filename = datetime.now().strftime("logs/app_%Y%m%d_%H%M%S.log")

# Define filter to exclude WARNING level logs
class NoWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Add filter to all handlers
logger = logging.getLogger()
for handler in logger.handlers:
    handler.addFilter(NoWarningFilter())
