import logging
import os
import warnings
import numpy as np  # Just for demonstration if you test with numpy

# Suppress runtime warnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Use a fixed log file name
log_filename = "logs/app.log"

# Define the filter to suppress convergence warnings
class SuppressConvergenceWarnings(logging.Filter):
    def filter(self, record):
        return "Model is not converging" not in record.getMessage()

# Configure basic logging
file_handler = logging.FileHandler(log_filename, mode='a')
file_handler.addFilter(SuppressConvergenceWarnings())

stream_handler = logging.StreamHandler()
stream_handler.addFilter(SuppressConvergenceWarnings())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[file_handler, stream_handler]
)
