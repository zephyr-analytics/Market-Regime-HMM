"""
Module for the TrainRunner class.
"""

import os
import pickle
import logging

import logger_config
from hmm.runner.base_runner import BaseRunner
from hmm.train.models_training_processor import ModelsTrainingProcessor

logger = logging.getLogger(__name__)


class TrainRunner(BaseRunner):
    """
    Class for running train tasks.
    """
    def run(self):
        """
        Method for processing the run pipeline.
        """
        for ticker in self.config["tickers"]:
            logger.info(f"Training model for {ticker}")
            self.config["current_end"] = self.config["end_date"]
            self.config["current_start"] = self.config["start_date"]
            model = ModelsTrainingProcessor(config=self.config, data=self.data, ticker=ticker)
            training = model.process()
            file_path = os.path.join("hmm", "train", "artifacts", "training", f"{ticker}.pkl")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(training, f)
