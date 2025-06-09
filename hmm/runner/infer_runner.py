"""
"""

import os
import pickle
import logging

import logger_config
from hmm.runner.base_runner import BaseRunner
from hmm.infer.models_inferencing_processor import ModelsInferenceProcessor

logger = logging.getLogger(__name__)


class InferRunner(BaseRunner):
    def run(self):
        for ticker in self.config["tickers"]:
            logger.info(f"Running inference for {ticker}")
            self.config["current_end"] = self.config["end_date"]
            self.config["current_start"] = self.config["start_date"]
            model = ModelsInferenceProcessor(config=self.config, ticker=ticker)
            inferencing = model.process()
            file_path = os.path.join("hmm", "infer", "artifacts", "inferencing", f"{ticker}.pkl")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(inferencing, f)
