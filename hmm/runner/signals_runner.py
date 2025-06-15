import os
import logging
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

from hmm.runner.base_runner import BaseRunner
from hmm.train.models_training_processor import ModelsTrainingProcessor
from hmm.infer.models_inferencing_processor import ModelsInferenceProcessor
from hmm.build.portfolio_processor import PortfolioProcessor

logger = logging.getLogger(__name__)

class SignalsRunner(BaseRunner):
    """
    Runner for single-instance model training, inferencing, and portfolio construction.
    """
    def run(self):
        original_start = datetime.strptime(self.config["start_date"], "%Y-%m-%d")
        final_end = datetime.strptime(self.config["end_date"], "%Y-%m-%d")
        initial_train_years = self.config["model_warmup"]
        max_train_years = self.config["max_train_years"]

        test_start = original_start + relativedelta(years=initial_train_years)
        test_window_end = final_end

        months_available = (test_window_end.year - test_start.year) * 12 + (test_window_end.month - test_start.month)

        if months_available >= max_train_years * 12:
            training_start = test_window_end - relativedelta(years=max_train_years)
        else:
            training_start = original_start

        self.config["current_start"] = training_start.strftime("%Y-%m-%d")
        self.config["current_end"] = test_window_end.strftime("%Y-%m-%d")

        logger.info(f"Training window: {training_start.date()} to {test_window_end.date()}")

        for ticker in self.config["tickers"]:
            logger.debug(f"[{ticker}] Starting training...")
            trainer = ModelsTrainingProcessor(config=self.config, data=self.data, ticker=ticker)
            training = trainer.process()
            training_path = os.path.join("hmm", "train", "artifacts", "training", f"{ticker}.pkl")
            with open(training_path, 'wb') as f:
                pickle.dump(training, f)
            logger.debug(f"[{ticker}] Training saved to {training_path}")

            logger.debug(f"[{ticker}] Starting inferencing...")
            inferencer = ModelsInferenceProcessor(config=self.config, ticker=ticker)
            inferencing = inferencer.process()
            inferencing_path = os.path.join("hmm", "infer", "artifacts", "inferencing", f"{ticker}.pkl")
            with open(inferencing_path, 'wb') as f:
                pickle.dump(inferencing, f)
            logger.debug(f"[{ticker}] Inferencing saved to {inferencing_path}")

        logger.info(f"Building portfolio for test date: {test_window_end.date()}...")
        builder = PortfolioProcessor(config=self.config, data=self.data)
        portfolio = builder.process()
        portfolio = {asset: weight for asset, weight in portfolio.items() if weight != 0.0}
        logger.info(f"Built portfolio: {portfolio}")
