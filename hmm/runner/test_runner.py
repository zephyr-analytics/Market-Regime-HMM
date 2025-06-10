"""
"""

import os
import logging
import pandas as pd
import pickle
from datetime import datetime

from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import logger_config
from hmm.runner.base_runner import BaseRunner
from hmm import utilities
from hmm.build.portfolio_processor import PortfolioProcessor
from hmm.results.final_portfolio_results import FinalResultsPortfolio
from hmm.infer.models_inferencing_processor import ModelsInferenceProcessor
from hmm.train.models_training_processor import ModelsTrainingProcessor

logger = logging.getLogger(__name__)

def process_ticker(config, data, ticker):
    """
    """
    logger.debug(f"[{ticker}] Starting processing...")
    trainer = ModelsTrainingProcessor(config=config, data=data, ticker=ticker)
    training = trainer.process()
    with open(os.path.join("hmm", "train", "artifacts", "training", f"{ticker}.pkl"), 'wb') as f:
        pickle.dump(training, f)

    inferencer = ModelsInferenceProcessor(config=config, ticker=ticker)
    inferencing = inferencer.process()
    with open(os.path.join("hmm", "infer", "artifacts", "inferencing", f"{ticker}.pkl"), 'wb') as f:
        pickle.dump(inferencing, f)

    logger.debug(f"[{ticker}] Finished.")
    return True


class TestRunner(BaseRunner):
    def run(self):
        original_start = datetime.strptime(self.config["start_date"], "%Y-%m-%d")
        final_end = datetime.strptime(self.config["end_date"], "%Y-%m-%d")

        initial_train_years = 2
        max_train_years = 8
        years_before_drop = 8

        # First test point: after 2 years of data
        test_start = original_start + relativedelta(years=initial_train_years)
        first_test_start = test_start

        results = []
        while test_start + relativedelta(months=1) <= final_end:
            test_window_end = test_start + relativedelta(months=1)

            # Calculate how long we've been testing
            months_tested = (test_start.year - first_test_start.year) * 12 + (test_start.month - first_test_start.month)

            # Determine training window length
            if months_tested < years_before_drop * 12:
                training_start = original_start
            else:
                # Start dropping from the front to maintain max_train_years
                training_start = test_start - relativedelta(years=max_train_years)

            self.config["current_start"] = training_start.strftime("%Y-%m-%d")
            self.config["current_end"] = test_start.strftime("%Y-%m-%d")

            logger.info(f"\n=== TEST WINDOW: {test_start.date()} to {test_window_end.date()} ===")
            logger.info(f"Training window: {training_start.date()} to {test_start.date()}")

            tickers = self.config["tickers"]
            with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as executor:
                futures = [executor.submit(process_ticker, self.config, self.data, ticker) for ticker in tickers]
                for _ in tqdm(futures, desc="Processing tickers", leave=False):
                    _.result()

            logger.info(f"Building portfolio for {test_start.date()} to {test_window_end.date()}...")
            builder = PortfolioProcessor(config=self.config, data=self.data)
            portfolio = builder.process()
            logger.info(f"Portfolio: {portfolio}")
            portfolio_return = utilities.calculate_portfolio_return(
                portfolio=portfolio,
                data=self.data,
                start_date=test_start,
                end_date=test_window_end
            )
            logger.info(f"Return: {portfolio_return * 100:.2f}%")

            results.append({
                "train_start": training_start.date(),
                "train_end": test_start.date(),
                "test_start": test_start.date(),
                "test_end": test_window_end.date(),
                "return_start": test_start.date(),
                "return_end": test_window_end.date(),
                "portfolio_return": portfolio_return
            })

            test_start += relativedelta(months=1)

        pd.DataFrame(results).to_csv("portfolio_test_results.csv", index=False)
        processor = FinalResultsPortfolio(results=results)
        processor.process()
        logger.info("Saved test results to portfolio_test_results.csv")
