"""
Module for the TestRunner class.
"""

import os
import logging
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

import logger_config
from hmm.runner.base_runner import BaseRunner
from hmm import utilities
from hmm.build.portfolio_processor import PortfolioProcessor
from hmm.results.final_portfolio_results import FinalResultsPortfolio
from hmm.infer.models_inferencing_processor import ModelsInferenceProcessor
from hmm.train.models_training_processor import ModelsTrainingProcessor

logger = logging.getLogger(__name__)

# Constants for artifact paths
TRAINING_ARTIFACTS_PATH = os.path.join("hmm", "train", "artifacts", "training")
INFERENCING_ARTIFACTS_PATH = os.path.join("hmm", "infer", "artifacts", "inferencing")
TRADE_DATA_PATH = os.path.join(os.getcwd(), "artifacts", "trade_data")


def save_pickle(obj, path, filename):
    """
    Save an object to a pickle file, ensuring the directory exists.
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(obj, f)


def process_ticker(config: dict, data: pd.DataFrame, ticker: str) -> bool:
    """
    Helper method to process model training and inferencing.

    Parameters
    ----------
    config : dict
        Dictionary containing configured properties.
    data : pd.DataFrame
        Dataframe containing data for the run being processed.
    ticker : str
        String representing the ticker being processed.

    Returns
    -------
    bool
        Whether the ticker was successfully processed through train and infer.
    """
    logger.debug(f"[{ticker}] Starting processing...")

    trainer = ModelsTrainingProcessor(config=config, data=data, ticker=ticker)
    training = trainer.process()
    save_pickle(training, TRAINING_ARTIFACTS_PATH, f"{ticker}.pkl")

    inferencer = ModelsInferenceProcessor(config=config, ticker=ticker)
    inferencing = inferencer.process()
    save_pickle(inferencing, INFERENCING_ARTIFACTS_PATH, f"{ticker}.pkl")

    logger.debug(f"[{ticker}] Finished.")

    return True


class TestRunner(BaseRunner):
    """
    Class for running test tasks.
    """

    def run(self):
        """
        Method for processing the run pipeline.
        """
        original_start = datetime.strptime(self.config["start_date"], "%Y-%m-%d")
        final_end = datetime.strptime(self.config["end_date"], "%Y-%m-%d")

        model_warmup = self.config["model_warmup"]
        max_train_years = self.config["max_train_years"]

        full_data_start = original_start - relativedelta(years=model_warmup)
        test_start = original_start

        all_trade_details = []
        results = []

        while test_start <= final_end:
            rolling_train_start = test_start - relativedelta(years=max_train_years)
            training_start = max(full_data_start, rolling_train_start)
            test_window_end = test_start + MonthEnd(0)

            self.config["current_start"] = training_start.strftime("%Y-%m-%d")
            self.config["current_end"] = test_window_end.strftime("%Y-%m-%d")

            logger.info(
                f"\n=== TEST WINDOW: {test_start.date()} to {test_window_end.date()} ===\n"
                f"Training window: {training_start.date()} to {test_window_end.date()}\n"
                f"Data slice: {self.config['current_start']} to {self.config['current_end']}"
            )

            tickers = self.config["tickers"]
            num_workers = min(len(tickers), 8)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_ticker, self.config, self.data, ticker) for ticker in tickers]
                for _ in tqdm(futures, desc="Processing tickers", leave=False):
                    _.result()

            logger.info(f"Building portfolio at month end: {test_window_end.date()}...")
            builder = PortfolioProcessor(config=self.config, data=self.data)
            portfolio = builder.process()
            portfolio = {asset: weight for asset, weight in portfolio.items() if weight != 0.0}
            logger.info(f"Portfolio to trade on {test_window_end.date()}: {portfolio}")

            trade_window_start = test_window_end
            trade_window_end = trade_window_start + relativedelta(months=1)

            logger.info(f"Trade starting: {trade_window_start.date()} and ending: {trade_window_end.date()}")

            portfolio_return, trade_stats, trade_details = utilities.calculate_portfolio_return(
                portfolio=portfolio,
                data=self.data,
                start_date=trade_window_start,
                end_date=trade_window_end,
                threshold=self.config["stop_loss"],
                use_stop_loss=self.config["use_stop_loss"]
            )

            logger.info(f"Return: {portfolio_return * 100:.2f}%")
            logger.info(f"Trade Counts: {trade_stats}")

            results.append({
                "train_start": training_start.date(),
                "train_end": test_start.date(),
                "test_start": test_start.date(),
                "test_end": test_window_end.date(),
                "return_start": trade_window_start.date(),
                "return_end": trade_window_end.date(),
                "portfolio_return": portfolio_return,
                "positive_trades": trade_stats["positive"],
                "negative_trades": trade_stats["negative"],
                "average_positive_trade": trade_stats["average_gain"],
                "average_negative_trade": trade_stats["average_loss"]
            })

            trade_details = trade_details.copy()
            trade_details["trade_window_end"] = trade_window_end
            all_trade_details.append(trade_details)

            test_start += relativedelta(months=1)

        if self.config["persist"]:
            utilities.plot_cumulative_returns(all_trade_details=all_trade_details)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        os.makedirs(TRADE_DATA_PATH, exist_ok=True)

        full_path_trades = os.path.join(TRADE_DATA_PATH, f"portfolio_trades_{timestamp}.csv")
        full_path_returns = os.path.join(TRADE_DATA_PATH, f"portfolio_returns_{timestamp}.csv")

        pd.concat(all_trade_details, ignore_index=True).to_csv(full_path_trades, index=False)
        pd.DataFrame(results).to_csv(full_path_returns, index=False)

        processor = FinalResultsPortfolio(results=results, config=self.config)
        processor.process()
        logger.info("Saved test results to portfolio_test_results.csv")

        return pd.DataFrame(results)
