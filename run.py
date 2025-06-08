"""
"""

import argparse
import logging
import os
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import pandas as pd

import logger_config
import hmm.utilities as utilities
from hmm.build.portfolio_processor import PortfolioProcessor
from hmm.data.data_processor import DataProcessor
from hmm.train.models_training_processor import ModelsTrainingProcessor
from hmm.infer.models_inferencing_processor import ModelsInferenceProcessor
from hmm.results.final_portfolio_results import FinalResultsPortfolio

logger = logging.getLogger(__name__)


def process_ticker(config: dict, data: pd.DataFrame, ticker: str) -> bool:
    """
    Method to create a Train and Infer pipeline, used by ThreadPoolExecutor.

    Parameters
    ----------
    config : dict
        Dictionary of the config file.
    data : pd.DataFrame
        Dataframe of asset_data.
    ticker : str
        String representing the ticker symbol.

    Returns
    -------
    bool : Bool representing that the ticker has completed the pipeline.
    """
    logger.debug(f"[{ticker}] Starting processing...")

    trainer = ModelsTrainingProcessor(config=config, data=data, ticker=ticker)
    training = trainer.process()
    train_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "training", f"{ticker}.pkl")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, 'wb') as f:
        pickle.dump(training, f)

    inferencer = ModelsInferenceProcessor(config=config, ticker=ticker)
    inferencing = inferencer.process()
    infer_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing", f"{ticker}.pkl")
    os.makedirs(os.path.dirname(infer_path), exist_ok=True)
    with open(infer_path, 'wb') as f:
        pickle.dump(inferencing, f)

    logger.debug(f"[{ticker}] Finished.")
    return True


def run_portfolio_test(config) -> dict:
    """
    Method to handle running a backtest over the time horizon.

    Parameters
    ----------
    config : dict
        Dictionary of the config file.

    Returns
    -------
    results : dict
        Dictionary containing all outcomes from the backtest.
    """
    data_process = DataProcessor(config=config)
    data = data_process.process()
    print(data)
    original_start = datetime.strptime(config["start_date"], "%Y-%m-%d")
    final_end = datetime.strptime(config["end_date"], "%Y-%m-%d")

    test_start = original_start + relativedelta(years=config["model_warmup"])
    results = []

    while test_start + relativedelta(months=1) <= final_end:
        test_window_end = test_start + relativedelta(months=1)
        logger.info(f"\n=== TEST WINDOW: {test_start.date()} to {test_window_end.date()} ===")

        config["current_start"] = original_start.strftime("%Y-%m-%d")
        config["current_end"] = test_start.strftime("%Y-%m-%d")
        tickers = config["tickers"]

        logger.info("Training and inferring models for tickers...")
        with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as executor:
            futures = [executor.submit(process_ticker, config, data, ticker) for ticker in tickers]
            for _ in tqdm(futures, desc="Processing tickers", leave=False):
                _.result()

        logger.info(f"Building portfolio for {test_start.date()} to {test_window_end.date()}...")
        build = PortfolioProcessor(config=config, data=data)
        portfolio = build.process()

        return_window_start = test_start
        return_window_end = test_start + relativedelta(months=1)
        logger.info(f"Evaluating return from {return_window_start.date()} to {return_window_end.date()}...")

        portfolio_return = utilities.calculate_portfolio_return(
            portfolio=portfolio,
            data=data,
            start_date=return_window_start,
            end_date=return_window_end
        )

        logger.info(f"Portfolio: {portfolio}")
        logger.info(f"Portfolio return: {portfolio_return:.2%}")

        results.append({
            "train_start": original_start.date(),
            "train_end": test_start.date(),
            "test_start": test_start.date(),
            "test_end": test_window_end.date(),
            "return_start": return_window_start.date(),
            "return_end": return_window_end.date(),
            "portfolio_return": portfolio_return
        })

        test_start += relativedelta(months=1)

    return results


def main():
    """
    Method to handle run pipelines based on terminal commands.
    """
    parser = argparse.ArgumentParser(description="Run Market Regime HMM Model using JSON Config")

    # Model operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train the model")
    mode_group.add_argument("--infer", action="store_true", help="Run inference using the model")
    mode_group.add_argument("--build", action="store_true", help="Build portfolios")
    mode_group.add_argument("--test", action="store_true", help="Test rolling portfolio performance")

    # Asset type selection (stock vs ETF)
    asset_group = parser.add_mutually_exclusive_group(required=True)
    asset_group.add_argument("--etf", action="store_true", help="Use ETF config")
    asset_group.add_argument("--stock", action="store_true", help="Use stock config")

    args = parser.parse_args()

    config = utilities.load_config(etf=args.etf, stocks=args.stock)
    tickers = config["tickers"]
    data_process = DataProcessor(config=config)
    data = data_process.process()

    if args.train or args.infer:
        for ticker in tickers:
            if args.train:
                logger.info(f"Training model for {ticker}...")
                config["current_end"] = config["end_date"]

                model = ModelsTrainingProcessor(config=config, data=data, ticker=ticker)
                training = model.process()
                file_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "training", f"{ticker}.pkl")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    pickle.dump(training, f)

            elif args.infer:
                logger.info(f"Running inference for {ticker}...")
                config["current_end"] = config["end_date"]
                model = ModelsInferenceProcessor(config=config, ticker=ticker)
                inferencing = model.process()
                file_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing", f"{ticker}.pkl")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    pickle.dump(inferencing, f)

    elif args.build:
        logger.info("Building single portfolio...")
        config["current_end"] = config["end_date"]
        build = PortfolioProcessor(config=config, data=data)
        portfolio = build.process()
        logger.info(f"Built portfolio: {portfolio}")

    elif args.test:
        results = run_portfolio_test(config)
        df = pd.DataFrame(results)
        processor = FinalResultsPortfolio(results=results)
        processor.process()
        df.to_csv("portfolio_test_results.csv", index=False)
        logger.info("\nSaved test results to portfolio_test_results.csv")


if __name__ == "__main__":
    main()
