"""
"""

import argparse
import os
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd

import hmm.utilities as utilities
from hmm.build.build_processor import BuildProcessor
from hmm.data.data_processor import DataProcessor
from hmm.train.models_training_processor import ModelsTrainingProcessor
from hmm.infer.models_inferencing_processor import ModelsInferenceProcessor


def calculate_portfolio_return(portfolio, start_date, end_date):
    """
    Calculate the portfolio return between start_date and end_date using
    price data provided as a dict of Series.

    Args:
        portfolio (dict): Dictionary of {ticker: weight}.
        start_date (datetime): Start of the return period.
        end_date (datetime): End of the return period.

    Returns:
        float: Weighted portfolio return as a decimal (e.g., 0.02 = 2%).
    """
    if not portfolio:
        return 0.0

    tickers = list(portfolio.keys())
    weights = pd.Series(portfolio)

    # Get price data dict {ticker: pd.Series}
    raw_price_data = utilities.load_price_data(
        tickers,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    # Convert to aligned DataFrame
    price_df = pd.DataFrame(raw_price_data)

    # Drop any rows with missing data (can be more sophisticated if needed)
    price_df.dropna(inplace=True)

    if price_df.empty or len(price_df) < 2:
        return 0.0

    start_prices = price_df.iloc[0]
    end_prices = price_df.iloc[-1]

    returns = (end_prices - start_prices) / start_prices
    portfolio_return = (returns * weights).sum()

    return portfolio_return


def run_portfolio_test(config):
    original_start = datetime.strptime(config["start_date"], "%Y-%m-%d")
    final_end = datetime.strptime(config["end_date"], "%Y-%m-%d")

    # Start testing 1 year after the original start date
    test_start = original_start + relativedelta(years=1)

    results = []

    while test_start + relativedelta(months=1) <= final_end:
        test_window_end = test_start + relativedelta(months=1)

        print(f"\n=== TEST WINDOW: {test_start.date()} to {test_window_end.date()} ===")

        # Training window remains static
        config["current_start"] = original_start.strftime("%Y-%m-%d")
        config["current_end"] = test_start.strftime("%Y-%m-%d")

        tickers = config["tickers"]

        # Train and infer for each ticker
        for ticker in tickers:
            print(f"Training model for {ticker}...")
            data_process = DataProcessor(config=config)
            data = data_process.process()
            trainer = ModelsTrainingProcessor(config=config, data=data, ticker=ticker)
            training = trainer.process()
            file_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "training", f"{ticker}.pkl")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(training, f)

            print(f"Running inference for {ticker}...")
            inferencer = ModelsInferenceProcessor(config=config, ticker=ticker)
            inferencing = inferencer.process()
            file_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing", f"{ticker}.pkl")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(inferencing, f)

        # Build portfolio using inferred data
        print(f"Building portfolio for {test_start.date()} to {test_window_end.date()}...")
        build = BuildProcessor(config=config)
        portfolio = build.process()

        # Calculate return over next window
        return_window_start = test_window_end
        return_window_end = return_window_start + relativedelta(months=1)
        print(f"Evaluating return from {return_window_start.date()} to {return_window_end.date()}...")

        portfolio_return = calculate_portfolio_return(portfolio, return_window_start, return_window_end)
        print(f"Portfolio return: {portfolio_return:.2%}")

        results.append({
            "train_start": original_start.date(),
            "train_end": test_start.date(),
            "test_start": test_start.date(),
            "test_end": test_window_end.date(),
            "return_start": return_window_start.date(),
            "return_end": return_window_end.date(),
            "portfolio_return": portfolio_return
        })

        # Move the test window forward by 1 month
        test_start += relativedelta(months=1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Market Regime HMM Model using JSON Config")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument("--infer", action="store_true", help="Run inference using the model")
    group.add_argument("--build", action="store_true", help="Build portfolios")
    group.add_argument("--test", action="store_true", help="Test rolling portfolio performance")

    args = parser.parse_args()
    config = utilities.load_config()
    tickers = config["tickers"]

    if args.train or args.infer:
        for ticker in tickers:
            if args.train:
                print(f"Training model for {ticker}...")
                config["current_end"] = config["end_date"]
                data_process = DataProcessor(config=config)
                data = data_process.process()
                model = ModelsTrainingProcessor(config=config, data=data, ticker=ticker)
                training = model.process()
                file_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "training", f"{ticker}.pkl")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    pickle.dump(training, f)

            elif args.infer:
                print(f"Running inference for {ticker}...")
                config["current_end"] = config["end_date"]
                model = ModelsInferenceProcessor(config=config, ticker=ticker)
                inferencing = model.process()
                file_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing", f"{ticker}.pkl")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    pickle.dump(inferencing, f)

    elif args.build:
        print("Building single portfolio...")
        config["current_end"] = config["end_date"]
        build = BuildProcessor(config=config)
        portfolio = build.process()
        print(f"Built portfolio: {portfolio}")

    elif args.test:
        results = run_portfolio_test(config)
        df = pd.DataFrame(results)
        df.to_csv("portfolio_test_results.csv", index=False)
        print("\nSaved test results to portfolio_test_results.csv")


if __name__ == "__main__":
    main()
