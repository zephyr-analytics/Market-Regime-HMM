"""
"""

import pandas as pd
import numpy as np
import json
import os


class MovingAverageTuner:
    """
    """
    def __init__(self, price_data: pd.DataFrame, start: str, end: str, config: dict):
        self.price_data = price_data
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.config = config
        self.tickers = config.get("tickers", [])
        self.ma_lengths = [147, 168, 189, 210, 231, 252]
        self.best_config = {}

    def process(self):
        """
        Method to process through MovingAverageTuner.
        """
        print("Starting moving average tuning process...")
        self.run()
        self.save_config()

    @staticmethod
    def compute_sma(prices: pd.Series, length: int) -> pd.Series:
        """
        Method to compute the moving average.

        Parameters
        ----------
        prices : pd.Series
            Series of asset prices.
        length : int
            Integer representing the moving average length of the time.
        Returns
        -------
        pd.Series
            Series representing the moving average.
        """

        return prices.rolling(window=length).mean()

    @staticmethod
    def score_moving_average(prices: pd.Series, moving_average: pd.Series) -> float:
        """
        Method to sharpe score the moving average.

        Parameters
        ----------
        prices : pd.Series

        moving_average : pd.Series

        Returns
        -------
        sharpe : float
            Float representing the sharpe ratio of the strategy.
        """
        eom_prices = prices.resample("M").last()
        eom_ma = moving_average.reindex(eom_prices.index)
        signal = eom_prices > eom_ma
        monthly_returns = eom_prices.pct_change().fillna(0)
        strategy_returns = monthly_returns * signal.shift(1).fillna(0)
        sharpe = strategy_returns.mean() / strategy_returns.std()

        return sharpe

    def tune_ticker(self, ticker: str) -> int:
        """
        Method to tune moving average length.

        Parameters
        ----------
        ticker : str
            String representing the ticker to tune.
        """
        prices = self.price_data.get(ticker)

        prices = prices.loc[(prices.index >= self.start) & (prices.index <= self.end)].dropna()

        best_score = -np.inf
        best_length = None

        for length in self.ma_lengths:
            ma = self.compute_sma(prices, length)
            score = self.score_moving_average(prices=prices, moving_average=ma)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_length = length

        return best_length

    def run(self) -> dict:
        """
        """
        for ticker in self.tickers:
            print(f"Tuning {ticker}...")
            result = self.tune_ticker(ticker)
            if result:
                self.best_config[ticker] = {"length": result}

        return self.best_config

    def save_config(self):
        """
        """
        filepath = os.path.join(os.getcwd(), "configs", "ma_config.json")
        with open(filepath, "w") as f:
            json.dump(self.best_config, f, indent=4)
        print(f"Configuration saved to {filepath}")
