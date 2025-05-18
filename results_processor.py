import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import os
import scipy.stats as stats

import utilities_general as utilities


class ResultsProcessor:

    def __init__(self):
        pass

    def process(self):
        pass

    def _save_plot(self, filename, plot_type):
        directory = os.path.join("artifacts", plot_type)
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        plt.close()

    def _plot_regimes(self):
        plt.figure(figsize=(15, 5))
        train_idx = self.train_data.index
        for state in np.unique(self.train_states):
            idx = self.train_states == state
            plt.plot(train_idx[idx], self.train_data['Momentum'].iloc[idx], '.', label=f'Train State {state}')

        test_idx = self.test_data.index
        for state in np.unique(self.test_states):
            idx = self.test_states == state
            plt.plot(test_idx[idx], self.test_data['Momentum'].iloc[idx], '.', label=f'Test State {state}')

        plt.legend()
        plt.title(f"Market Regimes for {self.ticker} (Train/Test)")
        self._save_plot(f"{self.ticker}_regime_plot.png", plot_type="regime_plots")

    def _plot_price_with_states(self):
        plt.figure(figsize=(15, 5))
        index_combined = np.concatenate([self.train_data.index, self.test_data.index])
        states_combined = np.concatenate([self.train_states, self.test_states])

        for state in np.unique(states_combined):
            idx = states_combined == state
            plt.plot(index_combined[idx], np.arange(len(index_combined[idx])), '.', label=f'State {state}')

        plt.title(f"Regime States for {self.ticker}")
        plt.ylabel("State Index")
        plt.xlabel("Date")
        plt.legend()
        self._save_plot(f"{self.ticker}_states_plot.png", plot_type="state_index_plots")

    def _plot_price_path_with_states(self):
        adj_close = yf.download(
            tickers=self.ticker,
            start=self.start_date,
            end=self.end_date,
            group_by='ticker',
            auto_adjust=False,
            progress=False,
            threads=True,
            interval="1d"
        )

        if isinstance(adj_close.columns, pd.MultiIndex):
            price_series = adj_close[self.ticker]["Adj Close"].dropna()
        else:
            price_series = adj_close["Adj Close"].dropna()

        plt.figure(figsize=(15, 5))
        index_combined = np.concatenate([self.train_data.index, self.test_data.index])
        states_combined = np.concatenate([self.train_states, self.test_states])

        for state in np.unique(states_combined):
            idx = states_combined == state
            dates = index_combined[idx]
            prices = price_series.loc[dates]
            plt.plot(prices.index, prices.values, '.', label=f'State {state}')

        plt.title(f"Price Path with Regimes for {self.ticker}")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend()
        self._save_plot(f"{self.ticker}_price_path.png", plot_type="price_path_plots")
