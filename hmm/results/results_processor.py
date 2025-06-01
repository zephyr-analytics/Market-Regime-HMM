"""
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

import hmm.utilities as utilities


class ResultsProcessor:
    """
    """
    # TODO this class needs to be parsed to ensure that train and infer are getting their own handling.
    def __init__(self, ticker, model, train_states, train_data, start_date, end_date, label_dict, plot_sub_folder):
        self.ticker = ticker
        self.model = model
        self.train_states = train_states
        self.train_data = train_data
        self.start_date = start_date
        self.end_date = end_date
        self.label_dict = label_dict
        self.plot_sub_folder = plot_sub_folder

    def _get_label(self, state):
        """
        """
        return self.label_dict.get(state, f'State {state}')

    def _plot_regimes(self):
        """
        """
        plt.figure(figsize=(15, 5))
        for state in np.unique(self.train_states):
            idx = self.train_states == state
            label = self._get_label(state)
            plt.plot(self.train_data.index[idx], self.train_data['Momentum'].iloc[idx], '.', label=label)
        plt.legend()
        plt.title(f"Market Regimes for {self.ticker} Training")
        utilities.save_plot(f"{self.ticker}_regime_plot.png", plot_type="regime_plots", plot_sub_folder=self.plot_sub_folder)

    def _plot_price_with_states(self):
        """
        """
        plt.figure(figsize=(15, 5))
        for state in np.unique(self.train_states):
            idx = self.train_states == state
            plt.plot(self.train_data.index[idx], np.arange(len(self.train_data.index[idx])), '.', label=self._get_label(state))
        plt.title(f"Regime States for {self.ticker}")
        plt.ylabel("State Index")
        plt.xlabel("Date")
        plt.legend()
        utilities.save_plot(f"{self.ticker}_states_plot.png", plot_type="state_index_plots", plot_sub_folder=self.plot_sub_folder)

    def _download_price_series(self):
        """
        """
        adj_close = yf.download(
            tickers=self.ticker,
            start=self.start_date,
            end=self.end_date,
            group_by='ticker',
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        return adj_close[self.ticker]["Adj Close"].dropna() if isinstance(adj_close.columns, pd.MultiIndex) else adj_close["Adj Close"].dropna()
