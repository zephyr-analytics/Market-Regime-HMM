import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

import hmm.utilities as utilities


class ResultsProcessor:

    def __init__(self, training, ticker, start_date, end_date, test_data=None, test_states=None):
        self.ticker = ticker
        self.model = training.model
        self.train_states = training.train_states
        self.train_data = training.train_data
        self.start_date = start_date
        self.end_date = end_date
        self.label_dict = training.state_labels
        self.test_data = training.test_data
        self.test_states = test_states

    def process(self):
        self._plot_regimes()
        self._plot_price_with_states(train_states=self.train_states)
        self._plot_price_path_with_states(
            train_states=self.train_states,
            test_data=self.test_data,
            test_states=self.test_states if self.test_data is not None and self.test_states is not None else None
        )

    def _get_label(self, state):
        return self.label_dict.get(state, f'State {state}')

    def _plot_regimes(self):
        plt.figure(figsize=(15, 5))
        train_idx = self.train_data.index
        for state in np.unique(self.train_states):
            idx = self.train_states == state
            label = self._get_label(state)
            plt.plot(train_idx[idx], self.train_data['Momentum'].iloc[idx], '.', label=label)

        plt.legend()
        plt.title(f"Market Regimes for {self.ticker} Training")
        utilities.save_plot(f"{self.ticker}_regime_plot.png", plot_type="regime_plots")

    def _plot_price_with_states(self, train_states):
        plt.figure(figsize=(15, 5))
        train_index = self.train_data.index

        for state in np.unique(train_states):
            idx = train_states == state
            label = self._get_label(state)
            plt.plot(train_index[idx], np.arange(len(train_index[idx])), '.', label=label)

        plt.title(f"Regime States for {self.ticker}")
        plt.ylabel("State Index")
        plt.xlabel("Date")
        plt.legend()
        utilities.save_plot(f"{self.ticker}_states_plot.png", plot_type="state_index_plots")

    def _plot_price_path_with_states(self, train_states, test_data=None, test_states=None):
        adj_close = yf.download(
            tickers=self.ticker,
            start=self.start_date,
            end=self.end_date,
            group_by='ticker',
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if isinstance(adj_close.columns, pd.MultiIndex):
            price_series = adj_close[self.ticker]["Adj Close"].dropna()
        else:
            price_series = adj_close["Adj Close"].dropna()

        plt.figure(figsize=(15, 5))

        # Plot training data
        train_index = self.train_data.index
        for state in np.unique(train_states):
            idx = train_states == state
            dates = train_index[idx]
            prices = price_series.loc[dates.intersection(price_series.index)]
            label = self._get_label(state) + " (Train)"
            plt.plot(prices.index, prices.values, '.', label=label)

        # Plot test data if available
        if test_data is not None and test_states is not None:
            test_index = test_data.index
            for state in np.unique(test_states):
                idx = test_states == state
                dates = test_index[idx]
                prices = price_series.loc[dates.intersection(price_series.index)]
                label = self._get_label(state) + " (Test)"
                plt.plot(prices.index, prices.values, 'o', label=label)  # Different marker

        plt.title(f"Price Path with Regimes for {self.ticker}")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend()
        utilities.save_plot(f"{self.ticker}_price_path.png", plot_type="price_path_plots")
