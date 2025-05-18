"""
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import os
import scipy.stats as stats

import utilities_general as utilities


class ResultsProcessor:

    def __init__(self, ticker, model, training_data, start_date, end_date):
        self.ticker = ticker
        self.model = model
        self.train_data = training_data
        self.start_date = start_date
        self.end_date = end_date
        print(model.__dict__)

    def process(self):
        self._plot_regimes()
        self._plot_price_with_states()
        self._plot_price_path_with_states()

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

    def build_portfolio(self):
        for ticker in self.tickers:
            model = ModelsTrainingProcessor(ticker=ticker, start_date=self.start_date, end_date=self.end_date)
            model.process()
            forecast_probs = model.forecast_state_distribution(n_steps=21)
            total_bullish_prob = forecast_probs[model.bullish_state]

            starting_weight = 0.09
            if total_bullish_prob >= 0.95:
                final_weight = starting_weight
            elif 0.7 <= total_bullish_prob < 0.95:
                final_weight = starting_weight / 2
            else:
                final_weight = 0.0

            self.allocations[ticker] = final_weight

        self._finalize_allocations()
        self._plot_portfolio()

    def _finalize_allocations(self):
        total_allocated = sum(self.allocations.values())
        self.cash_weight = max(0.0, 1.0 - total_allocated)
        self.allocations['CASH'] = self.cash_weight

        print("\nFinal Portfolio Allocation:")
        for ticker, weight in self.allocations.items():
            print(f"  {ticker}: {weight:.2%}")

    def _plot_portfolio(self):
        labels = list(self.allocations.keys())
        sizes = list(self.allocations.values())

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title("Final Portfolio Allocation")
        os.makedirs("artifacts", exist_ok=True)
        plt.savefig("artifacts/final_portfolio_pie_chart.png")
        plt.close()
        print("\nSaved portfolio pie chart to artifacts/final_portfolio_pie_chart.png")
