import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import os
import scipy.stats as stats

import utilities_general as utilities
from models_training_processor import ModelsTrainingProcessor


class ModelsInferenceProcessor:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.allocations = {}
        self.cash_weight = 0.0

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
