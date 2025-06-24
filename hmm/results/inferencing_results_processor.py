"""
"""

import matplotlib.pyplot as plt
import numpy as np

import hmm.utilities as utilities
from hmm.results.results_processor import ResultsProcessor


class InferencingResultsProcessor(ResultsProcessor):
    """
    """
    def __init__(self, inferencing):
        super().__init__(
            ticker=inferencing.ticker,
            model=inferencing.model,
            train_states=inferencing.train_states,
            train_data=inferencing.train_data,
            start_date=inferencing.start_date,
            end_date=inferencing.end_date,
            label_dict=inferencing.state_labels,
            plot_sub_folder = "infer"
        )
        self.test_data = inferencing.test_data
        self.test_states = inferencing.test_states


    def process(self):
        """
        """
        # self._plot_regimes()
        self._plot_price_with_states()
        self._plot_price_path_with_train_and_test()

    def _plot_price_path_with_train_and_test(self):
        """
        """
        price_series = self._download_price_series()
        plt.figure(figsize=(15, 5))
        # Plot training data
        for state in np.unique(self.train_states):
            idx = self.train_states == state
            dates = self.train_data.index[idx]
            prices = price_series.loc[dates.intersection(price_series.index)]
            plt.plot(prices.index, prices.values, '.', label=self._get_label(state) + " (Train)")

        # Plot test data
        for state in np.unique(self.test_states):
            idx = self.test_states == state
            dates = self.test_data.index[idx]
            prices = price_series.loc[dates.intersection(price_series.index)]
            plt.plot(prices.index, prices.values, '.', label=self._get_label(state) + " (Test)")

        plt.title(f"Price Path with Train/Test Regimes for {self.ticker}")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend()
        utilities.save_plot(f"{self.ticker}_price_path.png", plot_type="price_path_plots", plot_sub_folder=self.plot_sub_folder)
