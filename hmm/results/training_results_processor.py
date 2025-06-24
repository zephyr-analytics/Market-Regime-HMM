"""
"""

import matplotlib.pyplot as plt
import numpy as np

import hmm.utilities as utilities
from hmm.results.results_processor import ResultsProcessor


class TrainingResultsProcessor(ResultsProcessor):
    """
    """
    def __init__(self, training):
        super().__init__(
            ticker=training.ticker,
            model=training.model,
            train_states=training.train_states,
            train_data=training.train_data,
            start_date=training.start_date,
            end_date=training.end_date,
            label_dict=training.state_labels,
            plot_sub_folder = "train"
        )

    def process(self):
        """
        """
        # self._plot_regimes()
        self._plot_price_with_states()
        self._plot_price_path_train_only()

    def _plot_price_path_train_only(self):
        """
        """
        price_series = self._download_price_series()
        plt.figure(figsize=(15, 5))

        for state in np.unique(self.train_states):
            idx = self.train_states == state
            dates = self.train_data.index[idx]
            prices = price_series.loc[dates.intersection(price_series.index)]
            plt.plot(prices.index, prices.values, '.', label=self._get_label(state))

        plt.title(f"Training Price Path with Regimes for {self.ticker}")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend()
        utilities.save_plot(f"{self.ticker}_price_path.png", plot_type="price_path_plots", plot_sub_folder=self.plot_sub_folder)
