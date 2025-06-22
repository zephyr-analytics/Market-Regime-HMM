"""
Module for building portfolio.
"""

import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from hmm.build.portfolio_constructor import PortfolioConstructor
from hmm.results.portfolio_results_processor import PortfolioResultsProcessor
from hmm.build.portfolio_clustering import PortfolioClustering
from models.hierarchical_clustering import cluster_sequences

logger = logging.getLogger(__name__)


class PortfolioProcessor:
    """
    Class to take processed model data and build a portfolio.
    """
    def __init__(self, config: dict, data: pd.DataFrame):
        self.config = config
        self.start_date = config["current_start"]
        self.end_date = config["current_end"]
        self.persist = config["persist"]
        # self.min_clusters = config["min_clusters"]
        # self.max_clusters = config["max_clusters"]
        self.data = data.loc[self.start_date:self.end_date]

    def process(self):
        """
        Method for processing through the pipeline.
        """
        file_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing")

        parsed_objects = self.load_models_inference(directory=file_path, tickers=self.config["tickers"])

        state_data = self.extract_states(parsed_objects=parsed_objects)

        self.moving_average_check(state_data=state_data, moving_average_window=self.config["moving_average"])

        sequences, tickers = self.prepare_state_sequences(state_data, lookback=self.config["sequence_lookback"])

        forecast_data = self.extract_forecast_distributions(parsed_objects=parsed_objects)

        results = cluster_sequences(
            sequences=sequences, tickers=tickers, max_clusters=self.max_clusters, min_clusters=self.min_clusters
        )

        clusters = results["clusters"]

        constructor = PortfolioConstructor(
            config=self.config, clusters=clusters, forecast_data=forecast_data, price_data=self.data
        )
        portfolio = constructor.process()

        if self.persist:
            results_process = PortfolioResultsProcessor(
                config=self.config,
                n_clusters=results["n_clusters"],
                portfolio=portfolio
            )
            results_process.process()
            return portfolio

        else:

            return portfolio


    @staticmethod
    def initialize_portfolio_clustering(config: dict, data: pd.DataFrame):
        """
        Method to initialize PortfolioClustering.
        """
        clustering = PortfolioClustering()
        clustering.min_clusters = config["min_clusters"]
        clustering.max_clusters = config["max_clusters"]
        clustering.start_date = config["current_start"]
        clustering.end_date = config["current_end"]
        clustering.price_data = data[clustering.start_date:clustering.end_date]

        return clustering


    @staticmethod
    def load_models_inference(directory: str, tickers: list) -> dict:
        """
        Method to load the persisted ModelsInference instance for each ticker.

        Parameters
        ----------
        directory : str
            String representing the file path.
        tickers : list
            List of str ticker symbols.

        Returns
        -------
        parsed_objects : dict
            Dictionary of loaded pickle files representing persisted inference files.
        """
        parsed_objects = {}
        for ticker in tickers:
            pattern = os.path.join(directory, f"{ticker}.pkl")
            matched_files = glob.glob(pattern)
            objects = []
            for file_path in matched_files:
                with open(file_path, "rb") as f:
                    obj = pickle.load(f)
                    objects.append(obj)
            if objects:
                parsed_objects[ticker] = objects if len(objects) > 1 else objects[0]

        return parsed_objects


    @staticmethod
    def extract_states(parsed_objects: dict) -> dict:
        """
        Method to extract state data from the loaded ModelInference instance.

        Parameters
        ----------
        parsed_objects : dict
            Dictionary of loaded pickle files representing persisted inference files.

        Returns
        -------
        state_data : dict
            Dictionary of tickers and corresponding state data.
        """
        state_data = {}
        for ticker, obj in parsed_objects.items():
            states = []
            train_states = getattr(obj, "train_states", None)
            test_states = getattr(obj, "test_states", None)
            state_labels = getattr(obj, "state_labels", {})
            if train_states is not None:
                states.append(np.asarray(train_states))
            if test_states is not None:
                states.append(np.asarray(test_states))
            if states:
                combined_states = np.concatenate(states)
                labeled_states = np.vectorize(state_labels.get)(combined_states)
                state_data[ticker] = {
                    "raw": combined_states,
                    "labels": labeled_states
                }

        return state_data
    

    @staticmethod
    def moving_average_check(clustering, state_data, moving_average_window):
        """
        """
        lookback = clustering.moving_average.copy()
        lookback = self.config["moving_average"]
        valid_tickers = []

        for ticker in state_data:
            if ticker not in self.data.columns:
                continue

            prices = self.data[ticker].dropna()

            ma = prices.rolling(window=lookback).mean()

            if prices.iloc[-1] >= ma.iloc[-1]:
                valid_tickers.append(ticker)

        state_data = {t: state_data[t] for t in valid_tickers}
        if not state_data:
            state_data = {"SHV": 1}

        clustering.state_data = state_data


    @staticmethod
    def prepare_state_sequences(state_data: dict, lookback: int) -> tuple [np.ndarray, list]:
        """
        Method to parse state data into state sequences for clustering.

        Parameters
        ----------
        state_data : dict
            Dictionary of tickers and corresponding state data.
        lookback : int
            Integer representing the cutoff lookback period for clustering.

        Returns
        -------
        sequences: np.ndarray
            An array of state sequences.
        tickers : list
            List of string tickers.
        """
        all_labels = set()
        for data in state_data.values():
            trimmed = data["labels"][-lookback:]
            all_labels.update(trimmed)
        encoder = LabelEncoder()
        encoder.fit(list(all_labels))
        sequences = []
        tickers = []
        for ticker, data in state_data.items():
            trimmed = data["labels"][-lookback:]
            encoded = encoder.transform(trimmed)
            sequences.append(encoded)
            tickers.append(ticker)

        return np.array(sequences), tickers


    @staticmethod
    def extract_forecast_distributions(parsed_objects: dict) -> dict:
        """
        Method to extract forecast_data from the loaded ModelsInference instance.

        Parameters
        ----------
        parsed_objects : dict
            Dictionary of model inference objects with `forecast_distribution` attributes.

        Returns
        -------
        forecast_data : dict
            Dictionary of forecast distributions by ticker with keys normalized.
        """
        forecast_data = {}
        for ticker, obj in parsed_objects.items():
            forecast = getattr(obj, "forecast_distribution", None)
            if forecast is not None and isinstance(forecast, dict):
                if "Bullish" not in forecast:
                    mapped_forecast = {}
                    for k, v in forecast.items():
                        mapped_forecast[k] = v
                    forecast_data[ticker] = np.asarray(mapped_forecast)
                else:
                    forecast_data[ticker] = np.asarray(forecast)

        return forecast_data
