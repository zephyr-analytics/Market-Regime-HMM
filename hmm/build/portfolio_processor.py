"""
Module for building portfolio.
"""

import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from hmm import utilities
from hmm.build.portfolio_clustering import PortfolioClustering
from hmm.build.portfolio_constructor import PortfolioConstructor
from hmm.results.portfolio_results_processor import PortfolioResultsProcessor

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
        self.data = data.loc[self.start_date:self.end_date]

    def process(self):
        """
        Method for processing through the pipeline.
        """
        file_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing")
        clustering = self.initialize_portfolio_clustering(config=self.config, data=self.data)
        self.load_models_inference(clustering=clustering, directory=file_path, tickers=self.config["tickers"])
        self.extract_states(clustering=clustering)
        self.moving_average_check(clustering=clustering)
        # TODO sequence lookback needs to be set properly.
        self.prepare_state_sequences(clustering=clustering, lookback=self.config["sequence_lookback"])
        self.extract_forecast_distributions(clustering=clustering)

        results = self.cluster_sequences(clustering=clustering)

        clusters = results["clusters"]
        forecast_data = clustering.forecast_data

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

        Parameters
        ----------
        config : dict

        data : pd.DataFrame

        """
        clustering = PortfolioClustering()
        clustering.min_clusters = config["min_clusters"]
        clustering.max_clusters = config["max_clusters"]
        clustering.start_date = config["current_start"]
        clustering.end_date = config["current_end"]
        clustering.moving_average = config["moving_average"]
        clustering.price_data = data

        return clustering


    @staticmethod
    def load_models_inference(clustering: PortfolioClustering, directory: str, tickers: list):
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

        clustering.parsed_objects = parsed_objects


    @staticmethod
    def extract_states(clustering: PortfolioClustering):
        """
        Method to extract state data from the loaded ModelInference instance.

        Parameters
        ----------
        parsed_objects : dict
            Dictionary of loaded pickle files representing persisted inference files.
        """
        parsed_objects = clustering.parsed_objects.copy()
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

        clustering.state_data = state_data


    @staticmethod
    def moving_average_check(clustering: PortfolioClustering):
        """
        Method to moving average filter assets.

        Parameters
        ----------
        clustering : PortfolioClustering
            PortfolioClustering instance.
        """
        lookback = clustering.moving_average
        data = clustering.price_data.copy()
        state_data = clustering.state_data.copy()

        valid_tickers = []

        for ticker in state_data:
            if ticker not in data.columns:
                continue

            prices = data[ticker].dropna()

            ma = prices.rolling(window=lookback).mean()

            if prices.iloc[-1] >= ma.iloc[-1]:
                valid_tickers.append(ticker)

        state_data = {t: state_data[t] for t in valid_tickers}
        if not state_data:
            state_data = {"SHV": 1}

        clustering.state_data = state_data


    @staticmethod
    def prepare_state_sequences(clustering: PortfolioClustering, lookback: int):
        """
        Method to parse state data into state sequences for clustering.

        Parameters
        ----------
        clustering : PortfolioClustering
            PortfolioClustering instance.
        lookback : int
            Integer representing the cutoff lookback period for clustering.
        """
        state_data = clustering.state_data.copy()

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

        clustering.sequences = sequences
        clustering.clustering_tickers = tickers


    @staticmethod
    def extract_forecast_distributions(clustering: PortfolioClustering):
        """
        Method to extract forecast_data from the loaded ModelsInference instance.

        Parameters
        ----------
        clustering : PortfolioClustering
            PortfolioClustering instance.
        """
        parsed_objects = clustering.parsed_objects.copy()
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

        clustering.forecast_data = forecast_data

# TODO this needs to be condensed down to handle usage of clustering instance.
    @staticmethod
    def cluster_sequences(clustering: PortfolioClustering):
        """
        Method to cluster state sequences to determine portfolio categories.

        Parameters
        ----------
        clustering : PortfolioClustering
            PortfolioClustering instance.
        """
        tickers = clustering.clustering_tickers
        min_clusters = clustering.min_clusters
        max_clusters = clustering.max_clusters
        sequences = clustering.sequences

        epsilon = 1e-10

        sequences = np.array(sequences, dtype=np.float64)
        sequences = np.nan_to_num(sequences, nan=0.0)
        row_norms = np.linalg.norm(sequences, axis=1)
        zero_mask = row_norms == 0
        sequences[zero_mask] = epsilon

        if len(sequences) < 2:
            fallback_labels = np.ones(len(sequences), dtype=int)
            cluster_map = dict(zip(tickers, fallback_labels))
            return {
                'linkage_matrix': None,
                'clusters': cluster_map,
                'labels': fallback_labels,
                'n_clusters': 1
            }

        distance_matrix = pdist(sequences, metric='euclidean')
        Z = linkage(distance_matrix, method='ward')

        scores, label_map = utilities.evaluate_clustering_scores(
            sequences, Z, min_clusters, min(max_clusters, len(sequences))
        )

        if not label_map:
            fallback_labels = np.ones(sequences.shape[0], dtype=int)
            cluster_map = dict(zip(tickers, fallback_labels))
            return {
                'linkage_matrix': Z,
                'clusters': cluster_map,
                'labels': fallback_labels,
                'n_clusters': 1
            }

        scores[:, 2] = -scores[:, 2]

        scaler = MinMaxScaler()
        scaled_scores = scaler.fit_transform(scores)
        mean_scores = scaled_scores.mean(axis=1)

        valid_ks = np.array(list(label_map.keys()))
        best_idx = np.argmax(mean_scores)
        best_k = valid_ks[best_idx]

        if best_k < min_clusters:
            k_candidates = valid_ks[valid_ks >= min_clusters]
            if len(k_candidates) == 0:
                fallback_labels = np.ones(sequences.shape[0], dtype=int)
                cluster_map = dict(zip(tickers, fallback_labels))
                return {
                    'linkage_matrix': Z,
                    'clusters': cluster_map,
                    'labels': fallback_labels,
                    'n_clusters': 1
                }
            best_k = k_candidates[np.argmax(mean_scores[valid_ks >= min_clusters])]

        best_labels = label_map[best_k]
        cluster_map = dict(zip(tickers, best_labels))
        logger.info(f"Cluster Map: {cluster_map}")

        return {
            'linkage_matrix': Z,
            'clusters': cluster_map,
            'labels': best_labels,
            'n_clusters': best_k
        }
