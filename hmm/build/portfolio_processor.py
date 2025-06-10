"""
Module for building portfolio.
"""

import glob
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hmm.utilities as utilities
from hmm.build.portfolio_constructor import PortfolioConstructor
from hmm.results.portfolio_results_processor import PortfolioResultsProcessor
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from failed_models.hierarchical_clustering import cluster_sequences

logger = logging.getLogger(__name__)


class PortfolioProcessor:
    """
    Class to take processed model data and build a portfolio.
    """
    def __init__(self, config: dict, data: pd.DataFrame):
        self.config = config
        self.start_date = config["start_date"]
        self.end_date = config["current_end"]
        self.persist = config["persist"]
        self.data = data.loc[self.start_date:self.end_date]


    def process(self):
        """
        Method to process through the PortfolioProcessor.
        """
        file_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing")
        parsed_objects = self.load_models_inference(directory=file_path, tickers=self.config["tickers"])
        state_data = self.extract_states(parsed_objects=parsed_objects)
        sequences, tickers = self.prepare_state_sequences(state_data, lookback=126)
        forecast_data = self.extract_forecast_distributions(parsed_objects=parsed_objects)
        results = cluster_sequences(sequences=sequences, tickers=tickers)
        clusters = results["clusters"]
        # clusters = self.cluster_forecast_distributions_auto(forecast_data, max_clusters=15)

        category_weights = self.compute_categorical_weights_by_cluster(
            forecast_data=forecast_data, clusters=clusters
        )

        constructor = PortfolioConstructor(
            config=self.config, clusters=clusters, forecast_data=forecast_data, category_weights=category_weights, price_data=self.data
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
            pattern = os.path.join(directory, f'{ticker}.pkl')
            matched_files = glob.glob(pattern)
            objects = []
            for file_path in matched_files:
                with open(file_path, 'rb') as f:
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
            train_states = getattr(obj, 'train_states', None)
            test_states = getattr(obj, 'test_states', None)
            state_labels = getattr(obj, 'state_labels', {})
            if train_states is not None:
                states.append(np.asarray(train_states))
            if test_states is not None:
                states.append(np.asarray(test_states))
            if states:
                combined_states = np.concatenate(states)
                labeled_states = np.vectorize(state_labels.get)(combined_states)
                state_data[ticker] = {
                    'raw': combined_states,
                    'labels': labeled_states
                }

        return state_data


    @staticmethod
    def prepare_state_sequences(state_data: dict, lookback: int) -> np.ndarray:
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
        np.ndarray : An array of state sequences.
        """
        all_labels = set()
        for data in state_data.values():
            trimmed = data['labels'][-lookback:]
            all_labels.update(trimmed)
        encoder = LabelEncoder()
        encoder.fit(list(all_labels))
        sequences = []
        tickers = []
        for ticker, data in state_data.items():
            trimmed = data['labels'][-lookback:]
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
            forecast = getattr(obj, 'forecast_distribution', None)
            if forecast is not None and isinstance(forecast, dict):
                if "Bullish" not in forecast:
                    mapped_forecast = {}
                    for k, v in forecast.items():
                        mapped_forecast[k] = v
                    forecast_data[ticker] = np.asarray(mapped_forecast)
                else:
                    forecast_data[ticker] = np.asarray(forecast)

        return forecast_data


    @staticmethod
    def compute_categorical_weights_by_cluster(forecast_data: dict, clusters: dict) -> dict:
        """
        Calculate cluster weights for each category based on forecast data,
        and discount final weights by (Bullish - Bearish) to prioritize net bullish sentiment.

        Parameters
        ----------
        forecast_data : dict
            Dictionary of ticker to forecast probability dictionaries.
        clusters : dict
            Dictionary of ticker to cluster ID mapping.

        Returns
        -------
        category_weights : dict
            Dictionary of cluster weights for each category, normalized and discounted.
        """
        valid_categories = ['Bullish', 'Neutral', 'Bearish']
        cluster_scores = defaultdict(lambda: {cat: 0.0 for cat in valid_categories})

        for ticker, forecast_array in forecast_data.items():
            cluster_id = clusters.get(ticker)
            if cluster_id is None:
                continue
            forecast_dict = forecast_array.item() if isinstance(forecast_array, np.ndarray) else forecast_array
            if not isinstance(forecast_dict, dict):
                continue
            bullish = forecast_dict.get("Bullish", 0.0)
            bearish = forecast_dict.get("Bearish", 0.0)
            neutral = forecast_dict.get("Neutral", 0.0)

            cluster_scores[cluster_id]["Bullish"] += bullish
            cluster_scores[cluster_id]["Neutral"] += neutral
            cluster_scores[cluster_id]["Bearish"] += bearish

        total_per_category = {cat: 0.0 for cat in valid_categories}
        for cluster_vals in cluster_scores.values():
            for cat in valid_categories:
                total_per_category[cat] += cluster_vals[cat]
        raw_weights = {cat: {} for cat in valid_categories}
        for cluster_id, scores in cluster_scores.items():
            for cat in valid_categories:
                total = total_per_category[cat]
                raw_weights[cat][cluster_id] = scores[cat] / total if total > 0 else 0.0
        category_weights = {cat: {} for cat in valid_categories}
        for cat in valid_categories:
            weighted = {
                cid: weight
                for cid, weight in raw_weights[cat].items()
            }
            total = sum(weighted.values())
            if total > 0:
                category_weights[cat] = {
                    cid: w / total for cid, w in weighted.items()
                }
            else:
                category_weights[cat] = {cid: 0.0 for cid in weighted}

        return category_weights


    @staticmethod
    def evaluate_scores(X, labels_dict):
        """
        Evaluate clustering metrics for multiple cluster labelings.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        labels_dict : dict
            Dictionary of k -> labels from clustering.

        Returns
        -------
        tuple
            np.ndarray of scores and list of valid cluster counts.
        """
        scores = []
        valid_ks = list(labels_dict.keys())
        for k in valid_ks:
            labels = labels_dict[k]
            try:
                sil = silhouette_score(X, labels)
                ch = calinski_harabasz_score(X, labels)
                db = davies_bouldin_score(X, labels)
                scores.append([sil, ch, db])
            except:
                scores.append([0, 0, np.inf])

        return np.array(scores), valid_ks


    @staticmethod
    def cluster_forecast_distributions_auto(forecast_data: dict, max_clusters: int = 15) -> pd.Series:
        """
        Cluster assets based on their forecasted regime probability vectors using
        AgglomerativeClustering and automatic selection of number of clusters.

        Parameters
        ----------
        forecast_data : dict
            Dictionary of {ticker: np.ndarray of regime probabilities}.
        max_clusters : int
            Maximum number of clusters to consider.

        Returns
        -------
        pd.Series
            Series mapping tickers to their assigned cluster ID.
        """
        # print(forecast_data)
        states = ['Bullish', 'Bearish', 'Neutral']
        tickers = list(forecast_data.keys())

        matrix = np.array([
            [forecast_data[ticker].item().get(state, 0.0) for state in states]
            for ticker in tickers
        ], dtype=np.float64)

        matrix = np.nan_to_num(matrix, nan=0.0)

        distance_matrix = pairwise_distances(matrix, metric='euclidean')

        labels_dict = {}
        for k in range(3, min(max_clusters, len(tickers)) + 1):
            model = AgglomerativeClustering(n_clusters=k, metric='euclidean')
            labels = model.fit_predict(matrix)
            labels_dict[k] = labels
            # print(model.__dict__)
        scores, valid_ks = PortfolioProcessor.evaluate_scores(matrix, labels_dict)
        scores[:, 2] = -scores[:, 2]
        scaled_scores = MinMaxScaler().fit_transform(scores)
        mean_scores = scaled_scores.mean(axis=1)
        best_k = valid_ks[np.argmax(mean_scores)]
        best_labels = labels_dict[best_k]

        return pd.Series(best_labels, index=tickers, name="Cluster")
