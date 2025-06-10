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


from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import hmm.utilities as utilities
from hmm.build.portfolio_constructor import PortfolioConstructor
from hmm.results.portfolio_results_processor import PortfolioResultsProcessor

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
        seq_matrix, ticker_list = self.prepare_state_sequences(state_data, lookback=126)
        results = self.cluster_sequences(seq_matrix, ticker_list)
        clusters = results["clusters"]
        forecast_data = self.extract_forecast_distributions(parsed_objects=parsed_objects)
        category_weights = self.compute_categorical_weights_by_cluster(
            forecast_data=forecast_data, clusters=clusters
        )
        
        constructor = PortfolioConstructor(
            config=self.config, clusters=clusters, forecast_data=forecast_data, category_weights=category_weights, price_data=self.data
        )
        portfolio = constructor.process()

        if self.persist:
            # NOTE possibly use a getter and setter for all results.
            results_process = PortfolioResultsProcessor(
                config=self.config,
                Z=results["linkage_matrix"],
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
            Dictionary of tickers and cooresponding state data.
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

# Though the rest of sequence cluster is being scraped relabeling states to be all the same is important. 
# This should be shifted into the inference processor.
    @staticmethod
    def prepare_state_sequences(state_data: dict, lookback: int) -> np.ndarray:
        """
        Method to parse state data into state sequences for clustering.

        Parameters
        ----------
        state_data : dict
            Dictionary of tickers and cooresponding state data.
        lookback : int
            Integer representing the cutoff lookback period for clustering.

        Returns 
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


# TODO Fill with new clustering based on state probability.


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
        cluster_adjusted_bullish = defaultdict(float)

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
            adjusted_bullish = bullish - bearish

            cluster_scores[cluster_id]["Bullish"] += bullish
            cluster_scores[cluster_id]["Neutral"] += neutral
            cluster_scores[cluster_id]["Bearish"] += bearish
            cluster_adjusted_bullish[cluster_id] += adjusted_bullish

        total_per_category = {cat: 0.0 for cat in valid_categories}
        for cluster_vals in cluster_scores.values():
            for cat in valid_categories:
                total_per_category[cat] += cluster_vals[cat]

        raw_weights = {cat: {} for cat in valid_categories}
        for cluster_id, scores in cluster_scores.items():
            for cat in valid_categories:
                total = total_per_category[cat]
                raw_weights[cat][cluster_id] = scores[cat] / total if total > 0 else 0.0

        # Apply adjusted bullish discounting
        category_weights = {cat: {} for cat in valid_categories}
        for cat in valid_categories:
            weighted = {
                cid: weight * max(cluster_adjusted_bullish.get(cid, 0.0), 0.0)
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
