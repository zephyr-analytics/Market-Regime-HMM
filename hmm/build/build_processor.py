"""
"""
import glob
import pickle
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import hmm.utilities as utilities
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import LabelEncoder


class BuildProcessor:
    """
    """
    def __init__(self, config: dict):
        self.config = config

    def process(self):
        """
        Method to process through the BuildProcessor.
        """
        file_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing")
        parsed_objects = self.load_pickles_by_ticker(directory=file_path, tickers=self.config["tickers"])
        state_data = self.extract_states(parsed_objects=parsed_objects)
        seq_matrix, ticker_list = self.prepare_state_sequences(state_data, lookback=252)
        results = self.cluster_and_plot_sequence(seq_matrix, ticker_list, percentile=self.config["diversification_level"])
        clusters = results["clusters"]
        forecast_data = self.extract_forecast_distributions(parsed_objects=parsed_objects)
        category_weights = self.compute_categorical_weights_by_cluster(
            forecast_data=forecast_data, clusters=clusters, bearish_cutoff=self.config["bearish_cutoff"]
        )

        portfolio = self.build_final_portfolio(
            clusters=clusters,
            forecast_data=forecast_data,
            category_weights=category_weights,
            bearish_cutoff=self.config["bearish_cutoff"]
        )

        self.plot_portfolio(ticker_weights=portfolio)

    @staticmethod
    def load_pickles_by_ticker(directory: str, tickers: list) -> dict:
        """

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

        Parameters
        ----------
        parsed_objects : 

        Returns
        -------
        state_data : 

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
    def prepare_state_sequences(state_data: dict, lookback: int=63) -> np.ndarray:
        """

        Parameters
        ----------
        state_data : 

        lookback : 

        Returns 
        np.ndarray : 
        """
        all_labels = set()
        for data in state_data.values():
            trimmed = data['labels'][-lookback:]
            all_labels.update(trimmed)

        encoder = LabelEncoder()
        encoder.fit(list(filter(None, all_labels)))\

        sequences = []
        tickers = []

        for ticker, data in state_data.items():
            trimmed = data['labels'][-lookback:]
            encoded = encoder.transform(trimmed)
            sequences.append(encoded)
            tickers.append(ticker)

        return np.array(sequences), tickers

    @staticmethod
    def cluster_and_plot_sequence(sequences: np.ndarray, tickers: list, percentile: float) -> dict:
        """

        Parameters
        ----------
        sequences : 

        tickers : 

        percentile : 

        Returns
        -------

        """
        distance_matrix = pdist(sequences, metric='euclidean')
        Z = linkage(distance_matrix, method='ward')

        linkage_distances = Z[:, 2]
        threshold = np.percentile(linkage_distances, percentile)

        labels = fcluster(Z, t=threshold, criterion='distance')
        cluster_map = dict(zip(tickers, labels))

        plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=tickers, leaf_rotation=90)
        plt.axhline(y=threshold, c='red', linestyle='dashed', label=f'Threshold: {threshold:.2f} (P{percentile})')
        plt.title("Hierarchical Clustering of Tickers by State Sequences")
        plt.xlabel("Ticker")
        plt.ylabel("Distance")
        plt.legend()
        plt.tight_layout()
        utilities.save_plot(filename="cluster_distribution.png", plot_type="cluster_distribution", plot_sub_folder="build")
        plt.close()

        return {
            'linkage_matrix': Z,
            'clusters': cluster_map,
            'labels': labels,
            'threshold': threshold
        }


    @staticmethod
    def extract_forecast_distributions(parsed_objects: dict) -> dict:
        """
        Extracts forecast distributions from parsed inference objects and normalizes state keys
        to 'Bullish' if 'Bullish' is not already present.

        Parameters
        ----------
        parsed_objects : dict
            Dictionary of model inference objects with `forecast_distribution` attributes.

        Returns
        -------
        dict
            Dictionary of forecast distributions by ticker with keys normalized.
        """
        forecast_data = {}

        for ticker, obj in parsed_objects.items():
            forecast = getattr(obj, 'forecast_distribution', None)
            if forecast is not None and isinstance(forecast, dict):
                # Only map if 'Bullish' is not already present
                if "Bullish" not in forecast:
                    mapped_forecast = {}
                    for k, v in forecast.items():
                        if k in {"State 0", "State 1", "State 2"}:
                            mapped_forecast["Bullish"] = mapped_forecast.get("Bullish", 0.0) + v
                        else:
                            mapped_forecast[k] = v
                    forecast_data[ticker] = np.asarray(mapped_forecast)
                else:
                    forecast_data[ticker] = np.asarray(forecast)
        print(forecast_data)
        return forecast_data


    @staticmethod
    def compute_categorical_weights_by_cluster(forecast_data: dict, clusters: dict, bearish_cutoff: float) -> dict:
        """
        Computes normalized weights for Bullish, Neutral, and Bearish forecasts per cluster.
        Clusters with an average Bearish sentiment > 0.15 are assigned zero weight across all categories.

        Parameters
        ----------
        forecast_data : dict
            Mapping of tickers to forecast dictionaries or np.ndarrays containing {"Bullish", "Neutral", "Bearish"} scores.
        clusters : dict
            Mapping of tickers to cluster IDs.

        Returns
        -------
        dict
            Nested dictionary where category_weights[category][cluster_id] = normalized weight.
        """
        valid_categories = ['Bullish', 'Neutral', 'Bearish']
        cluster_category_sums = defaultdict(lambda: {cat: 0.0 for cat in valid_categories})
        cluster_counts = defaultdict(int)
        cluster_bearish_totals = defaultdict(float)

        for ticker, forecast_array in forecast_data.items():
            cluster_id = clusters.get(ticker)
            if cluster_id is None:
                continue

            forecast_dict = forecast_array.item() if isinstance(forecast_array, np.ndarray) else forecast_array
            if not isinstance(forecast_dict, dict):
                continue

            for cat in valid_categories:
                cluster_category_sums[cluster_id][cat] += forecast_dict.get(cat, 0.0)

            cluster_bearish_totals[cluster_id] += forecast_dict.get("Bearish", 0.0)
            cluster_counts[cluster_id] += 1

        for cluster_id in list(cluster_category_sums.keys()):
            count = cluster_counts.get(cluster_id, 1)
            avg_bearish = cluster_bearish_totals[cluster_id] / count if count > 0 else 0.0
            if avg_bearish > bearish_cutoff:
                cluster_category_sums[cluster_id] = {cat: 0.0 for cat in valid_categories}

        total_per_category = {cat: 0.0 for cat in valid_categories}
        for cluster_vals in cluster_category_sums.values():
            for cat in valid_categories:
                total_per_category[cat] += cluster_vals[cat]

        category_weights = {cat: {} for cat in valid_categories}
        for cluster_id, sums in cluster_category_sums.items():
            for cat in valid_categories:
                total = total_per_category[cat]
                category_weights[cat][cluster_id] = sums[cat] / total if total > 0 else 0.0
        print(category_weights)
        return category_weights


    @staticmethod
    def build_final_portfolio(clusters: dict, forecast_data: dict, category_weights: dict, bearish_cutoff: float):
        """
        Builds a final portfolio by allocating weights to tickers based on their forecasted category scores
        and cluster-based category weights. Bullish scores are discounted by Bearish sentiment.

        Parameters
        ----------
        clusters : dict
            Mapping of tickers to cluster IDs.
        forecast_data : dict
            Mapping of tickers to forecast dictionaries or arrays with {"Bullish", "Neutral", "Bearish"} scores.
        category_weights : dict
            Nested dictionary with category -> cluster_id -> weight.

        Returns
        -------
        dict
            Final ticker weight allocations.
        """
        valid_categories = ['Bullish', 'Neutral', 'Bearish']
        ticker_weights = defaultdict(float)
        orphaned_weight = 0.0

        for category in valid_categories:
            cluster_weights = category_weights.get(category, {})

            for cluster_id, cluster_weight in cluster_weights.items():
                tickers_in_cluster = [tkr for tkr, cid in clusters.items() if cid == cluster_id]

                scores = []
                for tkr in tickers_in_cluster:
                    forecast = forecast_data.get(tkr)
                    if isinstance(forecast, np.ndarray):
                        forecast = forecast.item()
                    if not forecast or not isinstance(forecast, dict):
                        continue

                    if forecast.get("Bearish", 0.0) > bearish_cutoff:
                        continue

                    bullish_score = max(forecast.get("Bullish", 0.0) - forecast.get("Bearish", 0.0), 0.0)
                    scores.append((tkr, bullish_score))

                if not scores:
                    orphaned_weight += cluster_weight
                    continue

                total_bullish = sum(score for _, score in scores)
                if total_bullish == 0:
                    equal_weight = cluster_weight / len(scores)
                    for tkr, _ in scores:
                        ticker_weights[tkr] += equal_weight
                else:
                    for tkr, score in scores:
                        norm_score = score / total_bullish
                        ticker_weights[tkr] += cluster_weight * norm_score

        if orphaned_weight > 0 and ticker_weights:
            total_allocated = sum(ticker_weights.values())
            for tkr in ticker_weights:
                proportion = ticker_weights[tkr] / total_allocated
                ticker_weights[tkr] += orphaned_weight * proportion

        total = sum(ticker_weights.values())
        if total > 0:
            ticker_weights = {tkr: w / total for tkr, w in ticker_weights.items()}
        dict(ticker_weights)

        return ticker_weights


    @staticmethod
    def plot_portfolio(ticker_weights: dict):
        """

        Parameters
        ----------
        ticker_wegihts : 

        """
        if not ticker_weights:
            print("No weights to plot.")
            return

        sorted_items = sorted(ticker_weights.items(), key=lambda x: x[1], reverse=True)
        labels, weights = zip(*sorted_items)

        plt.figure(figsize=(10, 10))
        plt.pie(
            weights,
            labels=[f"{label} ({w:.2%})" for label, w in zip(labels, weights)],
            startangle=140,
            counterclock=False,
            wedgeprops=dict(edgecolor='w'),
            textprops={'fontsize': 9}
        )
        plt.title("Final Portfolio Composition")
        plt.axis('equal')
        plt.tight_layout()
        utilities.save_plot(filename="portfolio_allocation.png", plot_type="portfolio_allocation", plot_sub_folder="build")
        plt.close()
