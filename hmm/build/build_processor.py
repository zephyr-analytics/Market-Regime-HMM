"""
"""
import glob
import pickle
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
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
        """
        file_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing")
        parsed_objects = self.load_pickles_by_ticker(directory=file_path, tickers=self.config["tickers"])
        state_data = self.extract_states(parsed_objects=parsed_objects)
        seq_matrix, ticker_list = self.prepare_state_sequences(state_data, lookback=252)
        results = self.cluster_and_plot_sequence(seq_matrix, ticker_list)
        clusters = results["clusters"]
        forecast_data = self.extract_forecast_distributions(parsed_objects=parsed_objects)
        category_weights = self.compute_categorical_weights_by_cluster(forecast_data=forecast_data, clusters=clusters)

        portfolio = self.build_final_portfolio(
            clusters=clusters,
            forecast_data=forecast_data,
            category_weights=category_weights
        )

        self.plot_portfolio(ticker_weights=portfolio)


    @staticmethod
    def load_pickles_by_ticker(directory: str, tickers: list) -> dict:
        """
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
    def prepare_state_sequences(state_data: dict, lookback: int=63):
        """
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
    def cluster_and_plot_sequence(sequences: np.ndarray, tickers: list, percentile: float = 80.0) -> dict:
        """
        Performs hierarchical clustering on sequences and dynamically determines the distance threshold
        based on a given percentile of the linkage distances.

        Args:
            sequences (np.ndarray): Encoded sequences per asset.
            tickers (list): List of asset tickers corresponding to the sequences.
            percentile (float): Percentile (0-100) to select the threshold from linkage distances.

        Returns:
            dict: A dictionary containing the linkage matrix, cluster labels, cluster mapping, and threshold.
        """
        # Compute pairwise distances and linkage matrix
        distance_matrix = pdist(sequences, metric='euclidean')
        Z = linkage(distance_matrix, method='ward')

        # Compute threshold from a chosen percentile of the linkage distances
        linkage_distances = Z[:, 2]
        threshold = np.percentile(linkage_distances, percentile)

        # Generate flat cluster labels
        labels = fcluster(Z, t=threshold, criterion='distance')
        cluster_map = dict(zip(tickers, labels))

        # Plot dendrogram with threshold line
        plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=tickers, leaf_rotation=90)
        plt.axhline(y=threshold, c='red', linestyle='dashed', label=f'Threshold: {threshold:.2f} (P{percentile})')
        plt.title("Hierarchical Clustering of Tickers by State Sequences")
        plt.xlabel("Ticker")
        plt.ylabel("Distance")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Return structured results
        return {
            'linkage_matrix': Z,
            'clusters': cluster_map,
            'labels': labels,
            'threshold': threshold
        }

    @staticmethod
    def extract_forecast_distributions(parsed_objects: dict) -> dict:
        """
        """
        forecast_data = {}

        for ticker, obj in parsed_objects.items():
            forecast = getattr(obj, 'forecast_distribution', None)
            if forecast is not None:
                forecast_data[ticker] = np.asarray(forecast)
        print(forecast_data)
        return forecast_data

    @staticmethod
    def compute_categorical_weights_by_cluster(forecast_data: dict, clusters: dict) -> dict:
        """
        """
        from collections import defaultdict

        valid_categories = ['Bullish', 'Neutral', 'Bearish']
        cluster_category_sums = defaultdict(lambda: {cat: 0.0 for cat in valid_categories})

        for ticker, forecast_array in forecast_data.items():
            cluster_id = clusters.get(ticker)
            if cluster_id is None:
                continue

            # Extract forecast dict from numpy array wrapper
            forecast_dict = forecast_array.item() if isinstance(forecast_array, np.ndarray) else forecast_array

            for category in valid_categories:
                value = forecast_dict.get(category)
                if value is not None:
                    cluster_category_sums[cluster_id][category] += value

        # Compute total sums across clusters per category
        total_per_category = {cat: 0.0 for cat in valid_categories}
        for cluster_vals in cluster_category_sums.values():
            for cat in valid_categories:
                total_per_category[cat] += cluster_vals[cat]

        # Normalize weights per category
        category_weights = {cat: {} for cat in valid_categories}
        for cluster_id, sums in cluster_category_sums.items():
            for cat in valid_categories:
                total = total_per_category[cat]
                category_weights[cat][cluster_id] = sums[cat] / total if total > 0 else 0.0

        return category_weights

    @staticmethod
    def build_final_portfolio(clusters: dict, forecast_data: dict, category_weights: dict):
        """
        """
        valid_categories = ['Bullish', 'Neutral', 'Bearish']
        ticker_weights = defaultdict(float)
        orphaned_weight = 0.0  # to collect unused category-cluster weights

        for category in valid_categories:
            cluster_weights = category_weights.get(category, {})

            for cluster_id, cluster_weight in cluster_weights.items():
                # Get tickers in this cluster
                tickers_in_cluster = [tkr for tkr, cid in clusters.items() if cid == cluster_id]

                # Filter and score
                scores = []
                for tkr in tickers_in_cluster:
                    forecast = forecast_data.get(tkr)
                    if isinstance(forecast, np.ndarray):
                        forecast = forecast.item()
                    if not forecast or not isinstance(forecast, dict):
                        continue

                    if forecast.get("Bearish", 0.0) > 0.15:
                        continue  # drop asset

                    bullish_score = forecast.get("Bullish", 0.0)
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

        # Redistribute orphaned_weight to remaining tickers proportionally
        if orphaned_weight > 0 and ticker_weights:
            total_allocated = sum(ticker_weights.values())
            for tkr in ticker_weights:
                proportion = ticker_weights[tkr] / total_allocated
                ticker_weights[tkr] += orphaned_weight * proportion

        # Final normalization
        total = sum(ticker_weights.values())
        if total > 0:
            ticker_weights = {tkr: w / total for tkr, w in ticker_weights.items()}
        ticker_weights = dict(ticker_weights)

        return ticker_weights

    @staticmethod
    def plot_portfolio(ticker_weights: dict):
        """
        """
        if not ticker_weights:
            print("⚠️ No weights to plot.")
            return

        # Sort for visual clarity (largest first)
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
        plt.axis('equal')  # Ensures the pie is round
        plt.tight_layout()
        plt.show()
