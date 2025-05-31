"""
"""
import glob
import pickle
import os

import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt


class BuildProcessor:
    """
    """
    def __init__(self, config):
        self.config = config

    def process(self):
        """
        """
        file_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "inferencing")
        parsed_objects = self.load_pickles_by_ticker(directory=file_path, tickers=self.config["tickers"])
        state_data = self.extract_states(parsed_objects=parsed_objects)
        seq_matrix, ticker_list = self.prepare_state_sequences(state_data, lookback=252)
        results = self.cluster_and_plot_sequence(seq_matrix, ticker_list)
        Z = results["linkage_matrix"]
        clusters = results["clusters"]
        labels = results["labels"]
        predicted_states = self.predict_future_states(parsed_objects=parsed_objects)


    @staticmethod
    def load_pickles_by_ticker(directory, tickers):
        """
        Load pickle files from a directory that match the given tickers.

        Args:
            directory (str): Path to the directory containing .pkl files.
            tickers (list): List of ticker strings to search for in filenames.

        Returns:
            dict: A dictionary mapping each ticker to its deserialized object(s).
                If multiple files match a ticker, a list of objects is stored.
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
    def extract_states(parsed_objects):
        """
        Extract and label train/test states from parsed model objects.

        Args:
            parsed_objects (dict): Mapping of ticker -> parsed model object.

        Returns:
            dict: Mapping of ticker -> dict with raw and labeled state arrays:
                {
                    'raw': np.ndarray of combined train/test states (int),
                    'labels': np.ndarray of combined labels (str)
                }
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
    def prepare_state_sequences(state_data, lookback=252):
        """
        Convert the last `lookback` labeled state values into a matrix
        of integer-encoded sequences (preserving temporal order).

        Args:
            state_data (dict): {ticker: {'labels': np.ndarray of state labels}}
            lookback (int): Number of most recent states to include.

        Returns:
            tuple: (np.ndarray of sequences, list of tickers)
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
    def cluster_and_plot_sequence(sequences, tickers, threshold=15.0, criterion='distance'):
            """
            Perform hierarchical clustering on encoded label sequences,
            plot dendrogram, and return clustering results.

            Args:
                sequences (np.ndarray): 2D array of encoded state labels.
                tickers (list): List of tickers corresponding to rows in `sequences`.
                threshold (float): Threshold to cut the dendrogram for cluster labels.
                criterion (str): Criterion to use in fcluster (e.g., 'distance', 'maxclust').

            Returns:
                dict: {
                    'linkage_matrix': np.ndarray,
                    'clusters': dict[ticker -> cluster_id],
                    'labels': np.ndarray of cluster labels
                }
            """
            # Compute pairwise distances and linkage matrix
            distance_matrix = pdist(sequences, metric='euclidean')
            Z = linkage(distance_matrix, method='ward')

            # Generate flat cluster labels
            labels = fcluster(Z, t=threshold, criterion=criterion)
            cluster_map = dict(zip(tickers, labels))

            # Plot dendrogram
            plt.figure(figsize=(12, 6))
            dendrogram(Z, labels=tickers, leaf_rotation=90)
            plt.title("Hierarchical Clustering of Tickers by State Sequences")
            plt.xlabel("Ticker")
            plt.ylabel("Distance")
            plt.tight_layout()
            plt.show()

            # Return structured results
            return {
                'linkage_matrix': Z,
                'clusters': cluster_map,
                'labels': labels
            }
