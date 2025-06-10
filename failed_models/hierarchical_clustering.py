"""
"""

# NOTE this model could be better, but the current implementation struggles with predictive power
# of just using sequences of states.

import logging

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler

import hmm.utilities as utilities

logger = logging.getLogger(__name__)

@staticmethod
def cluster_sequences(sequences: np.ndarray, tickers: list, max_clusters: int=15) -> dict:
    """
    Method to cluster state sequences to determine portfolio categories.

    Parameters
    ----------
    sequences : np.ndarray
        Numpy array of state sequences.
    tickers : list
        List of ticker symbols.
    max_clusters : int
        Upper limit of allowed clusters.

    Returns
    -------
    dict : Dictionary containing cluster components.
    """
    min_clusters = 3
    epsilon = 1e-10
    sequences = np.array(sequences, dtype=np.float64)
    sequences = np.nan_to_num(sequences, nan=0.0)
    row_norms = np.linalg.norm(sequences, axis=1)
    zero_mask = row_norms == 0
    sequences[zero_mask] = epsilon

    distance_matrix = pdist(sequences, metric='euclidean')
    Z = linkage(distance_matrix, method='ward')

    scores, label_map = utilities.evaluate_clustering_scores(
        sequences, Z, min_clusters, min(max_clusters, len(sequences))
    )

    scores[:, 2] = -scores[:, 2]

    scaler = MinMaxScaler()
    scaled_scores = scaler.fit_transform(scores)
    mean_scores = scaled_scores.mean(axis=1)

    valid_ks = np.array(list(label_map.keys()))
    best_idx = np.argmax(mean_scores)
    best_k = valid_ks[best_idx]

    if best_k < min_clusters:
        k_candidates = valid_ks[valid_ks >= min_clusters]
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
