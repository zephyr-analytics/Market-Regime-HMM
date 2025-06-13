"""
Module for hierarchical clustering model.
"""

import logging

import numpy as np

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

@staticmethod
def cluster_sequences(sequences: np.ndarray, tickers: list, max_clusters: int, min_clusters: int) -> dict:
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

    scores, label_map = evaluate_clustering_scores(
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


@staticmethod
def evaluate_clustering_scores(sequences: np.ndarray, linkage_matrix, min_clusters: int, max_clusters: int) -> tuple:
    """
    Evaluate clustering performance metrics across a range of cluster counts.

    Returns
    -------
    tuple
        scores : np.ndarray of shape (n_k, 3) with [silhouette, calinski_harabasz, -davies_bouldin]
        label_map : dict of {k: labels}
    """
    scores = []
    label_map = {}

    for k in range(min_clusters, max_clusters + 1):
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            continue

        try:
            sil = silhouette_score(sequences, labels)
        except ValueError:
            continue

        ch = calinski_harabasz_score(sequences, labels)
        db = davies_bouldin_score(sequences, labels)

        scores.append([sil, ch, db])
        label_map[k] = labels

    if not scores:
        fallback_labels = np.ones(sequences.shape[0], dtype=int)
        return np.array([[0.0, 0.0, 1.0]]), {1: fallback_labels}

    return np.array(scores), label_map
