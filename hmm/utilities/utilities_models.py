"""
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def calculate_portfolio_return(
        portfolio: dict, data: pd.DataFrame, start_date: str, end_date: str
    ):
    """
    Method to calculate portfolio returns. 

    Parameters
    ----------
    portfolio : dict
        Dictionary of ticker keys and weight values.
    data : pd.DataFrame
        Dataframe of price data.
    start_date : str
        String representing the start date.
    end_date : str
        String representing the end date.
    """
    if not portfolio:
        return 0.0

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    tickers = list(portfolio.keys())
    weights = pd.Series(portfolio)

    price_df = data[tickers]
    available_dates = price_df.index

    start_date = available_dates[available_dates >= start_date].min()
    end_date = available_dates[available_dates <= end_date].max()

    start_prices = price_df.loc[start_date]
    end_prices = price_df.loc[end_date]

    returns = (end_prices - start_prices) / start_prices
    portfolio_return = (returns * weights).sum()

    return portfolio_return


@staticmethod
def evaluate_clustering_scores(sequences: np.ndarray, linkage_matrix, min_clusters: int, max_clusters: int) -> tuple:
    """
    Evaluate clustering performance metrics across a range of cluster counts.

    Parameters
    ----------
    sequences : np.ndarray
        The data to be clustered.
    linkage_matrix : np.ndarray
        The hierarchical clustering linkage matrix.
    min_clusters : int
        Minimum number of clusters to test.
    max_clusters : int
        Maximum number of clusters to test.

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
        sil = silhouette_score(sequences, labels)
        ch = calinski_harabasz_score(sequences, labels)
        db = davies_bouldin_score(sequences, labels)
        scores.append([sil, ch, db])
        label_map[k] = labels

    return np.array(scores), label_map


def label_states(training=None, inferencing=None) -> dict:
    """
    Assigns labels 'Bearish', 'Neutral', 'Bullish' to HMM state IDs based on z-score of returns.
    Ensures all three labels are present by assigning duplicate labels to existing states if needed.

    Parameters
    ----------
    training : ModelsTraining
        ModelsTraining instance.
    inferencing : ModelsInferencing
        ModelsInferencing instance.

    Returns
    -------
    label_map : dict
        Dictionary with keys as HMM state IDs and values as labels.
    """
    if training:
        states = training.train_states.copy()
        returns = training.train_data.copy()
    elif inferencing:
        states = np.concatenate([inferencing.train_states, inferencing.test_states])
        returns = pd.concat([inferencing.train_data, inferencing.test_data], ignore_index=False)

    returns = returns['Momentum'].reset_index(drop=True)

    state_returns = []
    unique_states = np.unique(states)
    for state in unique_states:
        mask = states == state
        mean_ret = returns[mask].mean() if mask.any() else 0.0
        state_returns.append((state, mean_ret))

    mean_vals = np.array([r[1] for r in state_returns])
    zscores = zscore(mean_vals) if len(mean_vals) > 1 else np.array([0.0])

    sorted_states = [state_returns[i][0] for i in np.argsort(zscores)]

    label_order = ['Bearish', 'Neutral', 'Bullish']
    label_map = {}

    for i, label in enumerate(label_order):
        state = sorted_states[i]
        label_map[state] = label

    return label_map
