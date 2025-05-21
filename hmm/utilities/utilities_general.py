"""
"""
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import linregress
from sklearn.cluster import KMeans

def compounded_return(series, window):
    """
    """
    daily_returns = series.pct_change().fillna(0) + 1
    return daily_returns.rolling(window).apply(lambda x: x.prod() - 1, raw=True)


def label_states(training):
    """
    Labels HMM states based on mean returns.

    Parameters:
    - training: Object with `train_states` (array) and `train_data` (DataFrame with 'Return' column)

    Returns:
    - state_label_dict: Dict mapping state index to label.
    - state_labels_array: Array of labels for each state in training.train_states.
    """
    states = training.train_states.copy()
    returns = training.train_data.copy()
    returns = returns['Momentum'].reset_index(drop=True)

    # Compute mean return for each state
    state_returns = []
    for state in np.unique(states):
        state_mask = states == state
        mean_return = returns[state_mask].mean()
        state_returns.append((state, mean_return))

    # Sort states by mean return
    state_returns.sort(key=lambda x: x[1])  # ascending order

    # Map based on rank
    labels = ['Bearish', 'Neutral', 'Bullish']
    state_label_dict = {
        state: labels[i] for i, (state, _) in enumerate(state_returns)
    }

    return state_label_dict


def load_from_pickle(file_path: str):
    """
    Load the ModelsTraining instance from a pickle file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def persist_to_pickle(file, file_path: str):
    """
    Pickle the ModelsTraining instance to a file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)


def smooth_states(states, window=5):
    """
    """
    return pd.Series(states).rolling(window, center=True, min_periods=1).apply(
        lambda x: stats.mode(x)[0][0], raw=False
    ).astype(int).values


def save_plot(filename, plot_type):
    """
    """
    directory = os.path.join("artifacts", plot_type)
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    plt.savefig(filepath)
    plt.close()
