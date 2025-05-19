"""
"""
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def compounded_return(series, window):
    """
    """
    daily_returns = series.pct_change().fillna(0) + 1
    return daily_returns.rolling(window).apply(lambda x: x.prod() - 1, raw=True)


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


def persist_to_pickle(file, file_path: str):
    """
    Pickle the ModelsTraining instance to a file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)


def load_from_pickle(file_path: str):
    """
    Load the ModelsTraining instance from a pickle file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
