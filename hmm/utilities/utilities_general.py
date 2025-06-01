"""
"""
import json
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def compounded_return(series, window):
    """

    Parameters
    ----------
    series : 

    window : 

    Returns
    -------

    """
    daily_returns = series.pct_change().fillna(0) + 1

    return daily_returns.rolling(window).apply(lambda x: x.prod() - 1, raw=True)


def evaluate_state_stability(states, overall_threshold=0.2, window_size=20, flip_threshold=5, flip_window_limit=5) -> dict:
    """

    Parameters
    ----------
    states : 

    overall_threshold : 

    window_size : 

    flip_threshold : 

    flip_window_limit : 

    Returns
    -------
    dict : 
    """
    if len(states) < window_size + 1:
        return {
            "transition_rate": None,
            "flip_flop_windows": None,
            "is_unstable": True,
            "reason": "Not enough data to evaluate stability."
        }

    num_transitions = sum(states[i] != states[i + 1] for i in range(len(states) - 1))
    transition_rate = num_transitions / (len(states) - 1)

    flip_flop_windows = 0
    for i in range(len(states) - window_size):
        window = states[i:i + window_size]
        changes = sum(window[j] != window[j + 1] for j in range(len(window) - 1))
        if changes >= flip_threshold:
            flip_flop_windows += 1

    is_unstable = (transition_rate > overall_threshold) or (flip_flop_windows > flip_window_limit)
    reason = []
    if transition_rate > overall_threshold:
        reason.append(f"High transition rate ({transition_rate:.2f})")
    if flip_flop_windows > flip_window_limit:
        reason.append(f"Flip-flopping in {flip_flop_windows} windows")

    return {
        "transition_rate": transition_rate,
        "transitions": flip_flop_windows,
        "is_unstable": is_unstable,
        "reason": "; ".join(reason) if reason else "Stable"
    }


def label_states(training=None, inferencing=None) -> dict:
    """
    Labels HMM states based on mean returns.

    Parameters
    ----------
    training : 

    inferencing : 

    Returns
    -------
    state_label_dict : dict
    
    """
    if training:
        states = training.train_states.copy()
        returns = training.train_data.copy()
    elif inferencing:
        states = inferencing.test_states.copy()
        returns = inferencing.test_data.copy()

    returns = returns['Momentum'].reset_index(drop=True)

    state_returns = []
    for state in np.unique(states):
        state_mask = states == state
        mean_return = returns[state_mask].mean()
        state_returns.append((state, mean_return))

    state_returns.sort(key=lambda x: x[1])

    labels = ['Bearish', 'Neutral', 'Bullish']
    state_label_dict = {
        state: labels[i] for i, (state, _) in enumerate(state_returns)
    }

    return state_label_dict


def load_from_pickle(file_path: str):
    """
    Load the ModelsTraining instance from a pickle file.

    Parameters
    ----------
    file_path : str

    Returns
    -------

    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def persist_to_pickle(file, file_path: str):
    """
    Pickle the ModelsTraining instance to a file.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    
    """
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)


def smooth_states(states, window=21):
    """

    Parameters
    ----------
    states : 

    window : int

    Returns
    -------
    pd.Series : 
    """
    return pd.Series(states).rolling(window, center=True, min_periods=1).apply(
        lambda x: stats.mode(x)[0][0], raw=False
    ).astype(int).values


def save_plot(filename, plot_type, plot_sub_folder):
    """

    Parameters
    ----------
    filename : 

    plot_type :

    plot_sub_folder : 
    
    """
    directory = os.path.join(os.getcwd(), "hmm", plot_sub_folder, "artifacts", plot_type)
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    plt.savefig(filepath)
    plt.close()

def load_config():
    """
    Method to load the config file.
    """
    config_path = os.path.join(os.getcwd(), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)
