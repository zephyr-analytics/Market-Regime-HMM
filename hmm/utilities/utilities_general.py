"""
"""
import json
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore


def compounded_return(series: pd.Series, window: int) -> pd.Series:
    """
    Method to aggregate price data into compound returns over a set horizon.

    Parameters
    ----------
    series : pd.Series
        Series of price data to calculate compounded returns from.
    window : int
        Integer representing the time steps in days to compound over.
    Returns
    -------
    pd.Series : 
    """
    daily_returns = series.pct_change().fillna(0) + 1

    return daily_returns.rolling(window).apply(lambda x: x.prod() - 1, raw=True)


def evaluate_state_stability(states, overall_threshold=0.2, window_size=20, flip_threshold=5, flip_window_limit=5) -> dict:
    """
    Evaluates the temporal stability of a sequence of hidden states to assess model quality.

    This function calculates the rate of state transitions and detects regions of frequent switching
    ("flip-flopping") within sliding windows. It flags instability based on configurable thresholds.

    Parameters
    ----------
    states : np.ndarray
        A sequence of inferred hidden states (e.g., from an HMM) over time.
    overall_threshold : float, optional
        The maximum acceptable overall transition rate (fraction of time steps where state changes),
        above which the sequence is considered unstable. Default is 0.2.
    window_size : int, optional
        The number of time steps in each sliding window used to assess local instability. Default is 20.
    flip_threshold : int, optional
        The minimum number of state changes within a window to count it as a flip-flop window. Default is 5.
    flip_window_limit : int, optional
        The maximum allowable number of flip-flop windows before flagging instability. Default is 5.

    Returns
    -------
    dict
        A dictionary containing:
        - "transition_rate" (float or None): Fraction of transitions across the full sequence.
        - "transitions" (int or None): Number of windows with excessive state changes.
        - "is_unstable" (bool): Whether the state sequence is considered unstable.
        - "reason" (str): Explanation for instability, or "Stable" if none detected.
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
    Assigns labels 'Bearish', 'Neutral', 'Bullish' to HMM state IDs based on z-score of returns.
    Ensures all three labels are present by assigning duplicate labels to existing states if needed.

    Parameters
    ----------
    training : ModelsTraining
    inferencing : ModelsInferencing

    Returns
    -------
    label_map : dict
        Dictionary with keys = HMM state IDs and values = labels.
    """
    if training:
        states = training.train_states.copy()
        returns = training.train_data.copy()
    elif inferencing:
        states = inferencing.test_states.copy()
        returns = inferencing.test_data.copy()
    else:
        raise ValueError("Either training or inferencing must be provided.")

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

    n_states = len(sorted_states)

    if n_states >= 3:
        for i, label in enumerate(label_order):
            state = sorted_states[i]
            label_map[state] = label
    elif n_states == 2:
        label_map[sorted_states[0]] = 'Bearish'
        label_map[sorted_states[1]] = 'Bullish'
        if sorted_states[1] not in label_map:
            label_map[sorted_states[1]] = 'Neutral'
        else:
            label_map[sorted_states[1]] = 'Bullish'
            label_map[sorted_states[0]] = 'Neutral'
    elif n_states == 1:
        only_state = sorted_states[0]
        label_map[only_state] = 'Bearish'

        label_map = {
            only_state: 'Bearish'
        }

        label_coverage = {
            'Bearish': only_state,
            'Neutral': only_state,
            'Bullish': only_state
        }

        label_map = {v: k for k, v in label_coverage.items()}

    return label_map


def load_from_pickle(file_path: str):
    """
    Loads a pickled persist instance from a pickle file.

    Parameters
    ----------
    file_path : str
        String representing the file path to load a pickle file from.
    Returns
    -------
    object : Deserialized object that was persisted.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_price_data(tickers: list[str], start_date: str, end_date: str) -> dict:
    """
    Loads adjusted close price data for multiple tickers using yfinance.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    dict
        Dictionary mapping ticker to its pd.Series of adjusted close prices.
    """
    import yfinance as yf
    import pandas as pd

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=False,
        progress=False,
        threads=True
    )

    price_data = {}

    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                series = data[ticker]['Adj Close'].dropna()
            else:
                series = data['Adj Close'].dropna()
            price_data[ticker] = series
        except Exception:
            continue

    return price_data


def persist_to_pickle(file, file_path: str):
    """
    Pickles the persist instance to a file.

    Parameters
    ----------
    file : object
        Object to be persisted.
    file_path : str
        String representing the file path to persist a pickle file from.    
    """
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)


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
