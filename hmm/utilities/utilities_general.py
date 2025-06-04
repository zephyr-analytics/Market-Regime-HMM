"""
"""
import json
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


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
    Labels HMM states based on mean returns.

    Parameters
    ----------
    training : ModelsTraining
        ModelsTraining instance.
    inferencing : ModelsInferencing
        MondelsInferencing instance.
    Returns
    -------
    state_label_dict : dict
        Dictionary containing states and state labels for an asset.
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
                # Single ticker fallback
                series = data['Adj Close'].dropna()
            price_data[ticker] = series
        except Exception:
            continue  # skip ticker if any issue arises

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
