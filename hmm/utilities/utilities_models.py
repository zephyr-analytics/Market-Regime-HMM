"""
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore


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


def evaluate_state_stability(
        states, overall_threshold=0.2, window_size=20, flip_threshold=5, flip_window_limit=5
    ) -> dict:
    """
    Evaluates the temporal stability of a sequence of hidden states to assess model quality.
    This function calculates the rate of state transitions and detects regions of frequent switching within a sliding windows.

    Parameters
    ----------
    states : np.ndarray
        A sequence of inferred hidden states (e.g., from an HMM) over time.
    overall_threshold : float
        The maximum acceptable overall transition rate (fraction of time steps where state changes),
        above which the sequence is considered unstable. Default is 0.2.
    window_size : int
        The number of time steps in each sliding window used to assess local instability. Default is 20.
    flip_threshold : int
        The minimum number of state changes within a window to count it as a flip-flop window. Default is 5.
    flip_window_limit : int
        The maximum allowable number of flip-flop windows before flagging instability. Default is 5.

    Returns
    -------
    dict : 
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
