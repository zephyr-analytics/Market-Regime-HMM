"""
"""

import numpy as np
import pandas as pd
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


def label_states(training) -> dict:
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
    states = training.train_states.copy()
    returns = training.train_data.copy()

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
