"""
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore


def calculate_portfolio_return(
    portfolio: dict, data: pd.DataFrame, start_date: str, end_date: str
):
    """
    Calculate portfolio return with same-day stop-loss logic and trade outcome counts.

    Parameters
    ----------
    portfolio : dict
        Dictionary of ticker keys and weight values.
    data : pd.DataFrame
        DataFrame of price data with datetime index.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    portfolio_return : float
        Weighted portfolio return with adjusted asset-level exits.
    trade_counts : dict
        Dictionary with counts of positive and negative trades.
    """
    if not portfolio:
        return 0.0, {'positive': 0, 'negative': 0}

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    tickers = list(portfolio.keys())
    weights = pd.Series(portfolio)

    price_df = data[tickers].copy()
    available_dates = price_df.index

    start_date = available_dates[available_dates >= start_date.strftime("%Y-%m-%d")].min()
    end_date = available_dates[available_dates <= end_date.strftime("%Y-%m-%d")].max()

    price_df = price_df.loc[start_date:end_date]

    adjusted_returns = {}
    for ticker in tickers:
        prices = price_df[ticker].dropna()
        adjusted_returns[ticker] = five_percent_drop_rule(prices)

    asset_returns = pd.Series(adjusted_returns)
    portfolio_return = (asset_returns * weights).sum()

    positive_returns = asset_returns[asset_returns > 0]
    negative_returns = asset_returns[asset_returns < 0]

    trade_stats = {
        'positive': int(positive_returns.count()),
        'negative': int(negative_returns.count()),
        'average_gain': positive_returns.mean() if not positive_returns.empty else 0.0,
        'average_loss': negative_returns.mean() if not negative_returns.empty else 0.0
    }

    return portfolio_return, trade_stats


def five_percent_drop_rule(prices: pd.Series, threshold: float = -0.05) -> float:
    """
    Calculate asset return with same-day stop-loss logic.

    If the asset drops more than the threshold on any day, the trade is
    closed at the same day's close.

    Parameters
    ----------
    prices : pd.Series
        Series of asset prices indexed by date.
    threshold : float
        Daily return threshold to trigger stop-loss (e.g., -0.05 for -5%).

    Returns
    -------
    float
        Adjusted return over the holding period (early exit if threshold breached).
    """
    if len(prices) < 2:
        return 0.0

    daily_returns = prices.pct_change().dropna()

    for i in range(len(daily_returns)):
        if daily_returns.iloc[i] <= threshold:
            exit_price = prices.iloc[i + 1] if i + 1 < len(prices) else prices.iloc[i]
            entry_price = prices.iloc[0]

            return (exit_price - entry_price) / entry_price

    return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]


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
