"""
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.cluster.hierarchy import fcluster
from scipy.stats import zscore
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def calculate_portfolio_return(
    portfolio: dict, data: pd.DataFrame, start_date: str, end_date: str, threshold: float, use_stop_loss: bool
):
    """
    Calculate portfolio return with same-day stop-loss logic.

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
        Weighted portfolio return with adjusted exits.
    trade_stats : dict
        Dictionary with counts and average returns of trades.
    asset_details : pd.DataFrame
        DataFrame with asset, weight, and adjusted return.
    """
    if not portfolio:
        return 0.0, {'positive': 0, 'negative': 0}, pd.DataFrame(columns=['ticker', 'weight', 'return'])

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
        if prices.empty or len(prices) < 2:
            adjusted_returns[ticker] = 0.0
            continue

        if use_stop_loss:
            adjusted_returns[ticker] = stop_loss(prices, threshold=threshold)
        else:
            # Simple return from start to end
            adjusted_returns[ticker] = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

    asset_returns = pd.Series(adjusted_returns)
    portfolio_return = (asset_returns * weights).sum()

    trade_stats, asset_details = calculate_trade_stats_and_details(asset_returns, weights)

    return portfolio_return, trade_stats, asset_details


def calculate_trade_stats_and_details(
    asset_returns: pd.Series, weights: pd.Series
) -> tuple[dict, pd.DataFrame]:
    """
    Calculate trade statistics and assemble asset detail DataFrame.

    Parameters
    ----------
    asset_returns : pd.Series
        Series of asset returns.
    weights : pd.Series
        Series of asset weights.

    Returns
    -------
    trade_stats : dict
        Dictionary with counts and average returns of trades.
    asset_details : pd.DataFrame
        DataFrame with ticker, weight, and return per asset.
    """
    positive_returns = asset_returns[asset_returns > 0]
    negative_returns = asset_returns[asset_returns < 0]

    trade_stats = {
        'positive': int(positive_returns.count()),
        'negative': int(negative_returns.count()),
        'average_gain': positive_returns.mean() if not positive_returns.empty else 0.0,
        'average_loss': negative_returns.mean() if not negative_returns.empty else 0.0
    }

    asset_details = pd.DataFrame({
        'ticker': asset_returns.index,
        'weight': weights[asset_returns.index],
        'return': asset_returns.values
    }).reset_index(drop=True)

    return trade_stats, asset_details


@staticmethod
def evaluate_clustering_scores(sequences: np.ndarray, linkage_matrix, min_clusters: int, max_clusters: int) -> tuple [list, dict]:
    """
    Evaluate clustering performance metrics across a range of cluster counts.

    Parameters
    ----------
    sequences : np.ndarray
        Array of trimmed state sequences
    linkage_matrix : 
        Matrix of linkage between clusters
    min_clusters : int
        Lower limit of allowed clusters.
    max_clusters : int
        Upper limit of allowed clusters.

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
            ch = calinski_harabasz_score(sequences, labels)
            db = davies_bouldin_score(sequences, labels)
        except ValueError:
            continue

        scores.append([sil, ch, db])
        label_map[k] = labels

    return np.array(scores), label_map


def stop_loss(prices: pd.Series, threshold: float = -0.05) -> float:
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
        # NOTE error out on Index error, request more warm up time.
        label_map[state] = label

    return label_map


def plot_cumulative_returns(all_trade_details, output_dir='cumulative_return_plots'):
    """
    Saves cumulative return plots by ticker as separate PNG files, with Y-axis in percentage.

    Parameters:
    - all_trade_details: List[pd.DataFrame]
        List of DataFrames, each containing trade details.
        Expected columns: 'ticker', 'return', 'trade_window_end'
    - output_dir: str
        Directory to save the individual plots.
    """
    all_trades_df = pd.concat(all_trade_details, ignore_index=True)
    all_trades_df['trade_window_end'] = pd.to_datetime(all_trades_df['trade_window_end'])

    all_trades_df = all_trades_df.sort_values(by='trade_window_end')

    def compound_returns(group):
        group = group.copy()
        group['cum_return'] = (1 + group['return']).cumprod()
        return group

    compounded_df = all_trades_df.groupby('ticker').apply(compound_returns).reset_index(drop=True)

    output_path = os.path.join(os.getcwd(), "artifacts", "Asset Analysis", output_dir)
    os.makedirs(output_path, exist_ok=True)

    for ticker, group in compounded_df.groupby('ticker'):
        plt.figure(figsize=(10, 6))
        plt.plot(group['trade_window_end'], group['cum_return'], label=f'{ticker}')
        plt.title(f'Cumulative Return: {ticker}')
        plt.xlabel("Trade Window End")
        plt.ylabel("Cumulative Return")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'cumulative_return_{ticker}.png'), dpi=300)
        plt.close()
