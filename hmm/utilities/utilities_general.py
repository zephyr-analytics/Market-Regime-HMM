"""
"""
import json
import os
import pickle
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


def compound_return(series: pd.Series, window: int) -> pd.Series:
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
    pd.Series : Series of rolling compound return over the time horizon.
    """
    daily_returns = series.pct_change().fillna(0) + 1

    return daily_returns.rolling(window).apply(lambda x: x.prod() - 1, raw=True)


def load_config(global_macro, gloabl_stocks, us_macro):
    """
    Method to load the config file.
    """
    if global_macro:
        config_path = os.path.join(os.getcwd(), "configs", "global_macro_config.json")
    elif gloabl_stocks:
        config_path = os.path.join(os.getcwd(), "configs", "global_stock_config.json")
    elif us_macro:
        config_path = os.path.join(os.getcwd(), "configs", "us_macro_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


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


def load_price_data(tickers: list, start_date: str, end_date: str) -> dict:
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


def save_html(fig, filename):
    """
    Save the HTML file to the 'artifacts' directory within the current working directory.

    Parameters:
    fig : plotly.graph_objects.Figure
        The figure object to save as an HTML file.
    filename : str
        The name of the HTML file.
    weights_filename : str
        The name of the directory for weights.
    processing_type : str
        The type of processing to include in the file path.
    """
    current_directory = os.getcwd()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    artifacts_directory = os.path.join(
        current_directory, "artifacts"
    )
    os.makedirs(artifacts_directory, exist_ok=True)

    file_path = os.path.join(artifacts_directory, f"{timestamp}_{filename}.html")
    fig.write_html(file_path)


def save_plot(filename: str, plot_type: str, plot_sub_folder: str):
    """
    Method to persist png plots.

    Parameters
    ----------
    filename : str
        String representing the desired file name.
    plot_type : str
        String representing the type of plot.
    plot_sub_folder : str
        String representing the sub-directory for storage. 
    """
    directory = os.path.join(os.getcwd(), "hmm", plot_sub_folder, "artifacts", plot_type)
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    plt.savefig(filepath)
    plt.close()
