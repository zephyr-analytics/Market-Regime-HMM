"""
"""

import os

import pandas as pd
from datetime import datetime, timedelta
from pandas_datareader import data as web

import hmm.utilities as utilities


class DataProcessor:
    """
    Handles data processing tasks including pulling historical market data and appending FRED short rates.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Configuration dictionary including keys:
            - 'tickers': list of tickers or single ticker (str)
            - 'data_file_path': path to the CSV file
            - 'start_date': start date in 'YYYY-MM-DD' format
        """
        self.config = config


    def process(self):
        """
        Processes historical price data and appends FRED short rate data ("DFF").

        Returns
        -------
        pd.DataFrame
            Combined dataset of market prices and interest rates.
        """
        data = self.pull_data(
            tickers=self.config["tickers"],
            file_path=self.config["data_file_path"],
            start_date=self.config["start_date"]
        )

        start_date = self.config["start_date"]
        end_date = self.get_latest_trading_day()
        fred_data = self.load_fred_rate("DFF", start_date, end_date)
        data = data.join(fred_data, how="left")

        return data


    def get_latest_trading_day(self):
        """
        Gets the most recent trading day (adjusting for weekends).

        Returns
        -------
        str
            Date in 'YYYY-MM-DD' format.
        """
        today = datetime.today()
        if today.weekday() == 5:  # Saturday
            today -= timedelta(days=1)
        elif today.weekday() == 6:  # Sunday
            today -= timedelta(days=2)
        return today.strftime('%Y-%m-%d')


    def pull_data(self, tickers, file_path, start_date):
        """
        Pulls historical price data for tickers. If a file exists, verifies the tickers and date range.
        If any condition fails, pulls fresh data using `load_price_data()` and saves to file.

        Parameters
        ----------
        tickers : list[str]
            Tickers to load.
        file_path : str
            CSV file path to load or save data.
        start_date : str
            Start date in 'YYYY-MM-DD' format.

        Returns
        -------
        pd.DataFrame
            DataFrame of adjusted close prices.
        """
        tickers = [tickers] if isinstance(tickers, str) else tickers
        today = self.get_latest_trading_day()

        def refresh_data():
            price_data = utilities.load_price_data(tickers, start_date, today)
            df = pd.DataFrame(price_data)
            df.to_csv(file_path)
            return df

        if not os.path.exists(file_path):
            return refresh_data()

        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)

            if isinstance(data, pd.Series):
                data = data.to_frame()

            if not all(ticker in data.columns for ticker in tickers):
                return refresh_data()

            if data.index.max().date() < (datetime.today().date() - pd.Timedelta(days=1)):
                return refresh_data()

            if start_date and pd.to_datetime(start_date) < data.index.min():
                return refresh_data()

            return data

        except Exception:
            return refresh_data()


    @staticmethod
    def load_fred_rate(series, start_date, end_date):
        """
        Load a short-term interest rate series from FRED.

        Parameters
        ----------
        series : str
            FRED series code (e.g., 'DFF').
        start_date : str
            Start date for data.
        end_date : str
            End date for data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the FRED rate series with no NaNs or zeros.
        """
        try:
            rate = web.DataReader(series, "fred", start_date, end_date)
            small_value = 1e-6

            # Replace NaNs
            rate.fillna(small_value, inplace=True)

            # Replace exact 0.0 values
            rate[series] = rate[series].replace(0.0, small_value)
            rate.columns = [series]
            return rate
        except Exception as e:
            print(f"Warning: Failed to load FRED series '{series}': {e}")
            return pd.DataFrame()
