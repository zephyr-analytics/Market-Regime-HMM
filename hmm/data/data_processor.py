import os
import pandas as pd
import yfinance as yf
from datetime import datetime

import hmm.utilities as utilities

class DataProcessor:
    """
    Handles configuration-driven data processing tasks, including pulling
    and validating historical market data.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Configuration dictionary including keys:
            - 'tickers': list of tickers or single ticker (str)
            - 'data_file': path to the CSV file
            - 'start_date': (optional) start date in 'YYYY-MM-DD' format
        """
        self.config = config

    def process(self):
        """
        Placeholder for user-defined data processing logic.
        Does not trigger data loading.
        """
        data = self.pull_data(
            tickers=self.config["tickers"],
            file_path=self.config["data_file_path"],
            start_date=self.config["start_date"]
        )
        return data

    @staticmethod
    def pull_data(tickers, file_path, start_date):
        """
        Pulls historical price data for tickers. If a file exists, verifies the tickers and date range.
        If any condition fails, pulls fresh data using `load_price_data()` and saves to file.

        Returns
        -------
        pd.DataFrame
            DataFrame of adjusted close prices.
        """
        tickers = [tickers] if isinstance(tickers, str) else tickers
        today = datetime.today().strftime('%Y-%m-%d')

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

            return data[tickers] if len(tickers) > 1 else data[tickers[0]]

        except Exception:
            return refresh_data()
