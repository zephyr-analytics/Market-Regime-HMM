"""
"""

import os

import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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
        Load market data along with FRED economic indicators over the specified time frame.
        Adds the following FRED series:
        - 'DFF' (short rate, daily)
        - 'A191RP1Q027SBEA' (quarterly, GDP-related)
        - 'CORESTICKM679SFRBATL' (monthly, core CPI-related)

        Returns
        -------
        pd.DataFrame
            DataFrame with market and FRED series aligned.
        """
        start_date = self.config["current_start"]
        end_date = self.get_latest_trading_day()

        # market tickers
        tickers = self.config["tickers"] + [self.config["Cash_Ticker"]]
        data = self.pull_data(
            tickers=tickers,
            file_path=self.config["data_file_path"],
            start_date=start_date
        )

        # add short rates (daily)
        fred_dff = self.load_short_rates(
            series="DFF",
            start_date=start_date,
            end_date=end_date
        )
        data = data.join(fred_dff, how="left")

        # add quarterly GDP-related series
        fred_gdp = self.load_fred_series(
            series="A191RP1Q027SBEA",
            freq="Q",
            start_date=start_date,
            end_date=end_date
        )
        data = data.join(fred_gdp, how="left")

        # add monthly core CPI-related series
        fred_core_cpi = self.load_fred_series(
            series="CORESTICKM679SFRBATL",
            freq="M",
            start_date=start_date,
            end_date=end_date
        )
        data = data.join(fred_core_cpi, how="left")

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
        today = self.config["end_date"]
        # today = self.get_latest_trading_day()

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
    def load_short_rates(series, start_date, end_date):
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

    def load_fred_series(self, series: str, freq: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load a FRED time series and align it to daily dates.

        Parameters
        ----------
        series : str
            FRED series code.
        freq : str
            Frequency of the FRED series: 'D', 'M', or 'Q'.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.

        Returns
        -------
        pd.DataFrame
            DataFrame with the series aligned to daily index, forward-filled.
        """
        import pandas_datareader.data as web
        import pandas as pd

        # load raw series
        s = web.DataReader(series, "fred", start=start_date, end=end_date)

        # reindex to daily and forward-fill
        daily_index = pd.date_range(start=start_date, end=end_date, freq="B")
        s = s.reindex(daily_index, method="ffill")
        s.columns = [series]
        return s
