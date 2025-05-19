"""
"""
import joblib
import os

import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM


import utilities
from hmm.results.results_processor import ResultsProcessor


class ModelsTrainingProcessor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.n_states = 3
        self.train_states = None
        self.test_states = None
        self.latest_state = None
        self.bullish_state = None
        self.bearish_state = None

    def process(self):
        data = self._load_data()
        features, train_data, test_data = self.prepare_data(training_data=data)
        model, train_states = self._fit_model(train_data=train_data)
        self._save_model(model=model)
        results = ResultsProcessor(
            ticker=self.ticker, model=model, train_states=train_states, training_data=train_data, start_date=self.start_date, end_date=self.end_date
        )
        results.process()

    def _load_data(self):
        """
        """
        adj_close = yf.download(
            tickers=self.ticker,
            start=self.start_date,
            end=self.end_date,
            group_by='ticker',
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if adj_close is None or adj_close.empty:
            print(f"[{self.ticker}] No data downloaded.")
            self.data = None
            return

        if isinstance(adj_close.columns, pd.MultiIndex):
            series = adj_close[self.ticker]["Adj Close"].dropna()
        else:
            if "Adj Close" not in adj_close:
                print(f"[{self.ticker}] Missing 'Adj Close' in data.")
                self.data = None
                return
            series = adj_close["Adj Close"].dropna()

        return series

    def prepare_data(self, training_data):
        """
        """
        ret_3m = utilities.compounded_return(training_data, 63)
        ret_6m = utilities.compounded_return(training_data, 126)
        ret_9m = utilities.compounded_return(training_data, 189)
        ret_12m = utilities.compounded_return(training_data, 252)

        momentum = (ret_3m + ret_6m + ret_9m + ret_12m) / 4

        rolling_vol_1m = training_data.pct_change().rolling(window=21).std()
        rolling_vol_3m = training_data.pct_change().rolling(window=63).std()
        vol_concat = pd.concat([rolling_vol_1m, rolling_vol_3m], axis=1)
        min_vol = vol_concat.min().min()
        max_vol = vol_concat.max().max()
        scaled_vol_1m = 1 - (rolling_vol_1m - min_vol) / (max_vol - min_vol)
        scaled_vol_3m = 1 - (rolling_vol_3m - min_vol) / (max_vol - min_vol)
        mean_scaled_vol = (scaled_vol_1m + scaled_vol_3m) / 2

        features = pd.concat([momentum, mean_scaled_vol], axis=1).dropna()
        features.columns = ['Momentum', 'Volatility']

        split_index = int(len(features) * 0.7)

        train_data = features.iloc[:split_index]
        test_data = features.iloc[split_index:]

        return features, train_data, test_data

    def _fit_model(self, train_data):
        """
        """
        model = GaussianHMM(n_components=self.n_states, covariance_type="diag", tol=0.0001, n_iter=10000)
        model.fit(train_data[['Momentum', 'Volatility']].values)
        train_states = utilities.smooth_states(model.predict(train_data[['Momentum', 'Volatility']].values))

        return model, train_states

    def _save_model(self, model):
        """
        """
        models_path = os.path.join(os.getcwd(), "artifacts", "models")
        os.makedirs(models_path, exist_ok=True)
        model_path = os.path.join(models_path, f"{self.ticker}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
