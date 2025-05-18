import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import os
import scipy.stats as stats

import utilities_general as utilities
from results_processor import ResultsProcessor


class ModelsTrainingProcessor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.n_states = 2
        self.data = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.train_states = None
        self.test_states = None
        self.latest_state = None
        self.bullish_state = None
        self.bearish_state = None

    def process(self):
        self._load_data()
        if self.data is None or self.data.empty:
            print(f"[{self.ticker}] Skipping due to empty data.")
            return
        self._fit_model()
        self._label_states()
        self.latest_state = self.test_states[-1]
        results = ResultsProcessor()
        results.process()

    def _load_data(self):
        adj_close = yf.download(
            tickers=self.ticker,
            start=self.start_date,
            end=self.end_date,
            group_by='ticker',
            auto_adjust=False,
            progress=False,
            threads=True,
            interval="1d"
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

        ret_3m = utilities.compounded_return(series, 63)
        ret_6m = utilities.compounded_return(series, 126)
        ret_9m = utilities.compounded_return(series, 189)
        ret_12m = utilities.compounded_return(series, 252)

        momentum = (ret_3m + ret_6m + ret_9m + ret_12m) / 4

        rolling_vol_1m = series.pct_change().rolling(window=21).std()
        rolling_vol_3m = series.pct_change().rolling(window=63).std()
        vol_concat = pd.concat([rolling_vol_1m, rolling_vol_3m], axis=1)

        # Find the global min and max across both windows
        min_vol = vol_concat.min().min()
        max_vol = vol_concat.max().max()

        # Apply inverted Min-Max scaling to both volatility windows
        scaled_vol_1m = 1 - (rolling_vol_1m - min_vol) / (max_vol - min_vol)
        scaled_vol_3m = 1 - (rolling_vol_3m - min_vol) / (max_vol - min_vol)

        # Average the two scaled volatilities to get a mean score where higher is better (lower volatility)
        mean_scaled_vol = (scaled_vol_1m + scaled_vol_3m) / 2

        features = pd.concat([momentum, mean_scaled_vol], axis=1).dropna()
        features.columns = ['Momentum', 'Volatility']

        split_index = int(len(features) * 0.7)
        self.train_data = features.iloc[:split_index]
        self.test_data = features.iloc[split_index:]
        self.data = features

    def _fit_model(self):
        model = GaussianHMM(n_components=self.n_states, covariance_type="diag", tol=0.0001, n_iter=10000)
        model.fit(self.train_data[['Momentum', 'Volatility']].values)
        self.train_states = self._smooth_states(model.predict(self.train_data[['Momentum', 'Volatility']].values))
        self.test_states = self._smooth_states(model.predict(self.test_data[['Momentum', 'Volatility']].values))
        self.model = model
        print(model.__dict__)

    def _smooth_states(self, states, window=5):
        return pd.Series(states).rolling(window, center=True, min_periods=1).apply(
            lambda x: stats.mode(x)[0][0], raw=False
        ).astype(int).values

    def _label_states(self):
        summary_stats = []
        for state in range(self.n_states):
            idx = self.train_states == state
            mean_momentum = self.train_data['Momentum'].iloc[idx].mean()
            mean_volatility = self.train_data['Volatility'].iloc[idx].mean()

            score = 0
            score += 1 if mean_momentum > 0 else -1
            score += 1 if mean_volatility < self.train_data['Volatility'].mean() else -1

            summary_stats.append({'state': state, 'score': score})

        sorted_states = sorted(summary_stats, key=lambda x: x['score'], reverse=False)

        self.bullish_state = sorted_states[0]['state']
        self.bearish_state = sorted_states[-1]['state']

        print(f"Bullish State: {self.bullish_state}, Bearish State: {self.bearish_state}")

    def forecast_state_distribution(self, n_steps=21):
        self.state_probs = self.model.predict_proba(self.test_data)
        current_state_prob = self.state_probs[-1]
        state_dist = current_state_prob.copy()

        # Step forward n_steps using the transition matrix
        for _ in range(n_steps):
            state_dist = state_dist @ self.model.transmat_

        print(f"Forecasted state distribution for {self.ticker}: {self.latest_state}: {state_dist}")
        return state_dist

