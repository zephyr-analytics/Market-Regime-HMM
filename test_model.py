import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import os
import scipy.stats as stats


class MarketRegimeHMM:
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
        self._plot_regimes()
        self._plot_price_with_states()
        self._plot_price_path_with_states()  # New addition

    def compounded_return(self, series, window):
        daily_returns = series.pct_change().fillna(0) + 1
        return daily_returns.rolling(window).apply(lambda x: x.prod() - 1, raw=True)

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

        ret_1m = self.compounded_return(series, 21)
        ret_3m = self.compounded_return(series, 63)
        ret_6m = self.compounded_return(series, 126)
        ret_9m = self.compounded_return(series, 189)
        ret_12m = self.compounded_return(series, 252)

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
        model = GaussianHMM(n_components=self.n_states, covariance_type="diag", tol=1, n_iter=10000)
        model.fit(self.train_data[['Momentum', 'Volatility']].values)
        self.train_states = self._smooth_states(model.predict(self.train_data[['Momentum', 'Volatility']].values))
        self.test_states = self._smooth_states(model.predict(self.test_data[['Momentum', 'Volatility']].values))
        self.model = model

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

    def _save_plot(self, filename, plot_type):
        directory = os.path.join("artifacts", plot_type)
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        plt.close()

    def _plot_regimes(self):
        plt.figure(figsize=(15, 5))
        train_idx = self.train_data.index
        for state in np.unique(self.train_states):
            idx = self.train_states == state
            plt.plot(train_idx[idx], self.train_data['Momentum'].iloc[idx], '.', label=f'Train State {state}')

        test_idx = self.test_data.index
        for state in np.unique(self.test_states):
            idx = self.test_states == state
            plt.plot(test_idx[idx], self.test_data['Momentum'].iloc[idx], '.', label=f'Test State {state}')

        plt.legend()
        plt.title(f"Market Regimes for {self.ticker} (Train/Test)")
        self._save_plot(f"{self.ticker}_regime_plot.png", plot_type="regime_plots")

    def _plot_price_with_states(self):
        plt.figure(figsize=(15, 5))
        index_combined = np.concatenate([self.train_data.index, self.test_data.index])
        states_combined = np.concatenate([self.train_states, self.test_states])

        for state in np.unique(states_combined):
            idx = states_combined == state
            plt.plot(index_combined[idx], np.arange(len(index_combined[idx])), '.', label=f'State {state}')

        plt.title(f"Regime States for {self.ticker}")
        plt.ylabel("State Index")
        plt.xlabel("Date")
        plt.legend()
        self._save_plot(f"{self.ticker}_states_plot.png", plot_type="state_index_plots")

    def _plot_price_path_with_states(self):
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

        if isinstance(adj_close.columns, pd.MultiIndex):
            price_series = adj_close[self.ticker]["Adj Close"].dropna()
        else:
            price_series = adj_close["Adj Close"].dropna()

        plt.figure(figsize=(15, 5))
        index_combined = np.concatenate([self.train_data.index, self.test_data.index])
        states_combined = np.concatenate([self.train_states, self.test_states])

        for state in np.unique(states_combined):
            idx = states_combined == state
            dates = index_combined[idx]
            prices = price_series.loc[dates]
            plt.plot(prices.index, prices.values, '.', label=f'State {state}')

        plt.title(f"Price Path with Regimes for {self.ticker}")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend()
        self._save_plot(f"{self.ticker}_price_path.png", plot_type="price_path_plots")


class PortfolioAllocator:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.allocations = {}
        self.cash_weight = 0.0

    def build_portfolio(self):
        for ticker in self.tickers:
            model = MarketRegimeHMM(ticker=ticker, start_date=self.start_date, end_date=self.end_date)
            model.process()
            forecast_probs = model.forecast_state_distribution(n_steps=21)
            total_bullish_prob = forecast_probs[model.bullish_state]

            starting_weight = 0.09
            if total_bullish_prob >= 0.95:
                final_weight = starting_weight
            elif 0.7 <= total_bullish_prob < 0.95:
                final_weight = starting_weight / 2
            else:
                final_weight = 0.0

            self.allocations[ticker] = final_weight

        self._finalize_allocations()
        self._plot_portfolio()

    def _finalize_allocations(self):
        total_allocated = sum(self.allocations.values())
        self.cash_weight = max(0.0, 1.0 - total_allocated)
        self.allocations['CASH'] = self.cash_weight

        print("\nFinal Portfolio Allocation:")
        for ticker, weight in self.allocations.items():
            print(f"  {ticker}: {weight:.2%}")

    def _plot_portfolio(self):
        labels = list(self.allocations.keys())
        sizes = list(self.allocations.values())

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title("Final Portfolio Allocation")
        os.makedirs("artifacts", exist_ok=True)
        plt.savefig("artifacts/final_portfolio_pie_chart.png")
        plt.close()
        print("\nSaved portfolio pie chart to artifacts/final_portfolio_pie_chart.png")


if __name__ == "__main__":
    tickers = ["IUSG", "IUSV", "EFG", "EFV", "VWO", "BND", "BNDX", "EMB", "TLT", "IAU", "DBC"]
    allocator = PortfolioAllocator(tickers=tickers, start_date='2008-01-01', end_date='2025-05-01')
    allocator.build_portfolio()
