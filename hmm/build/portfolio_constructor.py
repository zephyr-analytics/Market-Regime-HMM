"""
Module for handling final portfolio construction.
"""

from collections import defaultdict
import numpy as np
import pandas as pd

from scipy.optimize import minimize

from hmm.build.portfolio_clustering import PortfolioClustering


class PortfolioConstructor:
    """
    Class for constructing a final portfolio using sentiment forecasts.
    """
    def __init__(self, clustering: PortfolioClustering):
        self.risk_lookback = clustering.risk_lookback
        self.clusters = clustering.clusters
        self.forecast_data = clustering.forecast_data
        self.price_data = clustering.price_data.copy()
        self.sma_lookback = clustering.moving_average
        self.max_assets_per_cluster = clustering.max_assets_per_cluster


    def process(self) -> dict:
        """
        Method for processing through the pipeline.

        Returns
        -------
        normalized_weights : dict
            Dictionary containing tickers as keys and weights as values.
        """
        ticker_weights = defaultdict(float)
        orphaned_weight = 0.0

        # TODO this needs to be adjusted as raw returns lack the explanitory power and story telling
        # of the asset as well as the market.
        # NOTE this should be risk free rate - compounded return over the time period.
        # Step 1: Compute Synthetic Cluster Returns Using All Assets
        cluster_returns = {}
        cluster_assets = defaultdict(list)

        for tkr, cid in self.clusters.items():
            cluster_assets[cid].append(tkr)

        for cluster_id, tickers in cluster_assets.items():
            # TODO this needs to be compounded returns.
            returns = self.price_data[tickers].pct_change().dropna().tail(self.risk_lookback)
            if returns.empty:
                continue
            equal_weights = np.ones(len(tickers)) / len(tickers)
            cluster_ret_series = returns @ pd.Series(equal_weights, index=returns.columns)
            cluster_returns[cluster_id] = cluster_ret_series

        if not cluster_returns:
            return {"SHV": 1.0}

        # Step 2: Risk Parity Between Clusters
        # TODO this needs to be more explicitly handled.
        cluster_ret_df = pd.DataFrame(cluster_returns)
        cluster_cov = cluster_ret_df.cov().values
        cluster_ids = list(cluster_returns.keys())

        top_level_weights = self._risk_parity_weights(
            tickers=cluster_ids,
            price_data=cluster_ret_df,
            lookback=self.risk_lookback
        )

        # Step 3: For Each Cluster, Select Top-N by Sentiment and Apply Risk Parity
        sentiment_weights = defaultdict(float)

        for cluster_id in cluster_ids:
            tickers = [tkr for tkr, cid in self.clusters.items() if cid == cluster_id]
            contributions = {}

            for tkr in tickers:
                forecast = self.forecast_data.get(tkr)
                if isinstance(forecast, np.ndarray):
                    forecast = forecast.item()
                if not isinstance(forecast, dict):
                    continue

                bullish = forecast.get("Bullish", 0.0)
                if bullish > 0.0:
                    contributions[tkr] = bullish

# TODO this logic needs to be split based on cluster weighting and within cluster weighting. 
# Some of the logic is a a bit hard to follow as parameters are being set.
            if not contributions:
                orphaned_weight += top_level_weights.get(cluster_id, 0.0)
                continue

            top_tickers = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:self.max_assets_per_cluster]
            selected_tickers = [tkr for tkr, _ in top_tickers]

            rp_weights = self._risk_parity_weights(
                tickers=selected_tickers,
                price_data=self.price_data,
                lookback=self.risk_lookback
            )

            cluster_weight = top_level_weights.get(cluster_id, 0.0)
            for tkr, w in rp_weights.items():
                sentiment_weights[tkr] += w * cluster_weight

        # Step 4: Assign orphaned weight to SHV
        if orphaned_weight > 0:
            sentiment_weights["SHV"] += orphaned_weight

        if not sentiment_weights:
            return {"SHV": 1.0}

        # Step 5: Final normalization
        total = sum(sentiment_weights.values())
        normalized_weights = {tkr: w / total for tkr, w in sentiment_weights.items()}

        return normalized_weights

# TODO this needs to become a utilities method.
    @staticmethod
    def _risk_parity_weights(tickers: list, price_data: pd.DataFrame, lookback: int) -> dict:
        """
        Method to handle weighting clusters and assets by risk parity.

        Parameters
        ----------

        Returns
        -------
        """
        tickers = tickers.copy()

        if len(tickers) == 1:
            return {tickers[0]: 1.0}

        shv_weight = 0.0
        if "SHV" in tickers:
            tickers.remove("SHV")
            shv_weight = 1.0

        if not tickers:
            return {"SHV": 1.0}

        if set(tickers).issubset(price_data.index):
            returns = price_data.loc[tickers]
            returns = returns.to_frame().T if isinstance(returns, pd.Series) else returns.T
        else:
            returns = price_data[tickers].pct_change().dropna().tail(lookback)

        cov = returns.cov().values if isinstance(returns, pd.DataFrame) else np.array([[returns.var()]])

        n = len(tickers)
        b = np.ones(n) / n  # equal risk budget

        def objective(w):
            port_var = w @ cov @ w
            sigma = np.sqrt(port_var)
            rc = w * (cov @ w)  # risk contributions
            return np.sum((rc - b * sigma) ** 2)

        init = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=constraints)

        optimized = dict(zip(tickers, result.x))
        scaled = {t: w * (1 - shv_weight) for t, w in optimized.items()}
        scaled["SHV"] = shv_weight

        return scaled
