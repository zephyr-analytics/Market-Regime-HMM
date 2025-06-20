"""
Module for handling final portfolio construction.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioConstructor:
    """
    Class for constructing the final portfolio.
    """
    def __init__(
        self, config: dict, clusters: dict, forecast_data: dict, price_data: pd.DataFrame
    ):
        self.config = config
        self.clusters = clusters
        self.forecast_data = forecast_data
        self.price_data = price_data
        self.sma_lookback = config["moving_average"]
        self.max_assets_per_cluster = config["max_assets_per_cluster"]

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
            returns = self.price_data[tickers].pct_change().dropna().tail(self.config["risk_lookback"])
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
            lookback=self.config["risk_lookback"]
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
                if bullish > 0:
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
                lookback=self.config["risk_lookback"]
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
        print(normalized_weights)
        return normalized_weights

    @staticmethod
    def _risk_parity_weights(tickers: list, price_data: pd.DataFrame, lookback: int) -> dict:
        """
        Method for handling risk parity weighting.

        Parameters
        ----------
        tickers : list
            List of cluster_ids or tickers.
        price_data : pd.Dataframe
            Dataframe utilized for the return property of risk parity.
        lookback : int
            Lookback period used to calculate risk and returns.
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

        returns = price_data[tickers].pct_change().dropna().tail(lookback)
        cov_matrix = returns.cov().values

        if np.allclose(cov_matrix, 0):
            weights = {tkr: 1.0 / len(tickers) for tkr in tickers}
            weights["SHV"] = shv_weight

            return weights

        n = len(tickers)

        def objective(w):
            """
            Method to implement the objective function of risk parity.
            """
            port_vol = np.sqrt(w @ cov_matrix @ w)
            marginal = cov_matrix @ w
            contrib = w * marginal / port_vol

            return np.sum((contrib - np.mean(contrib)) ** 2)

        init = np.ones(n) / n
        bounds = [(0, 1)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        try:
            result = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=constraints)
            if not result.success:
                raise ValueError()
        except Exception:
            weights = {tkr: 1.0 / len(tickers) for tkr in tickers}
            weights["SHV"] = shv_weight

            return weights

        optimized = dict(zip(tickers, result.x))
        scaled = {tkr: w * (1 - shv_weight) for tkr, w in optimized.items()}
        scaled["SHV"] = shv_weight

        return scaled
