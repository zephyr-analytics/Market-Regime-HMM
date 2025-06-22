"""
Module for handling final portfolio construction.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from hmm import utilities


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
        Method to process the PortfolioConstructor.

        Returns
        -------
        normalized_weights : dict
            Dictionary of final portfolio weights.
        """
        # TODO if possible it might be better to not use compounded returns for the risk component.
        # It might be best to go back to using a raw series of returns. 
        cluster_assets = self._group_assets_by_cluster(clusters=self.clusters)
        cluster_returns = self._compute_cluster_returns(
            cluster_assets=cluster_assets, price_data=self.price_data, lookback=self.config["risk_lookback"]
        )

        if not cluster_returns:
            return {"SHV": 1.0}

        cluster_ret_df = pd.DataFrame([cluster_returns])
        cluster_ids = list(cluster_returns.keys())

        cluster_weights = self._risk_parity_weights(
            tickers=cluster_ids,
            price_data=cluster_ret_df,
            lookback=self.config["risk_lookback"]
        )

        sentiment_weights, orphaned_weight = self._compute_sentiment_weights(
            cluster_ids=cluster_ids, cluster_weights=cluster_weights, clusters=self.clusters, 
            forecast_data=self.forecast_data, price_data=self.price_data, lookback=self.config["risk_lookback"], 
            max_assets=self.max_assets_per_cluster
        )

        if orphaned_weight > 0:
            sentiment_weights["SHV"] += orphaned_weight

        if not sentiment_weights:
            return {"SHV": 1.0}

        total = sum(sentiment_weights.values())
        portfolio_weights = {tkr: w / total for tkr, w in sentiment_weights.items()}

        return portfolio_weights


    @staticmethod
    def _group_assets_by_cluster(clusters: dict):
        """
        Method to group assets into clusters.

        Parameters
        ----------
        clusters : dict
            Dictionary containing clusters.

        Returns
        -------
        cluster_assets : dict
            Dictionary with clusters as keys and list of assets as values.
        """
        cluster_assets = defaultdict(list)
        for tkr, cid in clusters.items():
            cluster_assets[cid].append(tkr)

        return cluster_assets


    @staticmethod
    def _compute_cluster_returns(cluster_assets: dict, price_data: pd.DataFrame, lookback: int):
        """
        Method to risk parity clusters.

        Parameters
        ----------
        cluster_assets : dict

        price_data : pd.DataFrame

        lookback : int


        Returns
        -------
        cluster_returns : dict

        """
        cluster_returns = {}
        for cluster_id, tickers in cluster_assets.items():
            returns = price_data[tickers].pct_change().dropna().tail(lookback)
            compounded = utilities.compound_returns(returns=returns)
            if compounded.empty:
                continue
            weights = PortfolioConstructor._risk_parity_weights(
                tickers=tickers,
                price_data=price_data,
                lookback=lookback
            )
            weight_vector = pd.Series([weights[tkr] for tkr in tickers], index=tickers)
            cluster_ret_series = compounded @ weight_vector
            cluster_returns[cluster_id] = cluster_ret_series
        print(cluster_returns)
        return cluster_returns


    @staticmethod
    def _compute_sentiment_weights(
        cluster_ids: list, cluster_weights: dict, clusters: dict, forecast_data: dict,
        price_data: pd.DataFrame, lookback: int, max_assets: int
    ):
        """

        Parameters
        ----------
        cluster_ids : list
            List of clusters.
        cluster_weights : dict
            Dictionary of cluster weights.
        clusters : dict
            Dictionary of cluster data.
        forecast_data : dict
            Dictionary of forward propagated probabilites per asset.
        price_data : pd.Dataframe
            Dataframe of price data for constructing returns.
        lookback : int
            Integer representing the length in trading days for risk parity.
        max_assets : int
            Integer representing the maximum assets to select per cluster.

        Returns
        -------
        """
        sentiment_weights = defaultdict(float)
        orphaned_weight = 0.0

        for cluster_id in cluster_ids:
            tickers = [tkr for tkr, cid in clusters.items() if cid == cluster_id]
            contributions = {}

            for tkr in tickers:
                forecast = forecast_data.get(tkr)
                if isinstance(forecast, np.ndarray):
                    forecast = forecast.item()
                if not isinstance(forecast, dict):
                    continue

                bullish = forecast.get("Bullish", 0.0)

                if bullish > 0:
                    contributions[tkr] = bullish

            if not contributions:
                orphaned_weight += cluster_weights.get(cluster_id, 0.0)
                continue

            top_tickers = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:max_assets]
            selected_tickers = [tkr for tkr, _ in top_tickers]

            rp_weights = PortfolioConstructor._risk_parity_weights(
                tickers=selected_tickers,
                price_data=price_data,
                lookback=lookback
            )

            cluster_weight = cluster_weights.get(cluster_id, 0.0)
            for tkr, w in rp_weights.items():
                sentiment_weights[tkr] += w * cluster_weight

        return sentiment_weights, orphaned_weight


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
