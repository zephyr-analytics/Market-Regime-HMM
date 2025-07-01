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

    def __init__(self, clustering: PortfolioClustering, config: dict):
        self.risk_lookback = clustering.risk_lookback
        self.clusters = clustering.clusters
        self.forecast_data = clustering.forecast_data
        self.price_data = clustering.price_data.copy()
        self.sma_lookback = clustering.moving_average
        self.max_assets_per_cluster = clustering.max_assets_per_cluster
        self.config = config

    def process(self) -> dict:
        """
        Executes the full portfolio construction pipeline:
        clustering → risk parity → sentiment-based selection → normalization.

        Returns
        -------
        dict
            Dictionary of final normalized portfolio weights keyed by ticker.
        """
        cluster_returns = self.compute_cluster_returns(
            self.price_data, self.clusters, self.risk_lookback
        )

        top_level_weights = self.compute_risk_parity_weights(
            self._risk_parity_weights, cluster_returns, self.risk_lookback
        )

        sentiment_weights, orphaned = self.compute_sentiment_weights(
            self.clusters,
            top_level_weights,
            self.forecast_data,
            self.config,
            self.max_assets_per_cluster
        )

        final_weights = self.normalize_weights(sentiment_weights, orphaned)

        return final_weights

    @staticmethod
    def compute_cluster_returns(price_data: pd.DataFrame, clusters: dict, risk_lookback: int) -> dict:
        """
        Computes synthetic equal-weighted returns for each cluster.

        Parameters
        ----------
        price_data : pd.DataFrame
            DataFrame of asset prices indexed by date and columns as tickers.
        clusters : dict
            Mapping of ticker to cluster ID.
        risk_lookback : int
            Number of recent observations used for computing return history.

        Returns
        -------
        dict
            Dictionary mapping cluster IDs to return series.
        """
        cluster_returns = {}
        cluster_assets = defaultdict(list)

        for tkr, cid in clusters.items():
            cluster_assets[cid].append(tkr)

        for cluster_id, tickers in cluster_assets.items():
            returns = price_data[tickers].pct_change().dropna().tail(risk_lookback)
            if returns.empty:
                continue
            equal_weights = np.ones(len(tickers)) / len(tickers)
            cluster_ret_series = returns @ pd.Series(equal_weights, index=returns.columns)
            cluster_returns[cluster_id] = cluster_ret_series

        return cluster_returns

    @staticmethod
    def compute_risk_parity_weights(risk_parity_fn, cluster_returns: dict, risk_lookback: int) -> dict:
        """
        Computes risk-parity weights across clusters using the provided optimization function.

        Parameters
        ----------
        risk_parity_fn : callable
            Function that implements risk parity optimization.
        cluster_returns : dict
            Dictionary of cluster return series.
        risk_lookback : int
            Number of observations to consider in the covariance matrix.

        Returns
        -------
        dict
            Dictionary of cluster-level weights.
        """
        if not cluster_returns:
            return {"SHV": 1.0}

        cluster_ret_df = pd.DataFrame(cluster_returns)
        cluster_ids = list(cluster_returns.keys())

        weights = risk_parity_fn(
            tickers=cluster_ids,
            price_data=cluster_ret_df,
            lookback=risk_lookback
        )
        return weights

    @staticmethod
    def compute_sentiment_weights(
        clusters: dict,
        top_level_weights: dict,
        forecast_data: dict,
        config: dict,
        max_assets_per_cluster: int
    ) -> tuple[dict, float]:
        """
        Selects top sentiment-weighted tickers per cluster and allocates weights accordingly.

        Parameters
        ----------
        clusters : dict
            Mapping of tickers to cluster IDs.
        top_level_weights : dict
            Dictionary of cluster-level weights from risk parity.
        forecast_data : dict
            Dictionary of sentiment forecasts per ticker (expects "Bullish" values).
        config : dict
            Configuration dictionary, may include `exclusive_pairs`.
        max_assets_per_cluster : int
            Maximum number of assets to select within each cluster.

        Returns
        -------
        tuple of dict and float
            Dictionary of ticker weights and total orphaned weight.
        """
        sentiment_weights = defaultdict(float)
        orphaned_weight = 0.0
        mutually_exclusive_pairs = config.get("exclusive_pairs", [])

        for cluster_id in top_level_weights:
            tickers = [tkr for tkr, cid in clusters.items() if cid == cluster_id]

            contributions = {}
            for tkr in tickers:
                forecast = forecast_data.get(tkr)
                if isinstance(forecast, np.ndarray):
                    forecast = forecast.item()
                if isinstance(forecast, dict):
                    bullish = forecast.get("Bullish", 0.0)
                    if bullish > 0.0:
                        contributions[tkr] = bullish

            if not contributions:
                orphaned_weight += top_level_weights.get(cluster_id, 0.0)
                continue

            to_drop = set()
            for tkr1, tkr2 in mutually_exclusive_pairs:
                if tkr1 in contributions and tkr2 in contributions:
                    to_drop.add(tkr2 if contributions[tkr1] >= contributions[tkr2] else tkr1)

            filtered = {
                tkr: w for tkr, w in contributions.items() if tkr not in to_drop
            }

            top_tickers = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:max_assets_per_cluster]
            if not top_tickers:
                orphaned_weight += top_level_weights.get(cluster_id, 0.0)
                continue

            total_bullish = sum(w for _, w in top_tickers)
            cluster_weight = top_level_weights.get(cluster_id, 0.0)
            if total_bullish > 0:
                for tkr, bullish_weight in top_tickers:
                    norm_weight = bullish_weight / total_bullish
                    sentiment_weights[tkr] += norm_weight * cluster_weight
            else:
                orphaned_weight += cluster_weight

        return sentiment_weights, orphaned_weight

    @staticmethod
    def normalize_weights(sentiment_weights: dict, orphaned_weight: float) -> dict:
        """
        Normalizes final weights and assigns unused (orphaned) allocation to SHV.

        Parameters
        ----------
        sentiment_weights : dict
            Dictionary of ticker-level sentiment weights.
        orphaned_weight : float
            Weight that could not be assigned due to filtering or missing forecasts.

        Returns
        -------
        dict
            Dictionary of final normalized weights including SHV fallback.
        """
        if orphaned_weight > 0:
            sentiment_weights["SHV"] += orphaned_weight

        if not sentiment_weights:
            return {"SHV": 1.0}

        total = sum(sentiment_weights.values())
        normalized_weights = {tkr: w / total for tkr, w in sentiment_weights.items()}
        return normalized_weights

    @staticmethod
    def _risk_parity_weights(tickers: list, price_data: pd.DataFrame, lookback: int) -> dict:
        """
        Computes risk-parity weights for a set of tickers using covariance minimization.

        Parameters
        ----------
        tickers : list of str
            List of asset or cluster identifiers.
        price_data : pd.DataFrame
            DataFrame containing price series or return series.
        lookback : int
            Number of observations to use for return estimation.

        Returns
        -------
        dict
            Dictionary of normalized weights with fallback to SHV where applicable.
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
            rc = w * (cov @ w)
            return np.sum((rc - b * sigma) ** 2)

        init = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=constraints)

        optimized = dict(zip(tickers, result.x))
        scaled = {t: w * (1 - shv_weight) for t, w in optimized.items()}
        scaled["SHV"] = shv_weight

        return scaled
