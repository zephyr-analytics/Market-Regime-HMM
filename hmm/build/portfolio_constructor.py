"""
Module for handling final portfolio construction.
"""

from collections import defaultdict
import numpy as np
import pandas as pd

from scipy.optimize import minimize


class PortfolioConstructor:
    """
    Class for constructing a final portfolio using sentiment forecasts.
    """
    def __init__(
        self, config: dict, clusters: dict, forecast_data: dict, price_data: dict
    ):
        self.config = config
        self.clusters = clusters
        self.forecast_data = forecast_data
        self.price_data = price_data
        self.sma_lookback = config["moving_average"]
        self.max_assets_per_cluster = config["max_assets_per_cluster"]


    def process(self):
        """
        Constructs the portfolio by filtering and weighting tickers through a
        multi-step process: equal cluster weighting, sentiment-based top asset selection,
        risk-parity weighting within clusters, and final aggregation.
        """
        ticker_weights = defaultdict(float)
        orphaned_weight = 0.0

        # Step 1: Equal cluster weighting
        adjusted_cluster_weights = self._equal_weight_clusters(self.clusters)

        # Step 2: Allocate top-N assets within clusters using sentiment
        sentiment_weights, sentiment_orphaned = self._allocate_assets_within_clusters(
            cluster_weights=adjusted_cluster_weights,
            clusters=self.clusters,
            forecast_data=self.forecast_data,
            top_n=self.max_assets_per_cluster
        )

        if not sentiment_weights:
            return {"SHV": 1.0}

        # Step 3: Filter clusters post-topN
        selected_ticker_set = set(sentiment_weights.keys())
        self.clusters = {
            tkr: cid for tkr, cid in self.clusters.items()
            if tkr in selected_ticker_set
        }

        # Step 4: Apply Risk Parity within clusters (SMA filter removed)
        for cluster_id, cluster_weight in adjusted_cluster_weights.items():
            cluster_tickers = [
                tkr for tkr, cid in self.clusters.items() if cid == cluster_id
            ]
            if not cluster_tickers:
                orphaned_weight += cluster_weight
                continue

            rp_weights = self._risk_parity_weights(
                tickers=cluster_tickers,
                price_data=self.price_data,
                lookback=self.config.get("risk_lookback", 252)
            )

            for tkr, w in rp_weights.items():
                ticker_weights[tkr] += w * cluster_weight

        # Step 5: Reassign orphaned cluster weight to SHV
        orphaned_weight += sentiment_orphaned
        if orphaned_weight > 0:
            ticker_weights["SHV"] += orphaned_weight

        if not ticker_weights:
            return {"SHV": 1.0}

        # Step 6: Final normalization
        total = sum(ticker_weights.values())
        normalized_weights = {tkr: w / total for tkr, w in ticker_weights.items()}
        print(sum(normalized_weights.values()))
        return normalized_weights


    @staticmethod
    def _equal_weight_clusters(
        clusters: dict
    ) -> dict:
        """
        Assigns equal weight to each cluster, adjusted only by the net Bullish sentiment
        of tickers that meet cutoff conditions.

        Args:
            clusters (dict): Mapping of tickers to cluster IDs.
            forecast_data (dict): Forecast sentiment scores for each ticker.
            bullish_cutoff (float): Minimum Bullish score required to include a ticker.
            bearish_cutoff (float): Maximum Bearish score allowed to include a ticker.

        Returns:
            dict: Dictionary with cluster IDs as keys and unnormalized net Bullish sentiment
                as values (equal weight treatment). Caller can normalize afterward.
        """
        unique_clusters = set(clusters.values())
        num_clusters = len(unique_clusters)

        if num_clusters == 0:
            return {}

        equal_weight = 1.0 / num_clusters
        adjusted =  {cluster_id: equal_weight for cluster_id in unique_clusters}
        # print(adjusted)
        return adjusted


    @staticmethod
    def _allocate_assets_within_clusters(
        cluster_weights: dict,
        clusters: dict,
        forecast_data: dict,
        top_n: int
    ) -> tuple[dict, float]:
        """
        Allocates weights to individual tickers within each cluster using net Bullish sentiment,
        and retains only the top-N tickers per cluster. Dropped tickers' weights are redistributed
        within the same cluster. Entire cluster weight is orphaned only if no valid tickers exist.

        Args:
            cluster_weights (dict): Normalized cluster weights.
            clusters (dict): Mapping of tickers to cluster IDs.
            forecast_data (dict): Forecast sentiment data.
            top_n (int): Max tickers to retain per cluster.

        Returns:
            (dict, float): (Ticker weights, orphaned weight from empty clusters)
        """
        ticker_weights = defaultdict(float)
        orphaned_weight = 0.0

        for cluster_id, weight in cluster_weights.items():
            tickers = [tkr for tkr, cid in clusters.items() if cid == cluster_id]
            contributions = {}

            for tkr in tickers:
                forecast = forecast_data.get(tkr)
                if isinstance(forecast, np.ndarray):
                    forecast = forecast.item()
                if not isinstance(forecast, dict):
                    continue

                bullish = forecast.get("Bullish", 0.0)
                bearish = forecast.get("Bearish", 0.0)

                if bullish > 0:
                    contributions[tkr] = bullish

            if not contributions:
                orphaned_weight += weight
                continue

            # Retain top-N assets by bullish score
            top_tickers = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_contributions = dict(top_tickers)

            # Redistribute full cluster weight to top assets only
            total_contribution = sum(top_contributions.values())
            for tkr, contrib in top_contributions.items():
                ticker_weights[tkr] += weight * (contrib / total_contribution)

        return ticker_weights, orphaned_weight


    @staticmethod
    def _risk_parity_weights(tickers: list, price_data: pd.DataFrame, lookback: int) -> dict:
        """
        Computes risk parity weights for a list of tickers based on return covariance,
        handling SHV as a fixed-weight component.

        Args:
            tickers (list): List of ticker symbols.
            price_data (pd.DataFrame): Price data with tickers as columns.
            lookback (int): Number of periods to use for calculating returns.

        Returns:
            dict: Risk parity weights including SHV if present (sum to 1.0).
        """
        tickers = tickers.copy()

        shv_weight = 0.0
        if "SHV" in tickers:
            tickers.remove("SHV")
            shv_weight = 1.0  # assume SHV had full weight before optimization

        if not tickers:
            return {"SHV": 1.0}

        returns = price_data[tickers].pct_change().dropna().tail(lookback)
        cov_matrix = returns.cov().values
        n = len(tickers)

        def objective(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            marginal = cov_matrix @ w
            contrib = w * marginal / port_vol
            return np.sum((contrib - np.mean(contrib)) ** 2)

        init = np.ones(n) / n
        bounds = [(0, 1)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            raise ValueError("Risk parity optimization failed: " + result.message)

        optimized = dict(zip(tickers, result.x))

        # Scale down and reintroduce SHV
        scaled = {tkr: w * (1 - shv_weight) for tkr, w in optimized.items()}
        scaled["SHV"] = shv_weight

        return scaled
