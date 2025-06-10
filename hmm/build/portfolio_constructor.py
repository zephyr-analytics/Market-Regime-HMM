"""
Module for handling final portfolio construction.
"""

from collections import defaultdict
import numpy as np
import pandas as pd


class PortfolioConstructor:
    """
    Class for constructing a final portfolio using sentiment forecasts.
    """
    def __init__(
        self, config: dict, clusters: dict, forecast_data: dict, 
        category_weights: dict, price_data: dict
    ):
        self.config = config
        self.clusters = clusters
        self.forecast_data = forecast_data
        self.category_weights = category_weights
        self.price_data = price_data
        self.sma_lookback = config["moving_average"]
        self.max_assets_per_cluster = config["max_assets_per_cluster"]

    def process(self):
        """
        Constructs the portfolio by filtering and weighting tickers through a
        multi-step process including sentiment adjustment, SMA filtering, and
        cluster-based selection.

        Returns:
            dict: A dictionary of final ticker weights summing to 1.0.
        """
        ticker_weights = defaultdict(float)

        # Step 1: Adjust each cluster's weight by its net Bullish sentiment
        if self.config["equal_cluster_weight"]:
            adjusted_cluster_weights = self._equal_weight_clusters(
                self.clusters,
                self.forecast_data,
                self.category_weights
            )
        else:
            adjusted_cluster_weights = self._adjust_cluster_weights(
                self.clusters,
                self.forecast_data,
                self.category_weights
            )

        # Step 3: Allocate intra-cluster weights based on net Bullish-Bearish contribution
        ticker_weights, orphaned_weight = self._allocate_within_clusters(
            adjusted_cluster_weights,
            self.clusters,
            self.forecast_data
        )

        # Step 4: Redirect unused cluster weight to SHV if nothing passed the filter
        if not ticker_weights:
            return {'SHV': 1.0}
        if orphaned_weight > 0:
            ticker_weights['SHV'] += orphaned_weight

        # Step 5: Apply momentum filter using simple moving average (SMA)
        filtered_weights = self._apply_sma_filter(
            ticker_weights, self.price_data, self.sma_lookback
        )

        if not filtered_weights:
            return {'SHV': 1.0}

        # Step 6: Optionally restrict to top-N assets per cluster
        if self.config["equal_asset_weight"]:
            return filtered_weights
        else:
            top_filtered_weights = self._filter_top_assets_per_cluster(
                filtered_weights,
                self.clusters,
                top_n=self.max_assets_per_cluster
            )

        if not top_filtered_weights:
            return {'SHV': 1.0}

        return top_filtered_weights


    @staticmethod
    def _adjust_cluster_weights(
        clusters: dict,
        forecast_data: dict,
        category_weights: dict
    ) -> dict:
        """
        Adjusts each cluster's weight based on the net Bullish sentiment of 
        tickers that meet cutoff conditions.

        Args:
            clusters (dict): Mapping of tickers to cluster IDs.
            forecast_data (dict): Forecast sentiment scores for each ticker.
            category_weights (dict): Initial Bullish weights assigned to each cluster.
            bullish_cutoff (float): Minimum Bullish score required to include a ticker.
            bearish_cutoff (float): Maximum Bearish score allowed to include a ticker.

        Returns:
            dict: Adjusted cluster weights accounting for filtered net Bullish sentiment.
        """
        bullish_clusters = category_weights.get("Bullish", {})
        adjusted = {}

        for cluster_id, weight in bullish_clusters.items():
            tickers = [tkr for tkr, cid in clusters.items() if cid == cluster_id]
            net_sum = 0.0

            for tkr in tickers:
                forecast = forecast_data.get(tkr)
                if isinstance(forecast, np.ndarray):
                    forecast = forecast.item()
                if not isinstance(forecast, dict):
                    continue

                bullish = forecast.get("Bullish", 0.0)
                bearish = forecast.get("Bearish", 0.0)
                neutral = forecast.get("Neutral", 0.0)

                # Only include if it passes bullish or bearish cutoff
                if bullish > bearish and bullish > neutral:
                    net = max(bullish, 0.0)
                    if net > 0:
                        net_sum += net

            # Multiply cluster weight by total valid net sentiment
            adjusted[cluster_id] = weight * net_sum

        total = sum(adjusted.values())
        if total == 0:
            return {}
        adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted


    @staticmethod
    def _equal_weight_clusters(
        clusters: dict,
        forecast_data: dict,
        category_weights: dict
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
    def _allocate_within_clusters(
        cluster_weights: dict,
        clusters: dict,
        forecast_data: dict
    ) -> dict:
        """
        Allocates weights to individual tickers within each cluster using
        net Bullish-Bearish sentiment, applying filter cutoffs.

        Args:
            cluster_weights (dict): Normalized cluster weights.
            clusters (dict): Mapping of tickers to cluster IDs.
            forecast_data (dict): Forecast sentiment data.
            bearish_cutoff (float): Maximum allowed Bearish probability.
            bullish_cutoff (float): Minimum required Bullish probability.

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
                neutral = forecast.get("Neutral", 0.0)

                # Only include if meets bullish/bearish cutoff
                if bullish > bearish and bullish > neutral:
                    net = max(bullish, 0.0)
                    if net > 0:
                        contributions[tkr] = net

            total_contribution = sum(contributions.values())
            if total_contribution == 0:
                orphaned_weight += weight
                continue

            # Allocate cluster weight proportionally by net sentiment
            for tkr, contribution in contributions.items():
                ticker_weights[tkr] += weight * (contribution / total_contribution)
        print(f"Within cluster weight sum:{sum(ticker_weights.values())}")
        return ticker_weights, orphaned_weight


    @staticmethod
    def _apply_sma_filter(ticker_weights: dict, price_data: pd.DataFrame, lookback: int) -> dict:
        """
        Filters tickers using simple moving average momentum criteria:
        Price must be above its SMA for last two observations.
        Orphaned weights are reassigned to SHV.

        Args:
            ticker_weights (dict): Pre-filtered ticker weights.
            price_data (pd.DataFrame): Price history for each ticker (each column is a ticker).
            lookback (int): Lookback period for SMA.

        Returns:
            dict: Adjusted ticker weights with SHV absorbing rejected weights.
        """
        filtered = {}
        orphaned_weight = 0.0

        for tkr, weight in ticker_weights.items():
            prices = price_data.get(tkr)
            if prices is None or len(prices) < lookback:
                orphaned_weight += weight
                continue

            sma = prices[-lookback:].mean()
            if prices.iloc[-1] > sma and prices.iloc[-2] > sma:
                filtered[tkr] = weight
            else:
                orphaned_weight += weight

        total_valid_weight = sum(filtered.values())
        adjusted_weights = {}

        if total_valid_weight > 0.0:
            for tkr, weight in filtered.items():
                adjusted_weights[tkr] = weight / total_valid_weight * (1.0 - orphaned_weight)

        if orphaned_weight > 0.0:
            if "SHV" in adjusted_weights.keys():
                adjusted_weights["SHV"] += orphaned_weight
            else:
                adjusted_weights["SHV"] = orphaned_weight
        print(f"SMA weight sum:{sum(adjusted_weights.values())}")
        return adjusted_weights


    @staticmethod
    def _filter_top_assets_per_cluster(filtered_weights: dict, clusters: dict, top_n: int) -> dict:
        """
        Retains only the top-N weighted tickers in each cluster.

        Args:
            filtered_weights (dict): Ticker weights post SMA filter.
            clusters (dict): Mapping of tickers to cluster IDs.
            top_n (int): Maximum number of tickers to keep per cluster.

        Returns:
            dict: Final filtered and normalized ticker weights.
        """
        cluster_to_tickers = defaultdict(list)

        for tkr, weight in filtered_weights.items():
            cluster_id = clusters.get(tkr)
            if cluster_id is not None:
                cluster_to_tickers[cluster_id].append((tkr, weight))

        final_filtered = {}
        for cluster_id, tickers in cluster_to_tickers.items():
            top_tickers = sorted(tickers, key=lambda x: x[1], reverse=True)[:top_n]
            for tkr, weight in top_tickers:
                final_filtered[tkr] = weight

        total = sum(final_filtered.values())
        if total == 0.0:
            return {}

        # Final normalization
        return {tkr: w / total for tkr, w in final_filtered.items()}
