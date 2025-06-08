"""
"""

from collections import defaultdict

import numpy as np

# TODO create new methods for different weighting mechanics.
class PortfolioConstructor:

    @staticmethod
    def build_final_portfolio(
        clusters: dict, forecast_data: dict, category_weights: dict, 
        bearish_cutoff: float, price_data: dict, sma_lookback: int
    ):
        ticker_weights = defaultdict(float)

        # Step 1: Adjust cluster weights by net Bullish sentiment
        adjusted_cluster_weights = PortfolioConstructor._adjust_cluster_weights(
            clusters, forecast_data, category_weights
        )

        # Step 2: Normalize cluster weights
        normalized_cluster_weights = PortfolioConstructor._normalize_weights(adjusted_cluster_weights)
        if not normalized_cluster_weights:
            return {'SHV': 1.0}

        # Step 3: Allocate weights within clusters based on net sentiment
        ticker_weights, orphaned_weight = PortfolioConstructor._allocate_within_clusters(
            normalized_cluster_weights, clusters, forecast_data, bearish_cutoff
        )

        # Step 4: Assign leftover weight to SHV
        if not ticker_weights:
            return {'SHV': 1.0}
        if orphaned_weight > 0:
            ticker_weights['SHV'] += orphaned_weight

        # Step 5: Apply SMA-based momentum filter
        filtered_weights = PortfolioConstructor._apply_sma_filter(
            ticker_weights, price_data, sma_lookback
        )

        if not filtered_weights:
            return {'SHV': 1.0}

        return filtered_weights

    @staticmethod
    def _adjust_cluster_weights(clusters, forecast_data, category_weights):
        bullish_clusters = category_weights.get("Bullish", {})
        adjusted = {}

        for cluster_id, weight in bullish_clusters.items():
            tickers = [tkr for tkr, cid in clusters.items() if cid == cluster_id]
            bullish_sum, bearish_sum = 0.0, 0.0

            for tkr in tickers:
                forecast = forecast_data.get(tkr)
                if isinstance(forecast, np.ndarray):
                    forecast = forecast.item()
                if not isinstance(forecast, dict):
                    continue
                bullish_sum += forecast.get("Bullish", 0.0)
                bearish_sum += forecast.get("Bearish", 0.0)

            net_bullish = max(bullish_sum - bearish_sum, 0.0)
            adjusted[cluster_id] = weight * net_bullish

        return adjusted

    @staticmethod
    def _normalize_weights(weight_dict):
        total = sum(weight_dict.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in weight_dict.items()}

    @staticmethod
    def _allocate_within_clusters(cluster_weights, clusters, forecast_data, bearish_cutoff):
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
                if forecast.get("Bearish", 0.0) <= bearish_cutoff:
                    net = max(forecast.get("Bullish", 0.0) - forecast.get("Bearish", 0.0), 0.0)
                    if net > 0:
                        contributions[tkr] = net

            total_contribution = sum(contributions.values())
            if total_contribution == 0:
                orphaned_weight += weight
                continue

            for tkr, contribution in contributions.items():
                ticker_weights[tkr] += weight * (contribution / total_contribution)

        return ticker_weights, orphaned_weight

    @staticmethod
    def _apply_sma_filter(ticker_weights, price_data, lookback):
        filtered = {}
        total = 0.0

        for tkr, weight in ticker_weights.items():
            prices = price_data.get(tkr)
            if prices is None or len(prices) < lookback:
                continue

            sma = prices[-lookback:].mean()
            if prices.iloc[-1] >= sma and prices.iloc[-2] >= sma:
                filtered[tkr] = weight
                total += weight

        if total == 0.0:
            return {}

        normalized = {tkr: w / total for tkr, w in filtered.items()}
        if any(np.isnan(w) or w <= 0 for w in normalized.values()):
            return {}

        return normalized
