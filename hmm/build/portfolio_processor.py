"""
Module for building portfolio.
"""

import glob
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from hmm.data.data_processor import DataProcessor
from hmm.results.portfolio_results_processor import PortfolioResultsProcessor

logger = logging.getLogger(__name__)


class PortfolioProcessor:
    """
    Class to take processed model data and build a portfolio.
    """
    def __init__(self, config: dict):
        self.config = config
        self.start_date = config["start_date"]
        self.end_date = config["current_end"]


    def process(self):
        """
        Method to process through the BuildProcessor.
        """
        file_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing")
        parsed_objects = self.load_models_inference(directory=file_path, tickers=self.config["tickers"])
        state_data = self.extract_states(parsed_objects=parsed_objects)
        seq_matrix, ticker_list = self.prepare_state_sequences(state_data, lookback=126)
        results = self.cluster_sequences(seq_matrix, ticker_list)
        clusters = results["clusters"]
        forecast_data = self.extract_forecast_distributions(parsed_objects=parsed_objects)
        category_weights = self.compute_categorical_weights_by_cluster(
            forecast_data=forecast_data, clusters=clusters
        )

        data_process = DataProcessor(config=self.config)
        price_data = data_process.process()
        price_data = price_data.loc[self.start_date:self.end_date]

        portfolio = self.build_final_portfolio(
            clusters=clusters,
            forecast_data=forecast_data,
            category_weights=category_weights,
            bearish_cutoff=self.config["bearish_cutoff"],
            price_data=price_data,
            sma_lookback=self.config["moving_average"]
        )
        # NOTE possibly use a getter and setter for all results.
        results_process = PortfolioResultsProcessor(
            config=self.config,
            Z=results["linkage_matrix"],
            n_clusters=results["n_clusters"],
            portfolio=portfolio
        )
        results_process.process()

        return portfolio


    @staticmethod
    def load_models_inference(directory: str, tickers: list) -> dict:
        """
        Method to load the persisted ModelsInference instance for each ticker.
        
        Parameters
        ----------
        directory : str
            String representing the file path.
        tickers : list
            List of str ticker symbols.

        Returns
        -------
        parsed_objects : dict
            Dictionary of loaded pickle files representing persisted inference files.
        """
        parsed_objects = {}

        for ticker in tickers:
            pattern = os.path.join(directory, f'{ticker}.pkl')
            matched_files = glob.glob(pattern)

            objects = []
            for file_path in matched_files:
                with open(file_path, 'rb') as f:
                    obj = pickle.load(f)
                    objects.append(obj)

            if objects:
                parsed_objects[ticker] = objects if len(objects) > 1 else objects[0]

        return parsed_objects


    @staticmethod
    def extract_states(parsed_objects: dict) -> dict:
        """
        Method to extract state data from the loaded ModelInference instance.

        Parameters
        ----------
        parsed_objects : dict
            Dictionary of loaded pickle files representing persisted inference files.
        Returns
        -------
        state_data : dict
            Dictionary of tickers and cooresponding state data.
        """
        state_data = {}

        for ticker, obj in parsed_objects.items():
            states = []

            train_states = getattr(obj, 'train_states', None)
            test_states = getattr(obj, 'test_states', None)
            state_labels = getattr(obj, 'state_labels', {})

            if train_states is not None:
                states.append(np.asarray(train_states))
            if test_states is not None:
                states.append(np.asarray(test_states))

            if states:
                combined_states = np.concatenate(states)

                labeled_states = np.vectorize(state_labels.get)(combined_states)

                state_data[ticker] = {
                    'raw': combined_states,
                    'labels': labeled_states
                }

        return state_data


    @staticmethod
    def prepare_state_sequences(state_data: dict, lookback: int) -> np.ndarray:
        """
        Method to parse state data into state sequences for clustering.

        Parameters
        ----------
        state_data : dict
            Dictionary of tickers and cooresponding state data.
        lookback : int
            Integer representing the cutoff lookback period for clustering.
        Returns 
        np.ndarray : An array of state sequences.
        """
        all_labels = set()
        for data in state_data.values():
            trimmed = data['labels'][-lookback:]
            all_labels.update(trimmed)

        encoder = LabelEncoder()
        encoder.fit(list(filter(None, all_labels)))

        sequences = []
        tickers = []

        for ticker, data in state_data.items():
            trimmed = data['labels'][-lookback:]
            encoded = encoder.transform(trimmed)
            sequences.append(encoded)
            tickers.append(ticker)

        return np.array(sequences), tickers


    @staticmethod
    def cluster_sequences(sequences: np.ndarray, tickers: list, max_clusters: int = 15) -> dict:
        """
        Method to cluster state sequences to determine portfolio categories.

        Parameters
        ----------
        sequences : np.ndarray

        tickers : list

        max_clusters : int

        Returns
        -------
        dict : 
        """
        epsilon = 1e-10
        sequences = np.array(sequences, dtype=np.float64)
        sequences = np.nan_to_num(sequences, nan=0.0)
        row_norms = np.linalg.norm(sequences, axis=1)
        zero_mask = row_norms == 0
        sequences[zero_mask] = epsilon

        distance_matrix = pdist(sequences, metric='cosine')

        Z = linkage(distance_matrix, method='average')

        scores = []
        label_map = {}

        for k in range(2, min(max_clusters + 1, len(sequences))):
            labels = fcluster(Z, k, criterion='maxclust')
            try:
                sil = silhouette_score(sequences, labels)
                ch = calinski_harabasz_score(sequences, labels)
                db = davies_bouldin_score(sequences, labels)
                scores.append([sil, ch, db])
                label_map[k] = labels
            except Exception:
                continue

        if not scores:
            raise ValueError("No valid clustering results found.")

        scores = np.array(scores)
        scores[:, 2] = -scores[:, 2]

        scaler = MinMaxScaler()
        scaled_scores = scaler.fit_transform(scores)

        mean_scores = scaled_scores.mean(axis=1)
        best_idx = np.argmax(mean_scores)
        best_k = list(label_map.keys())[best_idx]
        best_labels = label_map[best_k]

        cluster_map = dict(zip(tickers, best_labels))

        return {
            'linkage_matrix': Z,
            'clusters': cluster_map,
            'labels': best_labels,
            'n_clusters': best_k
        }


    @staticmethod
    def extract_forecast_distributions(parsed_objects: dict) -> dict:
        """
        Method to extract forecast_data from the loaded ModelsInference instance.

        Parameters
        ----------
        parsed_objects : dict
            Dictionary of model inference objects with `forecast_distribution` attributes.

        Returns
        -------
        forecast_data : dict
            Dictionary of forecast distributions by ticker with keys normalized.
        """
        forecast_data = {}

        for ticker, obj in parsed_objects.items():
            forecast = getattr(obj, 'forecast_distribution', None)
            if forecast is not None and isinstance(forecast, dict):
                if "Bullish" not in forecast:
                    mapped_forecast = {}
                    for k, v in forecast.items():
                        if k in {"State 0", "State 1", "State 2"}:
                            mapped_forecast["Bullish"] = mapped_forecast.get("Bullish", 0.0) + v
                        else:
                            mapped_forecast[k] = v
                    forecast_data[ticker] = np.asarray(mapped_forecast)
                else:
                    forecast_data[ticker] = np.asarray(forecast)

        return forecast_data


    @staticmethod
    def compute_categorical_weights_by_cluster(forecast_data: dict, clusters: dict) -> dict:
        """
        Calculate cluster weights for each category based on forecast data,
        and discount final weights by (Bullish - Bearish) to prioritize net bullish sentiment.

        Parameters
        ----------
        forecast_data : dict
            Dictionary of ticker to forecast probability dictionaries.
        clusters : dict
            Dictionary of ticker to cluster ID mapping.

        Returns
        -------
        category_weights : dict
            Dictionary of cluster weights for each category, normalized and discounted.
        """
        valid_categories = ['Bullish', 'Neutral', 'Bearish']
        cluster_scores = defaultdict(lambda: {cat: 0.0 for cat in valid_categories})
        cluster_adjusted_bullish = defaultdict(float)

        for ticker, forecast_array in forecast_data.items():
            cluster_id = clusters.get(ticker)
            if cluster_id is None:
                continue

            forecast_dict = forecast_array.item() if isinstance(forecast_array, np.ndarray) else forecast_array
            if not isinstance(forecast_dict, dict):
                continue

            bullish = forecast_dict.get("Bullish", 0.0)
            bearish = forecast_dict.get("Bearish", 0.0)
            neutral = forecast_dict.get("Neutral", 0.0)
            adjusted_bullish = bullish - bearish

            cluster_scores[cluster_id]["Bullish"] += bullish
            cluster_scores[cluster_id]["Neutral"] += neutral
            cluster_scores[cluster_id]["Bearish"] += bearish
            cluster_adjusted_bullish[cluster_id] += adjusted_bullish

        total_per_category = {cat: 0.0 for cat in valid_categories}
        for cluster_vals in cluster_scores.values():
            for cat in valid_categories:
                total_per_category[cat] += cluster_vals[cat]

        raw_weights = {cat: {} for cat in valid_categories}
        for cluster_id, scores in cluster_scores.items():
            for cat in valid_categories:
                total = total_per_category[cat]
                raw_weights[cat][cluster_id] = scores[cat] / total if total > 0 else 0.0

        # Apply adjusted bullish discounting
        category_weights = {cat: {} for cat in valid_categories}
        for cat in valid_categories:
            weighted = {
                cid: weight * max(cluster_adjusted_bullish.get(cid, 0.0), 0.0)
                for cid, weight in raw_weights[cat].items()
            }
            total = sum(weighted.values())
            if total > 0:
                category_weights[cat] = {
                    cid: w / total for cid, w in weighted.items()
                }
            else:
                category_weights[cat] = {cid: 0.0 for cid in weighted}

        return category_weights

# Equal Weight
    # @staticmethod
    # def build_final_portfolio(
    #     clusters: dict, forecast_data: dict, category_weights: dict, 
    #     bearish_cutoff: float, price_data: dict, sma_lookback: int
    # ):
    #     """
    #     Constructs a portfolio using discounted 'Bullish' cluster weights,
    #     where each cluster's weight is multiplied by net bullish sentiment
    #     (Bullish - Bearish). Assets are equally weighted within clusters,
    #     with reallocation to SHV as needed.

    #     Parameters
    #     ----------
    #     forecast_data : dict
    #         Dictionary of ticker to forecast probability dictionaries.
    #     clusters : dict
    #         Dictionary of ticker to cluster ID mapping.
    #     category_weights : dict
    #         Dictionary of cluster weights by category.
    #     bearish_cutoff : float
    #         Threshold beyond which assets are excluded due to bearish outlook.
    #     price_data : dict
    #         Dictionary mapping tickers to historical price Series (pd.Series).
    #     sma_lookback : int
    #         Lookback window for SMA calculation.

    #     Returns
    #     -------
    #     ticker_weights : dict
    #         Final portfolio weights per ticker.
    #     """
    #     ticker_weights = defaultdict(float)
    #     orphaned_weight = 0.0

    #     # Use only Bullish category cluster weights
    #     bullish_clusters = category_weights.get("Bullish", {})

    #     # Step 1: Compute adjusted (discounted) weights
    #     adjusted_cluster_weights = {}
    #     for cluster_id, weight in bullish_clusters.items():
    #         tickers_in_cluster = [tkr for tkr, cid in clusters.items() if cid == cluster_id]
    #         cluster_bullish = 0.0
    #         cluster_bearish = 0.0

    #         for tkr in tickers_in_cluster:
    #             forecast = forecast_data.get(tkr)
    #             if isinstance(forecast, np.ndarray):
    #                 forecast = forecast.item()
    #             if not forecast or not isinstance(forecast, dict):
    #                 continue
    #             cluster_bullish += forecast.get("Bullish", 0.0)
    #             cluster_bearish += forecast.get("Bearish", 0.0)

    #         adjusted_score = max(cluster_bullish - cluster_bearish, 0.0)
    #         adjusted_cluster_weights[cluster_id] = weight * adjusted_score

    #     # Step 2: Normalize adjusted cluster weights
    #     total_weight = sum(adjusted_cluster_weights.values())
    #     if total_weight == 0:
    #         return {'SHV': 1.0}
    #     for cid in adjusted_cluster_weights:
    #         adjusted_cluster_weights[cid] /= total_weight

    #     # Step 3: Distribute weights equally within valid assets per cluster
    #     for cluster_id, cluster_weight in adjusted_cluster_weights.items():
    #         tickers_in_cluster = [tkr for tkr, cid in clusters.items() if cid == cluster_id]

    #         valid_tickers = []
    #         for tkr in tickers_in_cluster:
    #             forecast = forecast_data.get(tkr)
    #             if isinstance(forecast, np.ndarray):
    #                 forecast = forecast.item()
    #             if not forecast or not isinstance(forecast, dict):
    #                 continue

    #             if forecast.get("Bearish", 0.0) <= bearish_cutoff:
    #                 valid_tickers.append(tkr)

    #         if not valid_tickers:
    #             orphaned_weight += cluster_weight
    #             continue

    #         equal_weight = cluster_weight / len(valid_tickers)
    #         for tkr in valid_tickers:
    #             ticker_weights[tkr] += equal_weight

    #     # Step 4: Handle SHV fallback
    #     if not ticker_weights:
    #         return {'SHV': 1.0}
    #     if orphaned_weight > 0:
    #         ticker_weights['SHV'] += orphaned_weight

    #     # Step 5: SMA filter
    #     filtered_weights = {}
    #     total_valid_weight = 0.0
    #     for tkr, weight in ticker_weights.items():
    #         prices = price_data.get(tkr)
    #         if prices is None or len(prices) < sma_lookback:
    #             continue
    #         sma = prices[-sma_lookback:].mean()
    #         if prices.iloc[-1] >= sma and prices.iloc[-2] >= sma:
    #             filtered_weights[tkr] = weight
    #             total_valid_weight += weight

    #     if not filtered_weights or total_valid_weight == 0:
    #         return {'SHV': 1.0}

    #     filtered_weights = {tkr: w / total_valid_weight for tkr, w in filtered_weights.items()}

    #     if any(np.isnan(w) or w <= 0 for w in filtered_weights.values()):
    #         return {'SHV': 1.0}

    #     return filtered_weights

#Bullish Wegiht
    @staticmethod
    def build_final_portfolio(
        clusters: dict, forecast_data: dict, category_weights: dict, 
        bearish_cutoff: float, price_data: dict, sma_lookback: int
    ):
        """
        Constructs a portfolio using discounted 'Bullish' cluster weights,
        where each cluster's weight is multiplied by net bullish sentiment
        (Bullish - Bearish). Assets are weighted within clusters based on 
        their net bullish contribution to the cluster total.

        Returns
        -------
        ticker_weights : dict
            Final portfolio weights per ticker.
        """
        from collections import defaultdict
        import numpy as np

        ticker_weights = defaultdict(float)
        orphaned_weight = 0.0

        # Step 1: Adjust cluster weights by net Bullish sentiment
        bullish_clusters = category_weights.get("Bullish", {})
        adjusted_cluster_weights = {}

        for cluster_id, weight in bullish_clusters.items():
            tickers_in_cluster = [tkr for tkr, cid in clusters.items() if cid == cluster_id]
            cluster_bullish = 0.0
            cluster_bearish = 0.0

            for tkr in tickers_in_cluster:
                forecast = forecast_data.get(tkr)
                if isinstance(forecast, np.ndarray):
                    forecast = forecast.item()
                if not forecast or not isinstance(forecast, dict):
                    continue
                cluster_bullish += forecast.get("Bullish", 0.0)
                cluster_bearish += forecast.get("Bearish", 0.0)

            adjusted_score = max(cluster_bullish - cluster_bearish, 0.0)
            adjusted_cluster_weights[cluster_id] = weight * adjusted_score

        # Step 2: Normalize cluster weights
        total_weight = sum(adjusted_cluster_weights.values())
        if total_weight == 0:
            return {'SHV': 1.0}
        for cid in adjusted_cluster_weights:
            adjusted_cluster_weights[cid] /= total_weight

        # Step 3: Distribute weights by (Bullish - Bearish) within each cluster
        for cluster_id, cluster_weight in adjusted_cluster_weights.items():
            tickers_in_cluster = [tkr for tkr, cid in clusters.items() if cid == cluster_id]

            valid_tickers = {}
            for tkr in tickers_in_cluster:
                forecast = forecast_data.get(tkr)
                if isinstance(forecast, np.ndarray):
                    forecast = forecast.item()
                if not forecast or not isinstance(forecast, dict):
                    continue

                if forecast.get("Bearish", 0.0) <= bearish_cutoff:
                    contribution = max(forecast.get("Bullish", 0.0) - forecast.get("Bearish", 0.0), 0.0)
                    if contribution > 0:
                        valid_tickers[tkr] = contribution

            total_contribution = sum(valid_tickers.values())
            if total_contribution == 0:
                orphaned_weight += cluster_weight
                continue

            for tkr, contribution in valid_tickers.items():
                ticker_weights[tkr] += cluster_weight * (contribution / total_contribution)

        # Step 4: Handle SHV fallback
        if not ticker_weights:
            return {'SHV': 1.0}
        if orphaned_weight > 0:
            ticker_weights['SHV'] += orphaned_weight

        # Step 5: SMA filter
        filtered_weights = {}
        total_valid_weight = 0.0
        for tkr, weight in ticker_weights.items():
            prices = price_data.get(tkr)
            if prices is None or len(prices) < sma_lookback:
                continue
            sma = prices[-sma_lookback:].mean()
            if prices.iloc[-1] >= sma and prices.iloc[-2] >= sma:
                filtered_weights[tkr] = weight
                total_valid_weight += weight

        if not filtered_weights or total_valid_weight == 0:
            return {'SHV': 1.0}

        filtered_weights = {tkr: w / total_valid_weight for tkr, w in filtered_weights.items()}

        if any(np.isnan(w) or w <= 0 for w in filtered_weights.values()):
            return {'SHV': 1.0}

        return filtered_weights
