"""
Module for building portfolio.
"""

import glob
import logging
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import hmm.utilities as utilities

logger = logging.getLogger(__name__)


class BuildProcessor:
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
        results = self.cluster_and_plot_sequence(seq_matrix, ticker_list)
        clusters = results["clusters"]
        forecast_data = self.extract_forecast_distributions(parsed_objects=parsed_objects)
        category_weights = self.compute_categorical_weights_by_cluster(
            forecast_data=forecast_data, clusters=clusters
        )

        price_data=utilities.load_price_data(
            tickers=self.config["tickers"], start_date=self.start_date, end_date=self.end_date
        )

        portfolio = self.build_final_portfolio(
            clusters=clusters,
            forecast_data=forecast_data,
            category_weights=category_weights,
            bearish_cutoff=self.config["bearish_cutoff"],
            price_data=price_data,
            sma_lookback=self.config["moving_average"]
        )

        self.plot_portfolio(ticker_weights=portfolio)
        self.generate_pdf_report(clusters, forecast_data, category_weights)

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
    def cluster_and_plot_sequence(sequences: np.ndarray, tickers: list, max_clusters: int = 15) -> dict:
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

        plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=tickers, leaf_rotation=90)
        plt.title(f"Hierarchical Clustering of Tickers (auto k={best_k})")
        plt.xlabel("Ticker")
        plt.ylabel("Distance")
        plt.tight_layout()
        utilities.save_plot(filename="cluster_distribution.png", plot_type="cluster_distribution", plot_sub_folder="build")
        plt.close()

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
        Method to calculate initial cluster weights.
        Bullish probability weighting of clusters is first discounted by 
        bearish probability, which should provide higher risk adjusted returns.

        Parameters
        ----------
        forecast_data : dict
            Dictionary of ticker mapping to probabilities.
        clusters : dict
            Dictionary of ticker mapping to cluster IDs.

        Returns
        -------
        category_ weights : dict
            Dictionary containing cluster weights.
        """
        valid_categories = ['Bullish', 'Neutral', 'Bearish']
        cluster_scores = defaultdict(lambda: {cat: 0.0 for cat in valid_categories})

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

            adjusted_bullish = (bullish - bearish) / neutral if neutral > 0 else 0.0

            cluster_scores[cluster_id]["Bullish"] += max(0.0, adjusted_bullish)
            cluster_scores[cluster_id]["Neutral"] += neutral
            cluster_scores[cluster_id]["Bearish"] += bearish

        total_per_category = {cat: 0.0 for cat in valid_categories}
        for cluster_vals in cluster_scores.values():
            for cat in valid_categories:
                total_per_category[cat] += cluster_vals[cat]

        category_weights = {cat: {} for cat in valid_categories}
        for cluster_id, scores in cluster_scores.items():
            for cat in valid_categories:
                total = total_per_category[cat]
                category_weights[cat][cluster_id] = scores[cat] / total if total > 0 else 0.0

        return category_weights


    @staticmethod
    def build_final_portfolio(
        clusters: dict, forecast_data: dict, category_weights: dict, 
        bearish_cutoff: float, price_data: dict, sma_lookback: int
    ):
        """
        Calculates final portfolio structure and weights with SMA filter.
        Assets below their SMA have their weights reallocated proportionally to remaining assets.

        Parameters
        ----------
        forecast_data : dict
            Dictionary of ticker mapping to probabilities.
        clusters : dict
            Dictionary of ticker mapping to cluster IDs.
        category_weights : dict
            Dictionary containing cluster weights.
        bearish_cutoff : float
            Threshold beyond which assets are excluded due to bearish outlook.
        price_data : dict
            Dictionary mapping tickers to historical price Series (pd.Series).
        sma_lookback : int
            Lookback window for SMA calculation.

        Returns
        -------
        ticker_weights : dict
            Final ticker weight allocations with full distribution.
        """
        valid_categories = ['Bullish', 'Neutral', 'Bearish']
        ticker_weights = defaultdict(float)
        orphaned_weight = 0.0

        for category in valid_categories:
            cluster_weights = category_weights.get(category, {})

            for cluster_id, cluster_weight in cluster_weights.items():
                tickers_in_cluster = [tkr for tkr, cid in clusters.items() if cid == cluster_id]

                scores = []
                for tkr in tickers_in_cluster:
                    forecast = forecast_data.get(tkr)
                    if isinstance(forecast, np.ndarray):
                        forecast = forecast.item()
                    if not forecast or not isinstance(forecast, dict):
                        continue

                    if forecast.get("Bearish", 0.0) > bearish_cutoff:
                        continue

                    bullish = forecast.get("Bullish", 0.0)
                    bearish = forecast.get("Bearish", 0.0)
                    neutral = forecast.get("Neutral", 1e-6)

                    adjusted_bullish = max(bullish - bearish, 0.0)

                    if neutral == 0.0:
                        adjusted_score = adjusted_bullish
                    else:
                        adjusted_score = adjusted_bullish / neutral

                    scores.append((tkr, adjusted_score))

                if not scores:
                    orphaned_weight += cluster_weight
                    continue

                total_bullish = sum(score for _, score in scores)
                if total_bullish == 0:
                    equal_weight = cluster_weight / len(scores)
                    for tkr, _ in scores:
                        ticker_weights[tkr] += equal_weight
                else:
                    for tkr, score in scores:
                        norm_score = score / total_bullish
                        ticker_weights[tkr] += cluster_weight * norm_score

        if orphaned_weight > 0 and ticker_weights:
            total_allocated = sum(ticker_weights.values())
            for tkr in ticker_weights:
                proportion = ticker_weights[tkr] / total_allocated
                ticker_weights[tkr] += orphaned_weight * proportion

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


    @staticmethod
    def plot_portfolio(ticker_weights: dict):
        """
        Method to plot the final portfolio weights.

        Parameters
        ----------
        ticker_weights : dict
            Final ticker weight allocations.
        """
        if not ticker_weights:
            print("No weights to plot.")
            return

        sorted_items = sorted(ticker_weights.items(), key=lambda x: x[1], reverse=True)
        labels, weights = zip(*sorted_items)

        plt.figure(figsize=(10, 10))
        plt.pie(
            weights,
            labels=[f"{label} ({w:.2%})" for label, w in zip(labels, weights)],
            startangle=140,
            counterclock=False,
            wedgeprops=dict(edgecolor='w'),
            textprops={'fontsize': 9}
        )
        plt.title("Final Portfolio Composition")
        plt.axis('equal')
        plt.tight_layout()
        utilities.save_plot(filename="portfolio_allocation.png", plot_type="portfolio_allocation", plot_sub_folder="build")
        plt.close()


    @staticmethod
    def generate_pdf_report(clusters, forecast_data, category_weights, output_path="portfolio_report.pdf"):
        """
        Method to generate a pdf report detailing the overall weighting mechanics used to generate the portfolio weights.

        Parameters
        ----------
        clusters : dict
            Dictionary of ticker mapping to cluster IDs.
        forecast_data : dict
            Dictionary of ticker mapping to probabilities.
        category_ weights : dict
            Dictionary containing cluster weights.
        """
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter

        y = height - inch
        line_height = 12

        def write_line(text, indent=0):
            nonlocal y
            if y < inch:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - inch

            c.drawString(inch + indent, y, text)
            y -= line_height

        c.setFont("Helvetica-Bold", 16)
        write_line("Portfolio Clustering Report")
        c.setFont("Helvetica", 10)
        y -= 10

        write_line("Cluster Breakdown", indent=0)
        cluster_assets = defaultdict(list)
        for ticker, cluster in clusters.items():
            cluster_assets[cluster].append(ticker)

        for cluster_id, tickers in sorted(cluster_assets.items()):
            write_line(f"Cluster {cluster_id}:", indent=10)
            for t in tickers:
                write_line(f"- {t}", indent=20)
            y -= 4

        y -= 10
        write_line("Category Weights by Cluster", indent=0)

        for category, cluster_weights in category_weights.items():
            write_line(f"{category}:", indent=10)
            for cluster_id, weight in sorted(cluster_weights.items()):
                write_line(f"- Cluster {cluster_id}: {weight:.2%}", indent=20)
            y -= 4

        y -= 10
        write_line("Forecast Distribution by Asset", indent=0)
        for ticker, forecast_array in forecast_data.items():
            forecast_dict = forecast_array.item() if isinstance(forecast_array, np.ndarray) else forecast_array
            write_line(f"{ticker}:", indent=10)
            for k in ['Bullish', 'Neutral', 'Bearish']:
                write_line(f"{k}: {forecast_dict.get(k, 0):.2%}", indent=20)
            y -= 4

        c.save()
