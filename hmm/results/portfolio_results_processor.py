"""
"""

from collections import defaultdict

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from scipy.cluster.hierarchy import dendrogram

from hmm import utilities


class PortfolioResultsProcessor:
    """
    Class to encapsulate all result-related methods from the PortfolioProcessor.
    """
    def __init__(self, config, n_clusters, portfolio):
        self.config = config
        self.n_clusters = n_clusters
        self.portfolio = portfolio


    def process(self):
        # self.plot_dendrogram(Z=self.Z, tickers=self.config["tickers"], best_k=self.n_clusters)
        self.plot_portfolio(ticker_weights=self.portfolio)
        # self.generate_pdf_report()


    # @staticmethod
    # def plot_dendrogram(Z, tickers, best_k):
    #     """
    #     """
    #     plt.figure(figsize=(12, 6))
    #     dendrogram(Z, labels=tickers, leaf_rotation=90)
    #     plt.title(f"Hierarchical Clustering of Tickers (auto k={best_k})")
    #     plt.xlabel("Ticker")
    #     plt.ylabel("Distance")
    #     plt.tight_layout()
    #     utilities.save_plot(filename="cluster_distribution.png", plot_type="cluster_distribution", plot_sub_folder="build")
    #     plt.close()


    @staticmethod
    def plot_portfolio(ticker_weights: dict):
        """
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


    # @staticmethod
    # def generate_pdf_report(clusters, forecast_data, category_weights, output_path="portfolio_report.pdf"):
    #     """
    #     """
    #     c = canvas.Canvas(output_path, pagesize=letter)
    #     width, height = letter
    #     y = height - inch
    #     line_height = 12

    #     def write_line(text, indent=0):
    #         nonlocal y
    #         if y < inch:
    #             c.showPage()
    #             c.setFont("Helvetica", 10)
    #             y = height - inch
    #         c.drawString(inch + indent, y, text)
    #         y -= line_height

    #     c.setFont("Helvetica-Bold", 16)
    #     write_line("Portfolio Clustering Report")
    #     c.setFont("Helvetica", 10)
    #     y -= 10

    #     write_line("Cluster Breakdown", indent=0)
    #     cluster_assets = defaultdict(list)
    #     for ticker, cluster in clusters.items():
    #         cluster_assets[cluster].append(ticker)

    #     for cluster_id, tickers in sorted(cluster_assets.items()):
    #         write_line(f"Cluster {cluster_id}:", indent=10)
    #         for t in tickers:
    #             write_line(f"- {t}", indent=20)
    #         y -= 4

    #     y -= 10
    #     write_line("Category Weights by Cluster", indent=0)
    #     for category, cluster_weights in category_weights.items():
    #         write_line(f"{category}:", indent=10)
    #         for cluster_id, weight in sorted(cluster_weights.items()):
    #             write_line(f"- Cluster {cluster_id}: {weight:.2%}", indent=20)
    #         y -= 4

    #     y -= 10
    #     write_line("Forecast Distribution by Asset", indent=0)
    #     for ticker, forecast_array in forecast_data.items():
    #         forecast_dict = forecast_array.item() if isinstance(forecast_array, np.ndarray) else forecast_array
    #         write_line(f"{ticker}:", indent=10)
    #         for k in ['Bullish', 'Neutral', 'Bearish']:
    #             write_line(f"{k}: {forecast_dict.get(k, 0):.2%}", indent=20)
    #         y -= 4

    #     c.save()
