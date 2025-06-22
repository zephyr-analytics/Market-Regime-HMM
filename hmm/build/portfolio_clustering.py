"""
Getter and setter module for clustering portfolio.
"""

import numpy as np
import pandas as pd


class PortfolioClustering:
    """
    Class for managing the data and configuration required for clustering portfolio assets.
    """

    def __init__(self):
        self._parsed_objects: dict=None
        self._min_clusters: int=None
        self._max_clusters: int=None
        self._start_date: str=None
        self._end_date: str=None
        self._price_data: pd.DataFrame=None
        self._moving_average: int=None
        self._state_data: dict=None
        self._forecast_data: dict=None
        self._sequences: list=None
        self._clustering_tickers: list=None
        self._clusters: dict=None
        self._cluster_assets: dict=None
        self._max_assets_per_cluster: int=None
        self._risk_lookback: int=None

    @property
    def parsed_objects(self) -> dict:
        """
        Getter method for parsed_objects.

        Returns
        -------
        dict : Dictionary of parsed inference objects.
        """
        return self._parsed_objects

    @parsed_objects.setter
    def parsed_objects(self, value: dict):
        """
        Setter method for parsed_objects.

        Parameters
        ----------
        value : dict
            Dictionary of parsed inference objects.
        """
        self._parsed_objects = value

    @property
    def min_clusters(self) -> int:
        """
        Getter method for min_clusters.

        Returns
        -------
        int : Integer of lower bound limit of clusters.
        """
        return self._min_clusters

    @min_clusters.setter
    def min_clusters(self, value: int):
        """
        Setter method for min_clusters.

        Parameters
        ----------
        value : int
            Integer of lower bound limit of clusters.
        """
        self._min_clusters = value

    @property
    def max_clusters(self) -> int:
        """
        Getter method for max_clusters.

        Returns
        -------
        int : Integer of upper bound limit of clusters.
        """
        return self._max_clusters

    @max_clusters.setter
    def max_clusters(self, value: int):
        """
        Setter method for max_clusters.

        Parameters
        ----------
        value : int
            Integer of upper bound limit of clusters.
        """
        self._max_clusters = value

    @property
    def start_date(self) -> str:
        """
        Getter method for start_date.

        Returns
        -------
        str : String representing the start date.
        """
        return self._start_date

    @start_date.setter
    def start_date(self, value: str):
        """
        Setter method for start_date.

        Parameters
        ----------
        value : str
            String representing the start date.
        """
        self._start_date = value

    @property
    def end_date(self) -> str:
        """
        Getter method for end_date.

        Returns
        -------
        str : String representing the end date.
        """
        return self._end_date

    @end_date.setter
    def end_date(self, value: str):
        """
        Setter method for end_date.

        Parameters
        ----------
        value : str
            String representing the end date.
        """
        self._end_date = value

    @property
    def price_data(self) -> str:
        """
        Getter method for price_data.

        Returns
        -------
        pd.DataFrame : Dataframe of price data for assets.
        """
        return self._price_data

    @price_data.setter
    def price_data(self, value: str):
        """
        Setter method for price_data.

        Parameters
        ----------
        value : pd.DataFrame
            Dataframe of price data for assets.
        """
        self._price_data = value

    @property
    def moving_average(self) -> int:
        """
        Getter method for moving_average.

        Returns
        -------
        int : Integer representing the moving average length.
        """
        return self._moving_average

    @moving_average.setter
    def moving_average(self, value: int):
        """
        Setter method for moving_average.

        Parameters
        ----------
        value : int
            Integer representing the moving average length.
        """
        self._moving_average = value

    @property
    def state_data(self) -> int:
        """
        Getter method for state_data.

        Returns
        -------
        int : Integer representing the moving average length.
        """
        return self._state_data

    @state_data.setter
    def state_data(self, value: int):
        """
        Setter method for state_data.

        Parameters
        ----------
        value : int
            Integer representing the moving average length.
        """
        self._state_data = value

    @property
    def forecast_data(self) -> dict:
        """
        Getter method for forecast_data.

        Returns
        -------
        dict : Dictionary of propagated forward forecast probability.
        """
        return self._forecast_data

    @forecast_data.setter
    def forecast_data(self, value: dict):
        """
        Setter method for forecast_data.

        Parameters
        ----------
        value : dict
            Dictionary of propagated forward forecast probability.
        """
        self._forecast_data = value

    @property
    def sequences(self) -> list:
        """
        Getter method for sequences.

        Returns
        -------
        list : List of state sequences.
        """
        return self._sequences

    @sequences.setter
    def sequences(self, value: list):
        """
        Setter method for sequences.

        Parameters
        ----------
        value : list
            List of state sequences.
        """
        self._sequences = value

    @property
    def clustering_tickers(self) -> list:
        """
        Getter method for clustering_tickers.

        Returns
        -------
        list : List of tickers to be clustered.
        """
        return self._clustering_tickers

    @clustering_tickers.setter
    def clustering_tickers(self, value: list):
        """
        Setter method for clustering_tickers.

        Parameters
        ----------
        value : list
            List of tickers to be clustered.
        """
        self._clustering_tickers = value

    @property
    def clusters(self) -> dict:
        """
        Getter method for clusters.

        Returns
        -------
        dict: Dictionary containing cluster data from initial clustering.
        """
        return self._clusters

    @clusters.setter
    def clusters(self, value: dict):
        """
        Setter method for clusters.

        Parameters
        ----------
        value : dict
            Dictionary containing cluster data from initial clustering.
        """
        self._clusters = value

    @property
    def cluster_assets(self) -> dict:
        """
        Getter method for cluster_assets.

        Returns
        -------
        dict : List of tickers to be clustered.
        """
        return self._cluster_assets

    @cluster_assets.setter
    def cluster_assets(self, value: dict):
        """
        Setter method for cluster_assets.

        Parameters
        ----------
        value : dict
            List of tickers to be clustered.
        """
        self._cluster_assets = value

    @property
    def cluster_returns(self) -> dict:
        """
        Getter method for cluster_returns.

        Returns
        -------
        dict : Dictionary containing clusters as keys and returns as values.
        """
        return self._cluster_returns

    @cluster_returns.setter
    def cluster_returns(self, value: dict):
        """
        Setter method for cluster_returns.

        Parameters
        ----------
        value : dict
            Dictionary containing clusters as keys and returns as values.
        """
        self._cluster_returns = value

    @property
    def max_assets_per_cluster(self) -> int:
        """
        Getter method for max_assets_per_cluster.

        Returns
        -------
        int : Integer representing the upper cap on allowable assets per cluster.
        """
        return self._max_assets_per_cluster

    @max_assets_per_cluster.setter
    def max_assets_per_cluster(self, value: int):
        """
        Setter method for max_assets_per_cluster.

        Parameters
        ----------
        value : int
            Integer representing the upper cap on allowable assets per cluster.
        """
        self._max_assets_per_cluster = value

    @property
    def risk_lookback(self) -> int:
        """
        Getter method for risk_lookback.

        Returns
        -------
        int : Integer representing the risk parity lookback window.
        """
        return self._risk_lookback

    @risk_lookback.setter
    def risk_lookback(self, value: int):
        """
        Setter method for risk_lookback.

        Parameters
        ----------
        value : int
            Integer representing the risk parity lookback window.
        """
        self._risk_lookback = value
