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
