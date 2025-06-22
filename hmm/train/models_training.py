"""
Getter and setter module for training models.
"""

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM


class ModelsTraining:
    """
    Class for managing the data and configuration required for training models.
    """

    def __init__(self):
        self._ticker: str=None
        self._start_date: str=None
        self._end_date: str=None
        self._data: pd.DataFrame=None
        self._train_data: pd.Series=None
        self._test_data: pd.Series=None
        self._train_states: np.ndarray=None
        self._features: pd.Series=None
        self._model: GaussianHMM=None
        self._state_labels: dict=None

    @property
    def ticker(self) -> str:
        """
        Getter method for ticker.

        Returns
        -------
        str : The ticker symbol of the financial asset.
        """
        return self._ticker

    @ticker.setter
    def ticker(self, value: str):
        """
        Setter method for ticker.

        Parameters
        ----------
        value : str
            The ticker symbol of the financial asset.
        """
        self._ticker = value

    @property
    def start_date(self) -> str:
        """
        Getter method for start_date.

        Returns
        -------
        str: The start date for training data selection (format: 'YYYY-MM-DD').
        """
        return self._start_date

    @start_date.setter
    def start_date(self, value: str):
        """
        Setter method for start date.

        Parameters
        ----------
        value : str
            Start date in 'YYYY-MM-DD' format.
        """
        self._start_date = value

    @property
    def end_date(self) -> str:
        """
        Getter method for end_date.

        str: The end date for training data selection (format: 'YYYY-MM-DD').
        """
        return self._end_date

    @end_date.setter
    def end_date(self, value: str):
        """
        Setter method for end_date.

        Parameters
        ----------
        value : str
            End date in 'YYYY-MM-DD' format.
        """
        self._end_date = value

    @property
    def train_data(self) -> pd.DataFrame:
        """
        Getter method for train_data.

        Returns
        -------
        pd.DataFrame: A string reference or path to the training dataset.
        """
        return self._train_data

    @train_data.setter
    def train_data(self, value: pd.DataFrame):
        """
        Setter method for train_data.

        Parameters
        ----------
        value : pd.DataFrame
            Reference or path to the training dataset.
        """
        self._train_data = value

    @property
    def test_data(self) -> pd.Series:
        """
        Getter method for test_data.

        Returns
        -------
        pandas.Series: The testing data series.
        """
        return self._test_data

    @test_data.setter
    def test_data(self, value: pd.Series):
        """
        Setter method for testing data.

        Parameters
        ----------
        value : pandas.Series
            Time series data used for testing.
        """
        self._test_data = value

    @property
    def features(self) -> list:
        """
        Getter method for features.

        Returns
        -------
        list: A list of feature names used for training.
        """
        return self._features

    @features.setter
    def features(self, value: list):
        """
        Setter method for features.

        Parameters
        ----------
        value : list
            List of feature names.
        """
        self._features = value

    @property
    def data(self) -> pd.Series:
        """
        Getter method for data.

        Returns
        -------
        pandas.Series: The complete dataset used for training and testing.
        """
        return self._data

    @data.setter
    def data(self, value: pd.Series):
        """
        Setter method for data.

        Parameters
        ----------
        value : pandas.Series
            Time series data used for training and evaluation.
        """
        self._data = value

    @property
    def model(self) -> GaussianHMM:
        """
        Getter method for the GaussianHMM model object.

        Returns
        -------
        GaussianHMM : Trained GaussianHMM model object.
        """
        return self._model

    @model.setter
    def model(self, value: GaussianHMM):
        """
        Setter method for the GaussianHMM model object.

        Parameters
        ----------
        value : GaussianHMM
            Trained GaussianHMM model object.
        """
        self._model = value

    @property
    def train_states(self) -> dict:
        """
        Getter method for train_states

        Returns
        -------
        dict: A dictionary of training states or results from model fitting.
        """
        return self._train_states

    @train_states.setter
    def train_states(self, value: dict):
        """
        Setter method for train_states.

        Parameters
        ----------
        value : dict
            Dictionary of training states, such as hidden states or results.
        """
        self._train_states = value

    @property
    def state_labels(self) -> dict:
        """
        Getter method for state_labels.

        Returns
        -------
        dict : Dictionary containing state label mappings to n_states.
        """
        return self._state_labels

    @state_labels.setter
    def state_labels(self, value: dict):
        """
        Setter method for state labels.

        Parameters
        ----------
        value : dict
            Dictionary containing state label mappings to n_states.
        """
        self._state_labels = value
