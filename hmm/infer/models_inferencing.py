"""
Getter and setter module for inferencing models.
"""

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM


class ModelsInferencing:
    """
    Class for managing the data and configuration required for inferencing models.
    """

    def __init__(self):
        self._ticker: str=None
        self._start_date: str=None
        self._end_date: str=None
        self._model: GaussianHMM=None
        self._train_data: pd.Series=None
        self._test_data: pd.Series=None
        self._train_states: pd.Series=None
        self._test_states:  np.ndarray=None
        self._state_labels: dict=None
        self._forecast_distribution: dict=None

    @property
    def ticker(self) -> str:
        """
        Getter method for ticker.

        Returns
        -------
        str: The ticker symbol of the financial asset.
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
    def train_data(self) -> pd.Series:
        """
        Getter method for train_data.

        Returns
        -------
        pd.Series : Series of trimmed training data.
        """
        return self._train_data

    @train_data.setter
    def train_data(self, value: pd.Series):
        """
        Setter method for train_data.

        Parameters
        ----------
        value : pd.Series
            Series of trimmed training data.
        """
        self._train_data = value

    @property
    def test_data(self) -> pd.Series:
        """
        Getter method for test_data.

        Returns
        -------
        pd.Series : Series of trimmed test_data.
        """
        return self._test_data

    @test_data.setter
    def test_data(self, value: pd.Series):
        """
        Setter method for test_data.

        Parameters
        ----------
        value : pd.Series
            Series of trimmed test_data.
        """
        self._test_data = value

    @property
    def train_states(self) -> dict:
        """
        dict: A dictionary of training states or results from model fitting.
        """
        return self._train_states

    @train_states.setter
    def train_states(self, value: dict):
        """
        Set the training states.

        Parameters
        ----------
        value : dict
            Dictionary of training states, such as hidden states or results.
        """
        self._train_states = value

    @property
    def test_states(self) -> dict:
        """
        dict: A dictionary of training states or results from model fitting.
        """
        return self._test_states

    @test_states.setter
    def test_states(self, value: dict):
        """
        Set the training states.

        Parameters
        ----------
        value : dict
            Dictionary of training states, such as hidden states or results.
        """
        self._test_states = value

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

    @property
    def forecast_distribution(self) -> dict:
        """
        Getter method for forecast_distribution.

        dict : Dictionary containing forecasted distributions of states.
        """
        return self._forecast_distribution

    @forecast_distribution.setter
    def forecast_distribution(self, value: dict):
        """
        Setter method for forecast_distribution.

        Parameters
        ----------
        value : dict
            Dictionary containing forecasted distributions of states.
        """
        self._forecast_distribution = value
