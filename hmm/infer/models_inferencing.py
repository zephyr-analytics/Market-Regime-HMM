"""
"""
import pandas as pd

class ModelsInferencing:
    """
    """

    def __init__(self):
        self._ticker = None
        self._start_date = None
        self._end_date = None
        self._model = None
        self._train_data = None
        self._test_data = None
        self._train_states = None
        self._test_states = None
        self._state_labels = None


    @property
    def ticker(self) -> str:
        """
        str: The ticker symbol of the financial instrument.
        """
        return self._ticker

    @ticker.setter
    def ticker(self, value: str):
        """
        Set the ticker symbol.

        Parameters
        ----------
        value : str
            The ticker symbol of the financial instrument.
        """
        self._ticker = value

    @property
    def start_date(self) -> str:
        """
        str: The start date for training data selection (format: 'YYYY-MM-DD').
        """
        return self._start_date

    @start_date.setter
    def start_date(self, value: str):
        """
        Set the start date for training data selection.

        Parameters
        ----------
        value : str
            Start date in 'YYYY-MM-DD' format.
        """
        self._start_date = value

    @property
    def end_date(self) -> str:
        """
        str: The end date for training data selection (format: 'YYYY-MM-DD').
        """
        return self._end_date

    @end_date.setter
    def end_date(self, value: str):
        """
        Set the end date for training data selection.

        Parameters
        ----------
        value : str
            End date in 'YYYY-MM-DD' format.
        """
        self._end_date = value

    @property
    def model(self) -> dict:
        """
        dict: A dictionary representing the trained model or model configuration.
        """
        return self._model

    @model.setter
    def model(self, value: dict):
        """
        Set the model configuration or trained model.

        Parameters
        ----------
        value : dict
            Dictionary containing model parameters or the model itself.
        """
        self._model = value

    @property
    def train_data(self) -> pd.DataFrame:
        """
        pd.DataFrame: A string reference or path to the training dataset.
        """
        return self._train_data

    @train_data.setter
    def train_data(self, value: pd.DataFrame):
        """
        Set the training data reference or path.

        Parameters
        ----------
        value : pd.DataFrame
            Reference or path to the training dataset.
        """
        self._train_data = value

    @property
    def test_data(self) -> pd.Series:
        """
        pandas.Series: The testing data series.
        """
        return self._test_data

    @test_data.setter
    def test_data(self, value: pd.Series):
        """
        Set the testing data.

        Parameters
        ----------
        value : pandas.Series
            Time series data used for testing.
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
        dict: A dictionary mapping states to labels or interpretations.
        """
        return self._state_labels

    @state_labels.setter
    def state_labels(self, value: dict):
        """
        Set the state labels.

        Parameters
        ----------
        value : dict
            Dictionary mapping state indices or codes to descriptive labels.
        """
        self._state_labels = value
