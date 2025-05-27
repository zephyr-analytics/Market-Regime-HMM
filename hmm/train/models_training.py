import pandas as pd

class ModelsTraining:
    def __init__(self):
        self._ticker = None
        self._start_date = None
        self._end_date = None
        self._data = None
        self._train_data = None
        self._test_data = None
        self._train_states = None
        self._features = None
        self._model = None
        self._state_labels = None

    @property
    def ticker(self) -> str:
        """Get the train data."""
        return self._ticker

    @ticker.setter
    def ticker(self, value: str):
        """Set the train data with type enforcement."""
        self._ticker = value

    @property
    def start_date(self) -> str:
        """Get the train data."""
        return self._start_date

    @start_date.setter
    def start_date(self, value: str):
        """Set the train data with type enforcement."""
        self._start_date = value

    @property
    def end_date(self) -> str:
        """Get the train data."""
        return self._end_date

    @end_date.setter
    def end_date(self, value: str):
        """Set the train data with type enforcement."""
        self._end_date = value

    @property
    def train_data(self) -> str:
        """Get the train data."""
        return self._train_data

    @train_data.setter
    def train_data(self, value: str):
        """Set the train data with type enforcement."""
        self._train_data = value

    @property
    def test_data(self) -> pd.Series:
        """Get the test data."""
        return self._test_data

    @test_data.setter
    def test_data(self, value: pd.Series):
        """Set the test data with type enforcement."""
        self._test_data = value

    @property
    def features(self) -> list:
        """Get the features."""
        return self._features

    @features.setter
    def features(self, value: list):
        """Set the features with type enforcement."""
        self._features = value

    @property
    def data(self) -> pd.Series:
        """
        Get the training data.
        """

        return self._data

    @data.setter
    def data(self, value: pd.Series):
        """
        Set the training data with type enforcement.
        """
        self._data = value

    @property
    def model(self) -> dict:
        """
        Get the training data.
        """

        return self._model

    @model.setter
    def model(self, value: dict):
        """
        Set the training data with type enforcement.
        """
        self._model = value

    @property
    def train_states(self) -> dict:
        """
        Get the training data.
        """

        return self._train_states

    @train_states.setter
    def train_states(self, value: dict):
        """
        Set the training data with type enforcement.
        """
        self._train_states = value

    @property
    def state_labels(self) -> dict:
        """
        Get the training data.
        """

        return self._state_labels

    @state_labels.setter
    def state_labels(self, value: dict):
        """
        Set the training data with type enforcement.
        """
        self._state_labels = value
