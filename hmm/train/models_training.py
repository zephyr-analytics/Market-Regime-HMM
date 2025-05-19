import pandas as pd

class ModelsTraining:
    def __init__(self):
        self._training_data = None
        self._train_data = None
        self._test_data = None
        self._features = None
        self._model = None

    @property
    def train_data(self) -> pd.Series:
        """Get the train data."""
        return self._train_data

    @train_data.setter
    def train_data(self, value: pd.Series):
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
    def training_data(self) -> pd.Series:
        """
        Get the training data.
        """

        return self._training_data

    @training_data.setter
    def training_data(self, value: pd.Series):
        """
        Set the training data with type enforcement.
        """
        self._training_data = value

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
