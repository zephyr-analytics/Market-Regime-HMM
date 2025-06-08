"""
Module for inferencing models.
"""

import joblib
import logging
import os

import numpy as np

import hmm.utilities as utilities
from hmm.infer.models_inferencing import ModelsInferencing
from hmm.results import InferencingResultsProcessor

logger = logging.getLogger(__name__)


class ModelsInferenceProcessor:
    """
    Class for infering on the trained Gaussian HMM.
    """
    def __init__(self, config: dict, ticker: str):
        self.config = config
        self.ticker = ticker
        self.start_date = config["start_date"]
        self.end_date = config["current_end"]
        self.forecast_distribution = {}
        self.persist = config["persist"]


    def process(self):
        """
        Method to process through inferencing.
        """
        inferencing = self.initialize_models_inferencing(
            ticker=self.ticker, start_date=self.start_date, end_date=self.end_date
        )

        self.load_model(inferencing=inferencing)
        self.load_training(inferencing=inferencing)
        self.infer_states(inferencing=inferencing)

        self.label_states(inferencing=inferencing)
        self.predict_future_state(inferencing=inferencing)
        if self.persist:
            results = InferencingResultsProcessor(inferencing=inferencing)
            results.process()

            return inferencing
        else:

            return inferencing


    @staticmethod
    def initialize_models_inferencing(ticker: str, start_date: str, end_date: str) -> ModelsInferencing:
        """
        Method to initalize ModelsInferencing.

        Parameters
        ----------
        ticker : str
            String representing ticker symbol for training.
        start_date : str
            String representing start date for data retrival.
        end_date : str
            String representing end date for data retrival.
        """
        inferencing = ModelsInferencing()
        inferencing.ticker = ticker
        inferencing.start_date = start_date
        inferencing.end_date = end_date

        return inferencing


    @staticmethod
    def load_model(inferencing: ModelsInferencing):
        """
        Method to load the trained model to be parsed for inferencing.

        Parameters
        ----------
        inferencing : ModelsInferencing
            ModelsInferencing instances.
        """
        model_path = os.path.join(
            os.getcwd(), "hmm", "train", "artifacts", "models", f"{inferencing.ticker}_model.pkl"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Saved model not found for {inferencing.ticker}: {model_path}")
        model = joblib.load(model_path)

        inferencing.model = model


    @staticmethod
    def load_training(inferencing: ModelsInferencing):
        """
        Method to load the training and testing data associated with the model.

        Parameters
        ----------
        inferencing : ModelsInferencing
            ModelsInferencing instances.
        """
        training_path = os.path.join(
            os.getcwd(), "hmm", "train", "artifacts", "training", f"{inferencing.ticker}.pkl"
        )
        training = joblib.load(training_path)
        inferencing.train_data = training.train_data
        inferencing.test_data = training.test_data
        inferencing.train_states = training.train_states


    @staticmethod
    def infer_states(inferencing: ModelsInferencing):
        """
        Method to predict states for test_data.

        Parameters
        ----------
        inferencing : ModelsInferencing
            ModelsInferencing instances.
        """
        model = inferencing.model
        test_data = inferencing.test_data[['Momentum', 'Volatility', "Short_Rates"]].values.copy()
        test_states = model.predict(test_data)
        inferencing.test_states = test_states


    @staticmethod
    def label_states(inferencing: ModelsInferencing):
        """
        Method to label states based on inferencing.

        Parameters
        ----------
        inferencing : ModelsInferencing
            ModelsInferencing instances.
        """
        state_label_dict = utilities.label_states(inferencing=inferencing)
        inferencing.state_labels = state_label_dict


    @staticmethod
    def predict_future_state(inferencing: ModelsInferencing, n_steps: int=21, n_days: int=63):
        """
        Predict future state probabilities by computing an exponentially weighted average
        of posterior probabilities over the last `n_days`, and projecting them forward
        using the model's transition matrix.

        Parameters
        ----------
        inferencing : ModelsInferencing
            The inference object containing the trained HMM, state labels, and test data.
        n_steps : int, default=21
            Number of time steps to forecast forward.
        n_days : int, default=21
            Number of most recent observations to use for computing initial state distribution.
        """
        # Step 1: Get all posterior probabilities from the model
        posteriors = inferencing.model.predict_proba(inferencing.test_data)

        # Step 2: Apply exponential weighting to the last n_days posteriors
        raw_weights = np.exp(np.linspace(-2, 0, n_days))  # Recent days get higher weight
        weights = raw_weights / raw_weights.sum()
        pi_t = np.average(posteriors[-n_days:], axis=0, weights=weights)

        # Step 3: Forecast forward using transition matrix
        A = inferencing.model.transmat_
        for _ in range(n_steps):
            pi_t = np.dot(pi_t, A)

        # Step 4: Label and store the result
        labeled_distribution = {
            inferencing.state_labels.get(i, f"State {i}"): round(prob, 4)
            for i, prob in enumerate(pi_t)
        }

        inferencing.forecast_distribution = labeled_distribution
