"""
Module for inferencing models.
"""

import joblib
import logging
import os

import numpy as np

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
        self.start_date = config["current_start"]
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
        self.collect_current_state_probability(inferencing=inferencing)
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
        inferencing.state_labels = training.state_labels


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
        test_data = inferencing.test_data[["Momentum", "Volatility", "Short_Rates"]].values.copy()
        test_states = model.predict(test_data)
        inferencing.test_states = test_states


    @staticmethod
    def collect_current_state_probability(inferencing: ModelsInferencing):
        """
        Get the model's forecasted state probabilities 21 steps ahead using the transition matrix.

        Parameters
        ----------
        inferencing : ModelsInferencing
            The inference object containing the trained HMM, state labels, and test data.
        """
        posteriors = inferencing.model.predict_proba(inferencing.test_data.copy())

        pi_t = posteriors[-1]

        A = inferencing.model.transmat_

        A_t = np.linalg.matrix_power(A, 21)
        pi_t_forward = pi_t @ A_t

        labeled_distribution = {
            inferencing.state_labels.get(i, f"State {i}"): round(prob, 4)
            for i, prob in enumerate(pi_t_forward)
        }
        print(labeled_distribution)
        inferencing.forecast_distribution = labeled_distribution
