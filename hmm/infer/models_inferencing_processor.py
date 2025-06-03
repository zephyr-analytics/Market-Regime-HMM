"""
Module for inferencing models.
"""
import joblib
import os

import numpy as np

import hmm.utilities as utilities
from hmm.infer.models_inferencing import ModelsInferencing
from hmm.results import InferencingResultsProcessor


class ModelsInferenceProcessor:
    """
    Class for infering on the trained Gaussian HMM.
    """
    def __init__(self, config: dict, ticker: str):
        self.config=config
        self.ticker = ticker
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.forecast_distribution = {}

    def process(self, max_retries: int=10):
        """
        Method to process through inferencing.

        Parameters
        ----------
        max_retries : int
            Number of attempts to ensure proper state transition.
        """
        inferencing = self.initialize_models_inferencing(
            ticker=self.ticker, start_date=self.start_date, end_date=self.end_date
        )

        self.load_model(inferencing=inferencing)
        self.load_training(inferencing=inferencing)

        for attempt in range(1, max_retries + 1):
            print(f"\n[{self.ticker}] Inference attempt {attempt}...")
            self.infer_states(inferencing=inferencing)

            is_stable = self._evaluate_model_quality(inferencing=inferencing)
            if is_stable:
                break
            elif attempt < max_retries:
                print(f"[{self.ticker}] Retrying inference due to instability...")
            else:
                print(f"[{self.ticker}] Maximum retries reached. Proceeding with last inference.")

        self.label_states(inferencing=inferencing)
        self.predict_future_state(inferencing=inferencing)
        results = InferencingResultsProcessor(inferencing=inferencing)
        results.process()

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
        end_data : str
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

        test_data = inferencing.test_data[['Momentum', 'Volatility']].values
        test_states = utilities.smooth_states(model.predict(test_data))
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
        print(f"{inferencing.state_labels}")
        print(f"{inferencing.ticker}: {state_label_dict}")

    @staticmethod
    def _evaluate_model_quality(inferencing: ModelsInferencing):
        """
        Method to evaluate quality of the model and retrain if necessary.

        Parameters
        ----------
        inferencing : ModelsInferencing
            ModelsInferencing instances.
        """
        result = utilities.evaluate_state_stability(inferencing.test_states)
        print(f"[{inferencing.ticker}] Model stability evaluation:")
        print(f"  - Transition rate: {result['transition_rate']}")
        print(f"  - Transition windows: {result['transitions']}")

        if result["is_unstable"]:
            print(f"  - WARNING: Model is unstable. Reason: {result['reason']}")
            return False
        else:
            print("  - Model is stable.")
            return True

    @staticmethod
    def predict_future_state(inferencing: ModelsInferencing, n_steps: int=21):
        """
        Method to predict future states based on n number of timesteps forward.

        Parameters
        ----------
        inferencing : ModelsInferencing
            ModelsInferencing instances.
        n_steps : int
            The number of time steps to predict forward.
        """
        last_state = inferencing.test_states[-1]
        state_labels = inferencing.state_labels.copy()
        A = inferencing.model.transmat_
        n_states = inferencing.model.n_components

        pi_t = np.zeros(n_states)
        pi_t[last_state] = 1.0

        for _ in range(n_steps):
            pi_t = np.dot(pi_t, A)

        labeled_distribution = {
            state_labels.get(i, f"State {i}"): round(prob, 4)
            for i, prob in enumerate(pi_t)
        }

        inferencing.forecast_distribution = labeled_distribution
