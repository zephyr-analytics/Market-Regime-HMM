"""
Module for training models.
"""

import datetime
import joblib
import logging
import os

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster._kmeans import KMeans
from sklearn.preprocessing import StandardScaler

import hmm.utilities as utilities
from hmm.train.models_training import ModelsTraining
from hmm.results import TrainingResultsProcessor

logger = logging.getLogger(__name__)


class ModelsTrainingProcessor:
    """
    Class for training Gaussian HMM.
    """
    def __init__(self, config: dict, data: pd.DataFrame, ticker: str):
        self.ticker = ticker
        self.config = config
        self.data = data
        self.start_date = config["start_date"]
        self.end_date = config["current_end"]
        self.max_retries = config["max_retries"]
        self.n_states = 3
        self.momentum_intervals = config["momentum_intervals"]
        self.volatility_interval = config["volatility_interval"]
        self.persist = config["persist"]

    def process(self):
        """
        Method to process through the training module.
        """
        training = self.initialize_models_training(
            ticker=self.ticker, start_date=self.start_date, end_date=self.end_date
        )
        self._load_data(training=training, data=self.data)

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"\n[{self.ticker}] Training attempt {attempt}...")
            self.prepare_data(
                training=training,
                momentum_intervals=self.momentum_intervals,
                volatility_interval=self.volatility_interval,
                split=self.config["train_test_split"]
            )

            converged = self._fit_model(
                n_states=self.n_states, training=training, max_retries=self.max_retries
            )
            if not converged:
                logger.info(f"[{self.ticker}] Retrying model training due to non-convergence...")
                continue

            self._label_states(training=training)

            is_stable = self._evaluate_model_quality(training=training)
            if is_stable:
                break
            elif attempt < self.max_retries:
                logger.info(f"[{self.ticker}] Retrying model training due to instability...")
            else:
                logger.info(f"[{self.ticker}] Maximum retries reached. Proceeding with last model.")

        self._save_model(training=training)
        if self.persist:
            results = TrainingResultsProcessor(training=training)
            results.process()
            
            return training
        else:

            return training


    @staticmethod
    def initialize_models_training(ticker: str, start_date: str, end_date: str) -> ModelsTraining:
        """
        Method to initalize ModelsTraining.

        Parameters
        ----------
        ticker : str
            String representing ticker symbol for training.
        start_date : str
            String representing start date for data retrival.
        end_data : str
            String representing end date for data retrival.
        """
        training = ModelsTraining()
        training.ticker = ticker
        training.start_date = start_date
        training.end_date = end_date

        return training


    @staticmethod
    def _load_data(training: ModelsTraining, data: pd.DataFrame):
        """
        Method to load data for preparation.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        ticker = training.ticker
        start_date = training.start_date
        end_date = training.end_date

        series = pd.Series(data[f"{ticker}"]).loc[start_date:end_date]

        training.data = series


    @staticmethod
    def prepare_data(
        training: ModelsTraining,
        momentum_intervals: list,
        volatility_interval: int,
        split: float,
        series="DFF"
    ):
        """
        Prepare data for model fitting/training using StandardScaler for normalization.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        start = training.start_date
        end = training.end_date
        try:
            rate = web.DataReader(series, "fred", start, end)
            rate.fillna(method="ffill", inplace=True)
            short_rate = rate

        except Exception as e:
            logger.error(f"Failed to load FRED short rate series '{series}': {e}")
            short_rate = None

        training_data = training.data.copy()

        ret_1m = utilities.compounded_return(training_data, momentum_intervals[0])
        ret_3m = utilities.compounded_return(training_data, momentum_intervals[1])
        ret_6m = utilities.compounded_return(training_data, momentum_intervals[2])
        ret_9m = utilities.compounded_return(training_data, momentum_intervals[3])
        momentum = (ret_1m + ret_3m + ret_6m + ret_9m) / 5

        rolling_vol_1 = training_data.pct_change().rolling(window=volatility_interval).std()

        features = pd.concat([momentum, rolling_vol_1, short_rate], axis=1).dropna()
        features.columns = ['Momentum', 'Volatility', "Short_Rates"]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        scaled_features = pd.DataFrame(scaled, index=features.index, columns=features.columns)

        split_index = int(len(scaled_features) * split)
        training.train_data = scaled_features.iloc[:split_index]
        training.test_data = scaled_features.iloc[split_index:]
        training.features = scaled_features


    @staticmethod
    def _fit_model(n_states: int, training: ModelsTraining, max_retries: int) -> bool:
        """
        Fits the HMM model with initialized means and covariances.

        Parameters
        ----------
        n_states : int
            Number of states the model should train on.
        training : ModelsTraining
            ModelsTraining instance.
        max_retries : int
            Number of retries to train the model.
        """
        X = training.train_data[['Momentum', 'Volatility', "Short_Rates"]].values

        kmeans = KMeans(n_clusters=n_states, random_state=42)
        labels = kmeans.fit_predict(X)
        initial_means = kmeans.cluster_centers_

        covariances = np.zeros((n_states, X.shape[1]))
        for i in range(n_states):
            cluster_points = X[labels == i]
            if len(cluster_points) > 1:
                covariances[i] = np.var(cluster_points, axis=0) + 1e-4
            else:
                covariances[i] = np.var(X, axis=0) + 1e-4

        for attempt in range(1, max_retries + 1):
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                tol=1e-5,
                n_iter=10000,
                verbose=False,
                params="stmc",
                init_params=""
            )

            model.startprob_ = np.full(n_states, 1.0 / n_states)
            model.transmat_ = np.full((n_states, n_states), 1.0 / n_states)
            model.means_ = initial_means
            model.covars_ = covariances

            model.fit(X)

            if model.monitor_.converged:
                logger.info(f"[{training.ticker}] Model converged on attempt {attempt}")
                training.model = model
                training.train_states = model.predict(X)
                return True
            else:
                logger.info(f"[{training.ticker}] WARNING: Model did not converge on attempt {attempt}")

        logger.info(f"[{training.ticker}] ERROR: Failed to converge after {max_retries} attempts.")
        training.model = model
        training.train_states = model.predict(X)

        return False


    @staticmethod
    def _label_states(training: ModelsTraining):
        """
        Method to label states based on training.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        state_label_dict = utilities.label_states(training=training)
        training.state_labels = state_label_dict
        logger.info(f"{training.ticker}: {state_label_dict}")


    @staticmethod
    def _evaluate_model_quality(training: ModelsTraining):
        """
        Method to evaluate quality of the model and retrain if necessary.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        result = utilities.evaluate_state_stability(training.train_states)

        if result["is_unstable"]:
            return False
        else:
            return True


    @staticmethod
    def _save_model(training: ModelsTraining):
        """
        Method to persist model for inferencing.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        models_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "models")
        os.makedirs(models_path, exist_ok=True)
        model_path = os.path.join(models_path, f"{training.ticker}_model.pkl")
        joblib.dump(training.model, model_path)
