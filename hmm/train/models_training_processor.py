"""
Module for training models.
"""

import joblib
import logging
import os

import numpy as np
import pandas as pd
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
        self.start_date = config["current_start"]
        self.end_date = config["current_end"]
        self.max_retries = config["max_retries"]
        self.n_states = 3
        self.momentum_intervals = config["momentum_intervals"]
        self.volatility_interval = config["volatility_interval"]
        self.persist = config["persist"]
        self.data = data.loc[self.start_date:self.end_date]


    def process(self):
        """
        Method to process through the training module.
        """
        training = self._initialize_models_training(
            ticker=self.ticker, start_date=self.start_date, end_date=self.end_date
        )
        self._load_data(training=training, data=self.data)

        for attempt in range(1, self.max_retries + 1):
            self._prepare_data(
                training=training,
                momentum_intervals=self.momentum_intervals,
                volatility_interval=self.volatility_interval,
                split=self.config["train_test_split"],
                data=self.data
            )

            converged = self._fit_model(
                n_states=self.n_states, training=training
            )
            if converged:
                self._label_states(training=training)
                break

        self._save_model(training=training)
        if self.persist:
            results = TrainingResultsProcessor(training=training)
            results.process()

        return training


    @staticmethod
    def _initialize_models_training(ticker: str, start_date: str, end_date: str) -> ModelsTraining:
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
        series = pd.Series(data[f"{ticker}"])
        training.data = series


    @staticmethod
    def _prepare_data(
        training: ModelsTraining,
        momentum_intervals: list,
        volatility_interval: int,
        split: float,
        data: pd.DataFrame
    ):
        """
        Method to prepare data for model fitting/training using StandardScaler for normalization, except for short rates.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        momentum_intervals : list
            List of integers representing the momentum time horizons.
        volatility_inverval : int
            Integer representing the volatility time horizon.
        split : float
            Float representing the percent of train and test splitting.
        data : pd.DataFrame
            DataFrame raw price and interest rate data.
        """
        training_data = training.data.copy()
        short_rate = data["DFF"].replace(0, 1e-6)

        returns = [
            utilities.compound_return(
                training_data.copy(), interval
            ) for interval in momentum_intervals[:4]
        ]
        momentum = sum(returns) / len(returns)

        rolling_vol_1 = training_data.pct_change().rolling(window=volatility_interval).std()

        features = pd.concat([momentum, rolling_vol_1, short_rate], axis=1).dropna()
        features.columns = ['Momentum', 'Volatility', "Short_Rates"]

        scaler = StandardScaler()
        scaled_part = scaler.fit_transform(features[['Momentum', 'Volatility']])
        scaled_features = pd.DataFrame(
            scaled_part, index=features.index, columns=['Momentum', 'Volatility']
        )

        scaled_features['Short_Rates'] = features['Short_Rates']

        split_index = int(len(scaled_features) * split)
        training.train_data = scaled_features.iloc[:split_index]
        training.test_data = scaled_features.iloc[split_index:]
        training.features = scaled_features


    @staticmethod
    def _fit_model(n_states: int, training: ModelsTraining) -> bool:
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
        X = training.train_data[['Momentum', 'Volatility', "Short_Rates"]].values.copy()

        kmeans = KMeans(n_clusters=n_states, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X)
        initial_means = kmeans.cluster_centers_

        covariances = np.zeros((n_states, X.shape[1]))
        for i in range(n_states):
            cluster_points = X[labels == i]
            if len(cluster_points) > 1:
                covariances[i] = np.var(cluster_points, axis=0) + 1e-4
            else:
                covariances[i] = np.var(X, axis=0) + 1e-4

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            tol=1e-6,
            n_iter=1000,
            verbose=False,
            params="stmc",
            init_params=""
        )

        model.startprob_ = np.full(n_states, 1.0 / n_states)
        model.transmat_ = np.full((n_states, n_states), 1.0 / n_states)
        model.means_ = initial_means
        model.covars_ = covariances
        model.fit(X)

        training.model = model
        training.train_states = model.predict(X)

        return model.monitor_.converged


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
