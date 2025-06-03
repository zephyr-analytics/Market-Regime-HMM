"""
Module for training models.
"""
import joblib
import os

import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import hmm.utilities as utilities
from hmm.train.models_training import ModelsTraining
from hmm.results import TrainingResultsProcessor


class ModelsTrainingProcessor:
    """
    """
    def __init__(self, config: dict, ticker: str):
        self.ticker = ticker
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.max_retries = config["max_retries"]
        self.n_states = 3

    def process(self):
        """
        """
        training = self.initialize_models_training(
            ticker=self.ticker, start_date=self.start_date, end_date=self.end_date
        )
        self._load_data(training=training)

        for attempt in range(1, self.max_retries + 1):
            print(f"\n[{self.ticker}] Training attempt {attempt}...")
            self.prepare_data(training=training)

            converged = self._fit_model(
                n_states=self.n_states, training=training, max_retries=self.max_retries
            )
            if not converged:
                print(f"[{self.ticker}] Retrying model training due to non-convergence...")
                continue

            self._label_states(training=training)

            is_stable = self._evaluate_model_quality(training=training)
            if is_stable:
                break
            elif attempt < self.max_retries:
                print(f"[{self.ticker}] Retrying model training due to instability...")
            else:
                print(f"[{self.ticker}] Maximum retries reached. Proceeding with last model.")

        self._save_model(training=training)
        results = TrainingResultsProcessor(training=training)
        results.process()

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
    def _load_data(training: ModelsTraining):
        """
        Method to load data for preparation.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        ticker = training.ticker
        adj_close = yf.download(
            tickers=ticker,
            start=training.start_date,
            end=training.end_date,
            group_by='ticker',
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if isinstance(adj_close.columns, pd.MultiIndex):
            series = adj_close[ticker]["Adj Close"].dropna()
        else:
            series = adj_close["Adj Close"].dropna()

        training.data = series

    @staticmethod
    def prepare_data(training: ModelsTraining):
        """
        Prepare data for model fitting/training using StandardScaler for normalization.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        training_data = training.data.copy()

        ret_1m = utilities.compounded_return(training_data, 21)
        ret_3m = utilities.compounded_return(training_data, 63)
        ret_6m = utilities.compounded_return(training_data, 126)
        ret_9m = utilities.compounded_return(training_data, 189)
        ret_12m = utilities.compounded_return(training_data, 252)
        momentum = (ret_1m + ret_3m + ret_6m + ret_9m + ret_12m) / 5

        rolling_vol_1m = training_data.pct_change().rolling(window=21).std()
        rolling_vol_3m = training_data.pct_change().rolling(window=63).std()
        mean_vol = (rolling_vol_1m + rolling_vol_3m) / 2

        features = pd.concat([momentum, mean_vol], axis=1).dropna()
        features.columns = ['Momentum', 'Volatility']

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        scaled_features = pd.DataFrame(scaled, index=features.index, columns=features.columns)

        split_index = int(len(scaled_features) * 0.7)
        training.train_data = scaled_features.iloc[:split_index]
        training.test_data = scaled_features.iloc[split_index:]
        training.features = scaled_features

    @staticmethod
    def _fit_model(n_states: int, training: ModelsTraining, max_retries: int) -> bool:
        """
        Fits HMM model using training.train_data[['Momentum', 'Volatility']]
        and returns whether convergence was successful.
        """
        X = training.train_data[['Momentum', 'Volatility']].values

        for attempt in range(1, max_retries + 1):
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                tol=0.00001,
                n_iter=10000,
                verbose=False,
                params="stmc",
                init_params="stmc"
            )

            model.fit(X)

            if model.monitor_.converged:
                print(f"[{training.ticker}] Model converged on attempt {attempt}")
                training.model = model
                training.train_states = utilities.smooth_states(model.predict(X))
                return True
            else:
                print(f"[{training.ticker}] WARNING: Model did not converge on attempt {attempt}")

        print(f"[{training.ticker}] ERROR: Failed to converge after {max_retries} attempts.")
        training.model = model
        training.train_states = utilities.smooth_states(model.predict(X))
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
        print(f"{training.ticker}: {state_label_dict}")

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
        print(f"[{training.ticker}] Model stability evaluation:")
        print(f"  - Transition rate: {result['transition_rate']}")
        print(f"  - Transition windows: {result['transitions']}")

        if result["is_unstable"]:
            print(f"  - WARNING: Model is unstable. Reason: {result['reason']}")
            return False
        else:
            print("  - Model is stable.")
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
