"""
Module for training models.
"""
import joblib
import os

import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

import hmm.utilities as utilities
from hmm.train.models_training import ModelsTraining
from hmm.results.results_processor import ResultsProcessor


class ModelsTrainingProcessor:
    def __init__(self, config, ticker):
        self.ticker = ticker
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.n_states = 3

    def process(self, max_retries=5):
        """
        Method to process through training.

        Parameters
        ----------
        max_retries : int
            Number of attempts to ensure proper state transition.
        """
        data = self._load_data(ticker=self.ticker)
        if data is None:
            print(f"[{self.ticker}] Data loading failed. Skipping.")
            return None

        for attempt in range(1, max_retries + 1):
            print(f"\n[{self.ticker}] Training attempt {attempt}...")
            training = self.initialize_models_training(training_data=data)
            self.prepare_data(training=training)
            self._fit_model(training=training)
            self._label_states(training=training)

            is_stable = self._evaluate_model_quality(training=training)
            if is_stable:
                break
            elif attempt < max_retries:
                print(f"[{self.ticker}] Retrying model training due to instability...")
            else:
                print(f"[{self.ticker}] Maximum retries reached. Proceeding with last model.")

        self._save_model(training=training)
        results = ResultsProcessor(
            training=training, ticker=self.ticker, start_date=self.start_date, end_date=self.end_date
        )
        results.process()

        return training

    def _load_data(self, ticker):
        """
        """
        adj_close = yf.download(
            tickers=ticker,
            start=self.start_date,
            end=self.end_date,
            group_by='ticker',
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if adj_close is None or adj_close.empty:
            print(f"[{ticker}] No data downloaded.")
            self.data = None
            return

        if isinstance(adj_close.columns, pd.MultiIndex):
            series = adj_close[ticker]["Adj Close"].dropna()
        else:
            if "Adj Close" not in adj_close:
                print(f"[{ticker}] Missing 'Adj Close' in data.")
                self.data = None
                return
            series = adj_close["Adj Close"].dropna()

        return series
    
    def initialize_models_training(self, training_data) -> ModelsTraining:
        training = ModelsTraining()
        training.training_data = training_data

        return training

    def prepare_data(self, training: ModelsTraining):
        """
        Prepare data for model fitting/trianing.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        training_data = training.training_data.copy()
        ret_3m = utilities.compounded_return(training_data, 63)
        ret_6m = utilities.compounded_return(training_data, 126)
        ret_9m = utilities.compounded_return(training_data, 189)
        ret_12m = utilities.compounded_return(training_data, 252)

        momentum = (ret_3m + ret_6m + ret_9m + ret_12m) / 4

        rolling_vol_1m = training_data.pct_change().rolling(window=21).std()
        rolling_vol_3m = training_data.pct_change().rolling(window=63).std()
        vol_concat = pd.concat([rolling_vol_1m, rolling_vol_3m], axis=1)
        min_vol = vol_concat.min().min()
        max_vol = vol_concat.max().max()
        scaled_vol_1m = 1 - (rolling_vol_1m - min_vol) / (max_vol - min_vol)
        scaled_vol_3m = 1 - (rolling_vol_3m - min_vol) / (max_vol - min_vol)
        mean_scaled_vol = (scaled_vol_1m + scaled_vol_3m) / 2

        features = pd.concat([momentum, mean_scaled_vol], axis=1).dropna()
        features.columns = ['Momentum', 'Volatility']

        split_index = int(len(features) * 0.7)

        train_data = features.iloc[:split_index]
        test_data = features.iloc[split_index:]

        training.train_data = train_data
        training.test_data = test_data
        training.features = features


    def _fit_model(self, training: ModelsTraining):
        """
        Method to fit the data to the model.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        model = GaussianHMM(n_components=self.n_states, covariance_type="diag", tol=0.0001, n_iter=10000)
        model.fit(training.train_data[['Momentum', 'Volatility']].values)
        train_states = utilities.smooth_states(model.predict(training.train_data[['Momentum', 'Volatility']].values))

        training.model = model
        training.train_states = train_states

    def _label_states(self, training: ModelsTraining):
        """
        Method to label states based on training.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        state_label_dict = utilities.label_states(training=training)
        training.state_labels = state_label_dict
        print(f"{self.ticker}: {state_label_dict}")
    
    def _evaluate_model_quality(self, training: ModelsTraining):
        """
        Method to evaluate quality of the model.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        result = utilities.evaluate_state_stability(training.train_states)
        print(f"[{self.ticker}] Model stability evaluation:")
        print(f"  - Transition rate: {result['transition_rate']}")
        print(f"  - Transition windows: {result['transitions']}")

        if result["is_unstable"]:
            print(f"  - WARNING: Model is unstable. Reason: {result['reason']}")
            return False
        else:
            print("  - Model is stable.")
            return True

    def _save_model(self, training: ModelsTraining):
        """
        Method to persist model for inferencing.

        Parameters
        ----------
        training : ModelsTraining
            ModelsTraining instance.
        """
        models_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "models")
        os.makedirs(models_path, exist_ok=True)
        model_path = os.path.join(models_path, f"{self.ticker}_model.pkl")
        joblib.dump(training.model, model_path)
