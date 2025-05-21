"""
"""
import joblib
import os

import hmm.utilities as utilities


class ModelsInferenceProcessor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def process(self):
        """
        """
        model = self.load_model(ticker=self.ticker)
        training = self.load_training(ticker=self.ticker)
        self.infer_states(model=model, training=training)
        self.label_states(training=training)

    def load_model(self, ticker):
        """
        Load a previously saved model from disk.
        """
        model_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "models", f"{ticker}_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Saved model not found for {ticker}: {model_path}")

        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

        return model

    def load_training(self, ticker):
        """
        """
        training_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "training", f"{ticker}_training.pkl")
        training = joblib.load(training_path)

        return training

    def infer_states(self, model, training):
        """
        """
        test_states = utilities.smooth_states(model.predict(training.test_data[['Momentum', 'Volatility']].values))

    def label_states(self, training):
        state_label_dict = utilities.label_states(training=training)
        print(f"{training.state_labels}")
        print(f"{self.ticker}: {state_label_dict}")
