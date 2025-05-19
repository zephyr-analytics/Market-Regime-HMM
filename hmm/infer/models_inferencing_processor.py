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
        model = self.load_model()
        self.infer_states(model=model)
        self.label_states()

    def load_model(self):
        """
        Load a previously saved model from disk.
        """
        model_path = os.path.join("models", f"{self.ticker}_hmm_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Saved model not found for {self.ticker}: {model_path}")

        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

        return model

    def infer_states(self, model):
        """
        """
        self.train_states = utilities.smooth_states(model.predict(self.train_data[['Momentum', 'Volatility']].values))
        self.test_states = utilities.smooth_states(model.predict(self.test_data[['Momentum', 'Volatility']].values))

    def label_states(self):
        pass
