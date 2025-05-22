"""
"""
import joblib
import os

import numpy as np

import hmm.utilities as utilities


class ModelsInferenceProcessor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_distribution = {}
        self.raw_weight = 0
        self.weight = 0

    def process(self):
        """
        """
        model = self.load_model(ticker=self.ticker)
        training = self.load_training(ticker=self.ticker)
        self.infer_states(model=model, training=training)
        self.label_states(training=training)
        self.compute_weight()  # <- New method to compute weight based on forecast

    def load_model(self, ticker):
        """
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

    def label_states(self, training):
        """
        """
        state_label_dict = utilities.label_states(training=training)
        print(f"{training.state_labels}")
        print(f"{self.ticker}: {state_label_dict}")

    def infer_states(self, model, training, n_steps=21):
        """
        """
        test_data = training.test_data[['Momentum', 'Volatility']].values
        test_states = utilities.smooth_states(model.predict(test_data))
        training.test_states = test_states

        last_state = test_states[-1]
        state_labels = training.state_labels.copy()

        final_prob_dist = self.forecast_final_state_distribution(
            model=model,
            last_state_index=last_state,
            n_steps=n_steps,
            state_labels=state_labels
        )

        self.forecast_distribution = final_prob_dist

        print(f"\nTicker: {self.ticker} — Forecast at step {n_steps}:")
        for label, prob in final_prob_dist.items():
            print(f"  {label}: {prob}")

    def forecast_final_state_distribution(self, model, last_state_index, n_steps, state_labels):
        """
        """
        A = model.transmat_
        n_states = model.n_components

        pi_t = np.zeros(n_states)
        pi_t[last_state_index] = 1.0

        for _ in range(n_steps):
            pi_t = np.dot(pi_t, A)

        final_probs = {
            state_labels[i]: round(pi_t[i], 4)
            for i in range(n_states)
        }

        return final_probs

    def compute_weight(self, initial_weight=1/13):
        """
        """
        probs = self.forecast_distribution
        bullish = probs.get('Bullish', 0)
        neutral = probs.get('Neutral', 0)
        bearish = probs.get('Bearish', 0)
        top = max(probs, key=probs.get)

        if bearish > 0.15:
            raw_weight = 0
        elif bullish > 0.7:
            raw_weight = 1.5 * initial_weight
        elif bullish > 0.4:
            raw_weight = 1.0 * initial_weight
        elif bullish > 0.2:
            raw_weight = 0.5 * initial_weight
        elif neutral > 0.5 and top == 'Neutral':
            raw_weight = 0.25 * initial_weight
        else:
            raw_weight = 0.1 * initial_weight

        self.raw_weight = raw_weight
