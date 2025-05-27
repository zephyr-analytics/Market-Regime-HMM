"""
"""
import joblib
import os

import numpy as np

import hmm.utilities as utilities
from hmm.results.results_processor import ResultsProcessor


class ModelsInferenceProcessor:
    def __init__(self, config, ticker, start_date, end_date):
        self.config=config
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_distribution = {}
        self.raw_weight = 0
        self.weight = 0

    def process(self, max_retries=5):
        """
        Inference process with retry logic if model evaluation is unstable.
        """
        model = self.load_model(ticker=self.ticker)
        training = self.load_training(ticker=self.ticker)

        for attempt in range(1, max_retries + 1):
            print(f"\n[{self.ticker}] Inference attempt {attempt}...")
            test_states, test_data = self.infer_states(model=model, training=training)

            is_stable = self._evaluate_model_quality(test_states=test_states)
            if is_stable:
                break
            elif attempt < max_retries:
                print(f"[{self.ticker}] Retrying inference due to instability...")
            else:
                print(f"[{self.ticker}] Maximum retries reached. Proceeding with last inference.")

        self.label_states(training=training)
        self.compute_weight(initial_weight=self.config["weights"][self.ticker])
        results = ResultsProcessor(
            training=training, ticker=self.ticker, start_date=self.start_date,
            end_date=self.end_date, test_states=test_states, test_data=test_data
        )
        results.process()


    def load_model(self, ticker):
        """
        """
        model_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "models", f"{ticker}_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Saved model not found for {ticker}: {model_path}")

        model = joblib.load(model_path)

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

        return test_states, test_data
    
    def _evaluate_model_quality(self, test_states):
        result = utilities.evaluate_state_stability(test_states)
        print(f"[{self.ticker}] Model stability evaluation:")
        print(f"  - Transition rate: {result['transition_rate']}")
        print(f"  - Transition windows: {result['transitions']}")

        if result["is_unstable"]:
            print(f"  - WARNING: Model is unstable. Reason: {result['reason']}")
            return False
        else:
            print("  - Model is stable.")
            return True

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

    def compute_weight(self, initial_weight):
        """
        Computes a weight based on the forecasted probability distribution.
        Adjusts the base weight depending on market sentiment probabilities.
        """
        probs = self.forecast_distribution
        bullish = probs.get('Bullish', 0)
        neutral = probs.get('Neutral', 0)
        bearish = probs.get('Bearish', 0)
        top = max(probs, key=probs.get)

        # Immediate override: too much bearish risk
        if bearish > 0.15:
            raw_weight = 0

        elif bullish > 0.9:
            raw_weight = 1.5 * initial_weight

        elif 0.9 >= bullish > 0.75:
            raw_weight = 1.25 * initial_weight

        elif 0.75 >= bullish > 0.55:
            raw_weight = 1.1 * initial_weight
        # NOTE this is not correct logic.
        elif 0.55 >= bullish > 0.5:
            raw_weight = 1.0 * initial_weight

        elif top == 'Neutral':
            if bullish >= 0.05:
                raw_weight = initial_weight
            elif bullish <= bearish:
                raw_weight = 0.5 * initial_weight
            else:
                raw_weight = 0.25 * initial_weight

        else:
            # Default fallback — low exposure
            raw_weight = 0.1 * initial_weight

        print(f"Weight for: {self.ticker}: {raw_weight}\n")
