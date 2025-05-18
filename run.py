import argparse
import json
import os

from models_training_processor import ModelsTrainingProcessor

def load_config():
    config_path = os.path.join(os.getcwd(), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Market Regime HMM Model using JSON Config")
    args = parser.parse_args()

    config = load_config()
    tickers=config["tickers"]
    start_date=config["start_date"]
    end_date=config["end_date"]

    allocations = []

    for ticker in tickers:
        model = ModelsTrainingProcessor(ticker=ticker, start_date=start_date, end_date=end_date)
        model.process()
        # forecast_probs = model.forecast_state_distribution(n_steps=21)
        # total_bullish_prob = forecast_probs[model.bullish_state]

        # starting_weight = 0.09
        # if total_bullish_prob >= 0.95:
        #     final_weight = starting_weight
        # elif 0.7 <= total_bullish_prob < 0.95:
        #     final_weight = starting_weight / 2
        # else:
        #     final_weight = 0.0

        # allocations[ticker] = final_weight


if __name__ == "__main__":
    main()