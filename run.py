"""
"""
import argparse
import json
import os
import pickle

from hmm.train.models_training_processor import ModelsTrainingProcessor
from hmm.infer.models_inferencing_processor import ModelsInferenceProcessor

def load_config():
    config_path = os.path.join(os.getcwd(), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Market Regime HMM Model using JSON Config")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument("--infer", action="store_true", help="Run inference using the model")
    args = parser.parse_args()

    config = load_config()
    tickers = config["tickers"]
    start_date = config["start_date"]
    end_date = config["end_date"]

    for ticker in tickers:
        if args.train:
            print(f"Training model for {ticker}...")
            model = ModelsTrainingProcessor(ticker=ticker, start_date=start_date, end_date=end_date)
            training = model.process()

            file_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "training", f"{ticker}_training.pkl")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(training, f)

        elif args.infer:
            print(f"Running inference for {ticker}...")
            model = ModelsInferenceProcessor(ticker=ticker, start_date=start_date, end_date=end_date)
            model.process()
            pass

if __name__ == "__main__":
    main()
