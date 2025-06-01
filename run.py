"""
"""
import argparse
import os
import pickle

import hmm.utilities as utilities
from hmm.build.build_processor import BuildProcessor
from hmm.train.models_training_processor import ModelsTrainingProcessor
from hmm.infer.models_inferencing_processor import ModelsInferenceProcessor



def main():
    """
    """
    parser = argparse.ArgumentParser(description="Run Market Regime HMM Model using JSON Config")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument("--infer", action="store_true", help="Run inference using the model")
    group.add_argument("--build", action="store_true")
    args = parser.parse_args()

    config = utilities.load_config()
    tickers = config["tickers"]

    for ticker in tickers:
        if args.train:
            print(f"Training model for {ticker}...")
            model = ModelsTrainingProcessor(config=config, ticker=ticker)
            training = model.process()

            file_path = os.path.join(os.getcwd(), "hmm", "train", "artifacts", "training", f"{ticker}.pkl")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(training, f)

        elif args.infer:
            print(f"Running inference for {ticker}...")
            model = ModelsInferenceProcessor(config=config, ticker=ticker)
            inferencing = model.process()

            file_path = os.path.join(os.getcwd(), "hmm", "infer", "artifacts", "inferencing", f"{ticker}.pkl")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(inferencing, f)
 
    if args.build:
        print(f"Building portfolio category weights...")
        build = BuildProcessor(config=config)
        build.process()

if __name__ == "__main__":
    main()
