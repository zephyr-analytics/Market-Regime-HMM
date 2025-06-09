"""
"""

import argparse
import logging
from hmm.data.data_processor import DataProcessor
from hmm.runner.factory import get_runner
from hmm import utilities

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Market Regime HMM Model using JSON Config")

    # Model operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train the model")
    mode_group.add_argument("--infer", action="store_true", help="Run inference using the model")
    mode_group.add_argument("--build", action="store_true", help="Build portfolios")
    mode_group.add_argument("--test", action="store_true", help="Test rolling portfolio performance")

    # Asset type selection (stock vs ETF)
    asset_group = parser.add_mutually_exclusive_group(required=True)
    asset_group.add_argument("--etf", action="store_true", help="Use ETF config")
    asset_group.add_argument("--stock", action="store_true", help="Use stock config")

    args = parser.parse_args()

    # Load config
    config = utilities.load_config(etf=args.etf, stocks=args.stock)

    # Process data
    data = DataProcessor(config=config).process()

    # Determine operation mode
    if args.train:
        mode = "train"
    elif args.infer:
        mode = "infer"
    elif args.build:
        mode = "build"
    elif args.test:
        mode = "test"
    else:
        raise ValueError("No valid mode selected")

    # Run the selected runner
    runner = get_runner(mode, config, data)
    runner.run()

if __name__ == "__main__":
    main()
