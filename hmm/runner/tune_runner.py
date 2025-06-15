"""
Module for tuning hyperparameters using a BaseRunner subclass.
"""

import os
import copy
import json
import itertools
import logging
import pandas as pd

import logger_config
from hmm import utilities
from hmm.runner.base_runner import BaseRunner
from hmm.runner.test_runner import TestRunner

logger = logging.getLogger(__name__)


class TuneRunner(BaseRunner):
    """
    Class for tuning hyperparameters by running a BaseRunner subclass with different configs.
    """
    def __init__(self, config: dict, data: pd.DataFrame):
        """
        Initialize TuneRunner.

        Args:
            config (dict): Base configuration including a "grid" entry.
            data (pd.DataFrame): Market or asset data.
        """
        super().__init__(config=config, data=data)
        self.param_grid = config.get("grid", {})


    def custom_score(self, results_df: pd.DataFrame) -> float:
        """
        Custom scoring logic for evaluating model performance.

        Args:
            results_df (pd.DataFrame): DataFrame of test results.

        Returns:
            float: Score for this configuration.
        """
        portfolio_values = utilities.compute_portfolio_value(returns=results_df["portfolio_return"])
        cagr = utilities.calculate_cagr(portfolio_value=portfolio_values)
        max_drawdown = utilities.calculate_max_drawdown(portfolio_value=portfolio_values)
        return abs(cagr / max_drawdown)


    def run(self) -> tuple[dict, list]:
        """
        Executes tuning over all parameter combinations.

        Returns:
            tuple: (best_params_dict, list of all tuning logs)
        """
        tuning_log = []
        best_score = float("-inf")
        best_params = None

        param_names = list(self.param_grid.keys())
        param_combos = list(itertools.product(*self.param_grid.values()))

        logger.info(f"Beginning tuning over {len(param_combos)} combinations...")

        for i, combo in enumerate(param_combos, 1):
            trial_params = dict(zip(param_names, combo))
            logger.info(f"\n--- [{i}/{len(param_combos)}] Testing: {trial_params} ---")

            config_copy = copy.deepcopy(self.config)
            config_copy.update(trial_params)

            runner = TestRunner(config=config_copy, data=self.data)
            runner.run()

            if not os.path.exists("portfolio_test_results.csv"):
                logger.warning("Results file not found. Skipping this config.")
                continue

            results_df = pd.read_csv("portfolio_test_results.csv")
            score = self.custom_score(results_df)

            logger.info(f"Score: {score:.4f}")
            tuning_log.append({
                "params": trial_params,
                "score": score
            })
            print(tuning_log)
            if score > best_score:
                best_score = score
                best_params = trial_params

        # Dump all results to JSON
        with open("tuning_results.json", "w") as f:
            json.dump(tuning_log, f, indent=4)

        logger.info(f"\nBest Params: {best_params} with Score: {best_score:.4f}")
        return best_params, tuning_log
