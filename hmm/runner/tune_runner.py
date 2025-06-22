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

        Parameters
        ----------
        config : dict 
            Base configuration including a "grid" entry.
        data : pd.Dataframe
            Market or asset data.
        """
        super().__init__(config=config, data=data)
        self.param_grid = config.get("param_grid", {})

    @staticmethod
    def custom_score(results_df: pd.DataFrame) -> float:
        """
        Custom scoring logic for evaluating model performance.

        Parameters
        ----------
        results_df : pd.DataFrame: 
            DataFrame of test results.

        Returns
        -------
            float: Score for this configuration.
        """
        results_df["return_end"] = pd.to_datetime(results_df["return_end"])

        # Set index for resampling
        results_df = results_df.set_index("return_end").sort_index()

        # Get the return series
        filtered_returns = results_df["portfolio_return"]

        # Resample to yearly and compute compound returns
        yearly_returns = filtered_returns.resample('Y').apply(utilities.compound_returns)
        yearly_returns_df = yearly_returns.to_frame(name='Yearly Return')
        yearly_returns_df['Yearly Return'] *= 100  # Convert to percentage
        yearly_returns_df['Year'] = yearly_returns_df.index.year
        yearly_returns_df = yearly_returns_df.sort_values('Year')

        # Calculate AAR and Sharpe ratio
        aar = utilities.calculate_average_annual_return(returns=results_df["portfolio_return"])
        sharpe_ratio = aar / yearly_returns_df["Yearly Return"].std()

        return sharpe_ratio


    def run(self) -> tuple [dict, dict]:
        """
        Executes tuning over all parameter combinations.

        Returns
        -------
        best_params : dict 
            Dictionary of configuartions values as keys and params as values.
        tuning_log : dict
            Dictionary of all tuning runs and scores of the tuning run.
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
            results_df = runner.run()  # ← Must return DataFrame

            if results_df is None or results_df.empty:
                logger.warning("No results returned. Skipping this config.")
                continue

            score = self.custom_score(results_df)

            logger.info(f"Score: {score:.4f}")
            tuning_log.append({
                "params": trial_params,
                "score": float(score)  # Ensure JSON serializable
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
