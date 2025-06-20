"""
Module for directing runner tasks.
"""

import pandas as pd

from hmm.runner.train_runner import TrainRunner
from hmm.runner.infer_runner import InferRunner
from hmm.runner.build_runner import BuildRunner
from hmm.runner.signals_runner import SignalsRunner
from hmm.runner.test_runner import TestRunner
from hmm.runner.tune_runner import TuneRunner

def get_runner(mode: str, config: dict, data: pd.DataFrame) -> dict:
    """
    Method to handle processing different terminal configurations.

    mode : str
        String representing the terminal args passed.
    config : dict
        Dictionary containing configured properties.
    data : pd.DataFrame
        Dataframe containing data for the run being processed.
    """
    runners = {
        "train": TrainRunner,
        "infer": InferRunner,
        "build": BuildRunner,
        "test": TestRunner,
        "tune": TuneRunner,
        "signals": SignalsRunner
    }
    if mode not in runners:
        raise ValueError(f"Unsupported mode: {mode}")
    return runners[mode](config, data)
