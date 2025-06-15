"""
Module for directing runner tasks.
"""

from hmm.runner.train_runner import TrainRunner
from hmm.runner.infer_runner import InferRunner
from hmm.runner.build_runner import BuildRunner
from hmm.runner.signals_runner import SignalsRunner
from hmm.runner.test_runner import TestRunner
from hmm.runner.tune_runner import TuneRunner

def get_runner(mode: str, config: dict, data):
    """
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
