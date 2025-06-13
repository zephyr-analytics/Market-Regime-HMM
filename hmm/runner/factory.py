"""
"""

from hmm.runner.train_runner import TrainRunner
from hmm.runner.infer_runner import InferRunner
from hmm.runner.build_runner import BuildRunner
from hmm.runner.test_runner import TestRunner

def get_runner(mode: str, config, data):
    runners = {
        "train": TrainRunner,
        "infer": InferRunner,
        "build": BuildRunner,
        "test": TestRunner
    }
    if mode not in runners:
        raise ValueError(f"Unsupported mode: {mode}")
    return runners[mode](config, data)
