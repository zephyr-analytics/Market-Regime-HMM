"""
"""


class BaseRunner:
    def __init__(self, config, data):
        self.config = config
        self.data = data

    def run(self):
        raise NotImplementedError("Subclasses must implement the run method")
