"""
Module for the BuildRunner.
"""

import logging

import logger_config
from hmm.runner.base_runner import BaseRunner
from hmm.build.portfolio_processor import PortfolioProcessor

logger = logging.getLogger(__name__)


class BuildRunner(BaseRunner):
    """
    Class for running build tasks.
    """
    def run(self):
        logger.info("Building single portfolio...")
        self.config["current_end"] = self.config["end_date"]
        builder = PortfolioProcessor(config=self.config, data=self.data)
        portfolio = builder.process()
        logger.info(f"Built portfolio: {portfolio}")
