"""
Bittbridge Miner Model Plugin Package

This package provides a plugin system for integrating predictive models with the Bittbridge subnet.

Quick Start:
    1. Place your .h5 model file and .csv data file in miner_model/ directory
    2. Run: python -m miner_model.miner_plugin --netuid 420 --subtensor.network test ...
    
The my_model.py file will automatically discover and load your files.
"""

from .model_interface import PredictionModel
from .miner_plugin import Miner

__all__ = ['PredictionModel', 'Miner']

