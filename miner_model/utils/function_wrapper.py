"""
Function Wrapper Utility

Converts a simple predict() function into a PredictionModel interface.
This allows students to write simple functions instead of classes.
"""

from typing import Optional
from ..model_interface import PredictionModel


class FunctionBasedModel(PredictionModel):
    """
    Wraps a simple predict() function into PredictionModel interface.

    Example:
        def my_predict(timestamp):
            return 12000.0

        model = FunctionBasedModel(my_predict)
    """

    def __init__(self, predict_func):
        """
        Args:
            predict_func: Function that takes timestamp (str) and returns Optional[float] (LoadMw)
        """
        self.predict_func = predict_func

    def predict(self, timestamp: str) -> Optional[float]:
        try:
            return self.predict_func(timestamp)
        except Exception as e:
            import bittensor as bt
            bt.logging.error(f"Error in predict function: {e}")
            return None

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass
