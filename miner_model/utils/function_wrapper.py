"""
Function Wrapper Utility

Converts a simple predict() function into a PredictionModel interface.
This allows students to write simple functions instead of classes.
"""

from typing import Tuple, Optional, List
from ..model_interface import PredictionModel


class FunctionBasedModel(PredictionModel):
    """
    Wraps a simple predict() function into PredictionModel interface.
    
    This allows students to write just a function instead of a full class.
    
    Example:
        def my_predict(timestamp):
            return 7.25, [7.10, 7.40]
        
        model = FunctionBasedModel(my_predict)
    """
    
    def __init__(self, predict_func):
        """
        Initialize with a predict function.
        
        Args:
            predict_func: Function that takes timestamp (str) and returns
                         (prediction: float, interval: [lower, upper])
        """
        self.predict_func = predict_func
    
    def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
        """
        Call the wrapped predict function.
        
        Args:
            timestamp: ISO format timestamp string
        
        Returns:
            Tuple of (prediction, interval) from the wrapped function
        """
        try:
            return self.predict_func(timestamp)
        except Exception as e:
            import bittensor as bt
            bt.logging.error(f"Error in predict function: {e}")
            return None, None
    
    def initialize(self) -> bool:
        """
        Always returns True - function is already callable.
        """
        return True
    
    def cleanup(self) -> None:
        """
        No cleanup needed for functions.
        """
        pass

