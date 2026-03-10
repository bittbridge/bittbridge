# The MIT License (MIT)
"""
Model Interface for Bittbridge Miner

This module defines the abstract base class that all predictive models must implement
to work with the Bittbridge subnet. Your custom model must inherit from PredictionModel
and implement the predict() method.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import bittensor as bt


class PredictionModel(ABC):
    """
    Abstract base class for all prediction models in the Bittbridge subnet.
    
    All custom models must inherit from this class and implement the predict() method.
    The model is responsible for generating LoadMw (New England energy demand) predictions and confidence intervals.
    
    Example:
        class MyCustomModel(PredictionModel):
            def __init__(self):
                # Initialize your model here
                # Load weights, set up API clients, etc.
                pass
            
            def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
                # Your prediction logic here
                prediction = 12000.0  # Your predicted LoadMw (MW)
                interval = [11800.0, 12200.0]  # [lower_bound, upper_bound] for 90% confidence
                return prediction, interval
    """
    
    @abstractmethod
    def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
        """
        Generate a LoadMw (New England energy demand) prediction for the given timestamp.
        
        This is the core method that validators will call. It should:
        1. Use the timestamp to determine what prediction to make
        2. Return a point estimate (prediction) and confidence interval
        
        Args:
            timestamp: ISO format timestamp string (e.g., "2024-01-15T10:30:00+00:00")
                     This represents the time from which the prediction should be made.
                     Typically, you'll predict the LoadMw 1 hour ahead from this timestamp.
        
        Returns:
            Tuple containing:
            - prediction (Optional[float]): The predicted LoadMw (MW).
              Return None if prediction cannot be made (e.g., API failure, insufficient data).
            - interval (Optional[List[float]]): A list [lower_bound, upper_bound] representing
              the 90% confidence interval for the prediction.
              Return None if interval cannot be estimated.
        
        Important Notes:
            - Both prediction and interval can be None if the model fails
            - The validator will ignore responses with None predictions (miner gets zero reward)
            - The interval should represent a 90% confidence interval: [lower, upper]
            - LoadMw for New England is typically 10,000-15,000 MW
        
        Example:
            >>> model = MyCustomModel()
            >>> prediction, interval = model.predict("2024-01-15T10:30:00+00:00")
            >>> print(f"Prediction: {prediction}, Interval: {interval}")
            Prediction: 12000.0, Interval: [11800.0, 12200.0]
        """
        pass
    
    def initialize(self) -> bool:
        """
        Optional initialization method called when the miner starts.
        
        Override this method if your model needs to:
        - Load pre-trained weights
        - Connect to external services
        - Warm up caches
        - Validate configuration
        
        Returns:
            bool: True if initialization successful, False otherwise.
                 If False, the miner will log a warning but continue running.
        
        Default implementation returns True (no initialization needed).
        """
        return True
    
    def cleanup(self) -> None:
        """
        Optional cleanup method called when the miner shuts down.
        
        Override this method if your model needs to:
        - Save state
        - Close database connections
        - Release resources
        
        Default implementation does nothing.
        """
        pass

