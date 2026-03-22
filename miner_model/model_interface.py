# The MIT License (MIT)
"""
Model Interface for Bittbridge Miner

This module defines the abstract base class that all predictive models must implement
to work with the Bittbridge subnet. Your custom model must inherit from PredictionModel
and implement the predict() method.
"""

from abc import ABC, abstractmethod
from typing import Optional
import bittensor as bt


class PredictionModel(ABC):
    """
    Abstract base class for all prediction models in the Bittbridge subnet.

    All custom models must inherit from this class and implement the predict() method.
    The model returns a point forecast of LoadMw (New England energy demand) for the target timestamp.

    Example:
        class MyCustomModel(PredictionModel):
            def __init__(self):
                pass

            def predict(self, timestamp: str) -> Optional[float]:
                return 12000.0  # predicted LoadMw (MW)
    """

    @abstractmethod
    def predict(self, timestamp: str) -> Optional[float]:
        """
        Generate a LoadMw (New England energy demand) point prediction for the given timestamp.

        Args:
            timestamp: ISO format timestamp string (e.g., "2024-01-15T10:30:00+00:00")
                     This represents the time for which the forecast is requested.

        Returns:
            Predicted LoadMw (MW), or None if prediction cannot be made (e.g., API failure).

        Notes:
            - Return None if the model fails; the validator assigns zero reward for that round.
            - LoadMw for New England is typically 10,000-15,000 MW
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
