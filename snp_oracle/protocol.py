from typing import Optional

import bittensor as bt
import pydantic


class Challenge(bt.Synapse):
    """
    Protocol for handling encrypted prediction challenges between miners and validators.

    Attributes:
        timestamp: Time at which the validation is taking place
        prediction: Predicted value for next 60m candle
    """

    # Required request input, filled by sending dendrite caller
    timestamp: str = pydantic.Field(
        title="Timestamp",
        description="The timestamp at which the validation is taking place for",
        allow_mutation=False,
    )

    # Optional request output, filled by recieving axon
    prediction: Optional[float] = pydantic.Field(
        default=None,
        title="Prediction",
        description="The prediction to send to the dendrite caller",
    )

    # Optional request output, filled by recieving axon
    direction: Optional[bool] = pydantic.Field(
        default=None,
        title="Direction",
        description="The prediction to send to the dendrite caller",
    )

    def deserialize(self) -> float:
        """
        Deserialize the dummy output. This method retrieves the response from
        the miner in the form of dummy_output, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
            int: The deserialized response, which in this case is the value of dummy_output.

        Example:
        Assuming a Dummy instance has a dummy_output value of 5:
        >>> dummy_instance = Dummy(dummy_input=4)
        >>> dummy_instance.dummy_output = 5
        >>> dummy_instance.deserialize()
        5
        """
        return self.prediction
