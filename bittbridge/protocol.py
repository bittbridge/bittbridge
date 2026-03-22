# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Optional

import bittensor as bt
import pydantic

class Challenge(bt.Synapse):
    """
    Challenge Synapse:
    Used by validators to request a New England energy demand (LoadMw) prediction for a given timestamp.
    Miners respond with a point estimate only.
    """
    
    # Required request input, filled by sending dendrite caller.
    timestamp: str = pydantic.Field(
        ...,
        title="Timestamps",
        description="The timestamp to predict from",
        allow_mutation=False,
    )

    # Optional request output, filled by recieving axon.
    prediction: Optional[float] = pydantic.Field(
        default=None,
        title="Predictions",
        description="The predicted LoadMw (MW) to send to the dendrite caller",
    )

    def deserialize(self) -> float:
        # Return the point estimate prediction for scoring
        return self.prediction
