# The MIT License (MIT)


"""
Bittbridge Miner Plugin

This is the main miner file that integrates your predictive model with the Bittensor network.
The miner handles network communication, while your model handles predictions.

To use this miner:
1. Place your .h5 model file and data (CSV or use ISO-NE API for LoadMw) in miner_model/ directory
2. Run the miner - it will auto-discover your model and data files

Example:
    # Place your model.h5 and data.csv in miner_model/
    # Run: python -m miner_model.miner_plugin
    # The my_model.py file will automatically find and load your files
"""

import time
import typing
import sys
import bittensor as bt

# Bittensor Miner Template:
import bittbridge

# Import base miner class which takes care of most of the boilerplate
from bittbridge.base.miner import BaseMinerNeuron

# Import the model interface
from .model_interface import PredictionModel
# Import auto-discovery utility
from .utils.model_loader import load_student_model


class Miner(BaseMinerNeuron):
    """
    Miner neuron class that integrates your predictive model with the Bittensor network.
    
    This class handles:
    - Network communication (receiving challenges from validators)
    - Request filtering (blacklist/priority)
    - Delegating predictions to your model
    
    To customize:
    1. Replace the model with your own (see __init__)
    2. The forward() method delegates to your model - no changes needed
    
    This class inherits from BaseMinerNeuron, which handles:
    - Wallet and subtensor setup
    - Metagraph synchronization
    - Logging configuration
    - Axon (server) setup
    """

    def __init__(self, config=None, model: PredictionModel = None):
        """
        Initialize the miner with a predictive model.
        
        Args:
            config: Bittensor configuration (will use defaults)
            model: An instance of a PredictionModel. Required - no default fallback.
        
        Example:
            # Load your model from student_models/ folder
            from .utils.model_loader import load_student_model
            model = load_student_model()
            miner = Miner(model=model)
        """
        super(Miner, self).__init__(config=config)
        
        # ============================================================
        # STEP 1: SETUP YOUR MODEL
        # ============================================================
        # Your model must implement the PredictionModel interface.
        # The my_model.py file in student_models/ will automatically discover
        # your .h5 and .csv files from the miner_model/ directory.
        #
        # ============================================================
        
        if model is None:
            raise ValueError(
                "No model provided. Please set up your model files.\n"
                "Steps:\n"
                "  1. Place your .h5 model file in miner_model/ directory\n"
                "  2. Place your .csv data file in miner_model/ directory\n"
                "  3. Make sure student_models/my_model.py exists\n"
                "  4. Run the miner again - it will auto-discover your files"
            )
        
        self.model = model
        
        # Initialize the model (load weights, connect to APIs, etc.)
        if not self.model.initialize():
            bt.logging.warning(
                "Model initialization returned False. "
                "Miner will continue but predictions may fail."
            )
        else:
            bt.logging.success("!!!!! Model initialized successfully !!!!!")

    async def forward(self, synapse: bittbridge.protocol.Challenge) -> bittbridge.protocol.Challenge:
        """
        Responds to the Challenge synapse from the validator.
        
        This method:
        1. Extracts the timestamp from the synapse
        2. Calls your model's predict() method
        3. Attaches the point prediction to the synapse
        4. Returns the synapse to the validator

        The validator scores point forecasts per docs/guide/10-incentive mechanism.md.

        Args:
            synapse: Challenge synapse containing the timestamp to predict from

        Returns:
            Challenge synapse with prediction filled in
        """
        # Extract timestamp from the challenge
        timestamp = synapse.timestamp
        
        bt.logging.debug(f"Received challenge for timestamp: {timestamp}")
        
        # ============================================================
        # STEP 2: GET PREDICTION FROM YOUR MODEL
        # ============================================================
        # Your model.predict(timestamp) returns Optional[float] (LoadMw in MW).
        # ============================================================

        try:
            prediction = self.model.predict(timestamp)

            if prediction is None:
                bt.logging.warning(
                    f"Model returned None prediction for timestamp {timestamp}. "
                    "Validator will ignore this response."
                )
                return synapse

            synapse.prediction = prediction

            bt.logging.success(f"Prediction for {timestamp}: {prediction}")
            
        except Exception as e:
            # Handle unexpected errors gracefully
            bt.logging.error(
                f"Error in model.predict() for timestamp {timestamp}: {e}"
            )
            # Return synapse with None values - validator will ignore
            return synapse
        
        return synapse

    async def blacklist(self, synapse: bittbridge.protocol.Challenge) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted.
        
        This method filters requests before they are processed. You can customize
        this to implement your own security policies.
        
        Current implementation:
        - Rejects requests without hotkeys
        - Optionally rejects non-registered entities
        - Optionally rejects non-validators
        
        Args:
            synapse: Challenge synapse (headers only, data not deserialized yet)
        
        Returns:
            Tuple[bool, str]: (should_blacklist, reason)
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # ============================================================
        # STEP 3: CUSTOMIZE BLACKLIST LOGIC (OPTIONAL)
        # ============================================================
        # You can add custom blacklist logic here, such as:
        # - Rate limiting
        # - IP-based filtering
        # - Reputation-based filtering
        # ============================================================
        
        try:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            # Hotkey not found in metagraph
            if not self.config.blacklist.allow_non_registered:
                bt.logging.trace(
                    f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"
            else:
                return False, "Non-registered hotkey allowed"

        # Check if validator permit is required
        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bittbridge.protocol.Challenge) -> float:
        """
        Determines the priority of incoming requests.
        
        Higher priority requests are processed first. This implementation
        prioritizes based on the validator's stake in the metagraph.
        
        Args:
            synapse: Challenge synapse
        
        Returns:
            float: Priority score (higher = process first)
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0
        try:
            caller_uid = self.metagraph.hotkeys.index(
                synapse.dendrite.hotkey
            )
            priority = float(self.metagraph.S[caller_uid])
        except (ValueError, IndexError):
            # Hotkey not found or invalid UID
            priority = 0.0
        
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority
    


# ============================================================
# MAIN ENTRY POINT
# ============================================================
# This is where the miner starts.
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # AUTO-DISCOVER STUDENT MODEL
    # ============================================================
    # The miner will automatically find and load your model from
    # the student_models/ folder. Just put your model file there!
    #
    # If no student model is found, the miner will exit with an error.
    # ============================================================
    
    # Try to load student model from student_models/ folder
    model = load_student_model()
    
    # Raise error if no student model found
    if model is None:
        bt.logging.error("=" * 60)
        bt.logging.error("ERROR: No student model found!")
        bt.logging.error("=" * 60)
        bt.logging.error("To set up your model:")
        bt.logging.error("  1. Place your .h5 model file in miner_model/ directory")
        bt.logging.error("  2. Place your .csv data file in miner_model/ directory")
        bt.logging.error("  3. Make sure student_models/my_model.py exists (it should be in the repo)")
        bt.logging.error("  4. Run the miner again")
        bt.logging.error("")
        bt.logging.error("The my_model.py file will automatically find and load your files.")
        bt.logging.error("=" * 60)
        sys.exit(1)
    
    # Create and run the miner
    with Miner(model=model) as miner:
        bt.logging.info("Miner started. Waiting for challenges from validators...")
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
