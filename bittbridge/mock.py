import time
from typing import List, Optional
from unittest.mock import patch

import asyncio
import random
import bittensor as bt


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__(network=network)

        # Use chain_state, not subnet_exists() — the latter queries MagicMock substrate
        # and can return a false positive so create_subnet is skipped (Bittensor 10+).
        subtensor_state = self.chain_state["SubtensorModule"]
        if netuid not in subtensor_state["NetworksAdded"]:
            self.create_subnet(netuid)

        # Register ourself (the validator) as a neuron at uid=0
        if wallet is not None:
            self.force_register_neuron(
                netuid=netuid,
                hotkey_ss58=wallet.hotkey.ss58_address,
                coldkey_ss58=wallet.coldkey.ss58_address,
                balance=100000,
                stake=100000,
            )

        # Register n mock neurons who will be miners
        for i in range(1, n + 1):
            self.force_register_neuron(
                netuid=netuid,
                hotkey_ss58=f"miner-hotkey-{i}",
                coldkey_ss58="mock-coldkey",
                balance=100000,
                stake=100000,
            )

    def neuron_for_uid_lite(
        self, uid: int, netuid: int, block: Optional[int] = None
    ):
        """
        Build NeuronInfoLite from NeuronInfo without relying on parent (avoids stale
        NeuronInfoLite constructor vs NeuronInfo field mismatches across bittensor versions).
        """
        from bittensor.core.chain_data import NeuronInfoLite

        if uid is None:
            return NeuronInfoLite.get_null_neuron()

        if block is not None and self.block_number < block:
            raise Exception("Cannot query block in the future")
        else:
            block = self.block_number

        if netuid not in self.chain_state["SubtensorModule"]["NetworksAdded"]:
            return None

        neuron_info = self._neuron_subnet_exists(uid, netuid, block)
        if neuron_info is None:
            return None

        return NeuronInfoLite(
            hotkey=neuron_info.hotkey,
            coldkey=neuron_info.coldkey,
            uid=neuron_info.uid,
            netuid=neuron_info.netuid,
            active=neuron_info.active,
            stake=neuron_info.stake,
            stake_dict=neuron_info.stake_dict,
            total_stake=neuron_info.total_stake,
            emission=neuron_info.emission,
            incentive=neuron_info.incentive,
            consensus=neuron_info.consensus,
            validator_trust=neuron_info.validator_trust,
            dividends=neuron_info.dividends,
            last_update=neuron_info.last_update,
            validator_permit=neuron_info.validator_permit,
            prometheus_info=neuron_info.prometheus_info,
            axon_info=neuron_info.axon_info,
            is_null=neuron_info.is_null,
        )


class MockMetagraph(bt.Metagraph):
    def __init__(self, netuid=1, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, sync=False)

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = "127.0.0.0"
            axon.port = 8091

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")


class MockDendrite(bt.Dendrite):
    """
    Replaces a real bittensor network request with a mock request that just returns some static response for all axons that are passed and adds some random delay.
    """

    def __init__(self, wallet):
        import bittensor.utils.networking as networking

        # Dendrite.__init__ calls get_external_ip(); avoid HTTP in CI/tests
        with patch.object(networking, "get_external_ip", return_value="127.0.0.1"):
            super().__init__(wallet)

    async def forward(
        self,
        axons: List[bt.Axon],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ):
        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")

        async def query_all_axons(streaming: bool):
            """Queries all axons for responses."""

            async def single_axon_response(i, axon):
                """Queries a single axon for a response."""

                start_time = time.time()
                s = synapse.copy()
                # Attach some more required data so it looks real
                s = self.preprocess_synapse_for_request(axon, s, timeout)
                # We just want to mock the response, so we'll just fill in some data
                process_time = random.random()
                if process_time < timeout:
                    s.dendrite.process_time = str(time.time() - start_time)
                    # Update the status code and status message of the dendrite to match the axon
                    # TODO (developer): replace with your own expected synapse data
                    s.dummy_output = s.dummy_input * 2
                    s.dendrite.status_code = 200
                    s.dendrite.status_message = "OK"
                    synapse.dendrite.process_time = str(process_time)
                else:
                    s.dummy_output = 0
                    s.dendrite.status_code = 408
                    s.dendrite.status_message = "Timeout"
                    synapse.dendrite.process_time = str(timeout)

                # Return the updated synapse object after deserializing if requested
                if deserialize:
                    return s.deserialize()
                else:
                    return s

            return await asyncio.gather(
                *(
                    single_axon_response(i, target_axon)
                    for i, target_axon in enumerate(axons)
                )
            )

        return await query_all_axons(streaming)

    def __str__(self) -> str:
        """
        Returns a string representation of the Dendrite object.

        Returns:
            str: The string representation of the Dendrite object in the format "dendrite(<user_wallet_address>)".
        """
        return "MockDendrite({})".format(self.keypair.ss58_address)
