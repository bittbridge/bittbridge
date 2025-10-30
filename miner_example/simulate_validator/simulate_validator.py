import asyncio
import datetime as dt
import bittensor as bt

import bittbridge
from bittbridge.bittbridge.protocol import Challenge


async def main():
    # Create a local dendrite to call local axon
    dendrite = bt.dendrite(wallet=bt.MockWallet())
    # Current timestamp in ISO format
    timestamp = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    synapse = Challenge(timestamp=timestamp)

    # Local call: in a real setup you'd use the miner's axon address; here we rely on local environment/mocks
    try:
        response = await dendrite.forward(
            axons=[bt.axon()],  # mock local axon for example purposes
            synapse=synapse,
        )
        if not response:
            print("No response received")
            return
        res = response[0]
        print({
            "timestamp": timestamp,
            "prediction": res.prediction,
            "interval": res.interval,
        })
    finally:
        await dendrite.close()


if __name__ == "__main__":
    asyncio.run(main())


