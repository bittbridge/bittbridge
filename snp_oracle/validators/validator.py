import argparse
import asyncio
from pathlib import Path

from snp_oracle.utils.config import config
from snp_oracle.validators.weight_setter import weight_setter


class Validator:
    def __init__(self, config):
        self.config = config
        full_path = Path(
            f"{self.config.logging.logging_dir}/{self.config.wallet.name}/{self.config.wallet.hotkey}/netuid{self.config.netuid}/validator"
        ).expanduser()
        full_path.mkdir(parents=True, exist_ok=True)
        self.config.full_path = str(full_path)

    async def main(self):
        loop = asyncio.get_event_loop()
        self.weight_setter = await weight_setter.create(config=self.config, loop=loop)
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Clean shutdown all tasks
            for task in asyncio.all_tasks():
                task.cancel()
            await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = config(parser, neuron_type="validator")
    validator = Validator(config)
    try:
        asyncio.run(validator.main())
    except KeyboardInterrupt:
        pass
