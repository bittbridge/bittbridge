import os

import bittensor as bt
import wandb

WANDB_ENTITY = "faeze-safari"


def setup_wandb(self) -> None:
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None:
        wandb.init(
            project=f"sn{self.config.netuid}-validators",
            entity=WANDB_ENTITY,
            config={
                "hotkey": getattr(self.wallet.hotkey, "ss58_address", None),
                "uid": getattr(self, "my_uid", None),
                "subnet_version": getattr(self, "__class__", type("X",(object,),{})).__name__,
            },
            name=f"validator-{getattr(self, 'my_uid', 'na')}",
            resume="auto",
            dir=getattr(self.config.neuron, "full_path", None),
            reinit=True,
        )
    else:
        bt.logging.error("WANDB_API_KEY not found in environment variables.")


def log_wandb(responses, rewards, miner_uids, hotkeys, moving_average_scores):
    """Mirror Precog’s logging dict shape, but keep it robust to missing pieces."""
    try:
        # rewards may be list or numpy array; make it list
        if hasattr(rewards, "tolist"):
            rewards = rewards.tolist()

        miners_info = {}
        for uid, resp, rew in zip(miner_uids, responses, rewards):
            # Precog expects resp.prediction / resp.interval; fall back if your synapse differs.
            point_pred = getattr(resp, "prediction", None)
            interval    = getattr(resp, "interval", None)

            miners_info[uid] = {
                "miner_hotkey": hotkeys.get(uid),
                "miner_point_prediction": point_pred,
                "miner_interval_prediction": interval,
                "miner_reward": float(rew) if rew is not None else None,
                "miner_moving_average": float(moving_average_scores.get(uid, 0.0)),
            }

        wandb.log({"miners_info": miners_info})
    except Exception as e:
        bt.logging.error(f"Failed to log to wandb: {e}", exc_info=True)
