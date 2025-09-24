import os
import bittensor as bt
import wandb
from bittbridge import __version__

WANDB_ENTITY = "bittbridge_uconn" 

def setup_wandb(self) -> None:
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        bt.logging.error("WANDB_API_KEY not found in environment variables.")
        return

    # Gather fields safely
    netuid = getattr(getattr(self, "config", None), "netuid", "na")
    hotkey = getattr(getattr(self.wallet, "hotkey", None), "ss58_address", None)

    # Prefer self.my_uid; if missing, try to infer from metagraph by hotkey match
    uid = getattr(self, "my_uid", None)
    if uid is None and hotkey is not None:
        try:
            uid = self.metagraph.hotkeys.index(hotkey)
        except Exception:
            uid = None  # keep None if not found

    # Build a stable, human-friendly name:
    #   - validator-<uid>-<version> when we know uid
    #   - otherwise validator-<last6_of_hotkey>-<version>
    from bittbridge import __version__
    fallback = (hotkey[-6:] if isinstance(hotkey, str) and len(hotkey) >= 6 else "na")
    name_uid = uid if uid is not None else fallback
    run_name = f"validator-{name_uid}-{__version__}"

    # Init W&B
    wandb.init(
        project=f"sn{self.config.netuid}-validators",
        entity=WANDB_ENTITY,
        config={
            "hotkey": hotkey,
            "uid": uid,
            "subnet_version": __version__,
        },
        name=run_name,
        resume="auto",
        dir=getattr(getattr(getattr(self, "config", None), "neuron", None), "full_path", None),
        reinit='default',
    )

def log_wandb(responses, rewards, miner_uids, hotkeys, moving_average_scores):
    try:
        # rewards may be list or numpy array; make it list
        if hasattr(rewards, "tolist"):
            rewards = rewards.tolist()

        miners_info = {}
        for uid, resp, rew in zip(miner_uids, responses, rewards):
            point_pred = getattr(resp, "prediction", None)
            interval   = getattr(resp, "interval", None)
            miners_info[str(uid)] = {  # cast key to string for nicer W&B tables
                "miner_hotkey": hotkeys.get(uid),
                "miner_point_prediction": point_pred,
                "miner_interval_prediction": interval,
                "miner_reward": float(rew) if rew is not None else None,
                "miner_moving_average": float(moving_average_scores.get(uid, 0.0)),
            }

        if not miners_info:
            return

        wandb_val_log = {"miners_info": miners_info}

        # Pre-log trace of the exact payload (fallback to debug if trace is absent)
        if hasattr(bt.logging, "trace"):
            bt.logging.trace(f"Attempting to log data to wandb: {wandb_val_log}")
        else:
            bt.logging.debug(f"Attempting to log data to wandb: {wandb_val_log}")

        wandb.log(wandb_val_log)

    except Exception as e:
        bt.logging.error(f"Failed to log to wandb: {str(e)}")
        bt.logging.error("Full error: ", exc_info=True)
