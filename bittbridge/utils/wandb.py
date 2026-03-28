import os
import bittensor as bt
import wandb
from bittbridge import __version__

#WANDB_ENTITY = "bittbridge_uconn"
# TODO: change to actual entity
WANDB_ENTITY = "dwtest"

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

    # Build a stable, human-friendly run name:
    #   - validator-<uid>-<version> when we know uid
    #   - otherwise validator-<last6_of_hotkey>-<version>
    fallback = (hotkey[-6:] if isinstance(hotkey, str) and len(hotkey) >= 6 else "na")
    name_uid = uid if uid is not None else fallback
    run_name = f"validator-{name_uid}-{__version__}"

    # Init W&B — use resume="never" so a deleted run on the server doesn't cause
    # init to hang/timeout; each validator start gets a fresh run.
    wandb.init(
        project=f"sn{netuid}-validators",
        entity=WANDB_ENTITY,
        config={
            "hotkey": hotkey,
            "uid": uid,
            "subnet_version": __version__,
        },
        name=run_name,
        resume="never",
        dir=getattr(getattr(getattr(self, "config", None), "neuron", None), "full_path", None),
        reinit="default",
        settings=wandb.Settings(init_timeout=120),
    )


def log_wandb(
    responses,
    rewards,
    miner_uids,
    hotkeys,
    moving_average_scores,
    last_round_weights=None,
    ground_truth=None,
    timestamp=None,
):
    try:
        # rewards may be list or numpy array; make it list
        if hasattr(rewards, "tolist"):
            rewards = rewards.tolist()

        lw = last_round_weights if isinstance(last_round_weights, dict) else {}

        def _weight_lookup(weights_by_uid, uid):
            # supports dict {uid->val} or list/tuple indexed by uid
            try:
                if isinstance(weights_by_uid, dict):
                    return float(weights_by_uid[uid]) if uid in weights_by_uid else float('nan')
                if isinstance(weights_by_uid, (list, tuple)):
                    return float(weights_by_uid[uid]) if 0 <= uid < len(weights_by_uid) else float('nan')
            except Exception:
                pass
            return float('nan')

        miners_info = {}
        for uid, resp, rew in zip(miner_uids, responses, rewards):
            point_pred = getattr(resp, "prediction", None)
            #interval = getattr(resp, "interval", None)

            ma = _weight_lookup(moving_average_scores, uid) if moving_average_scores is not None else float("nan")
            miners_info[str(uid)] = { # cast key to string for nicer W&B tables
                "miner_hotkey": hotkeys.get(uid),
                "miner_point_prediction": point_pred,
                "miner_reward": float(rew) if rew is not None else None,
                "miner_moving_average_score": ma,
                "miner_last_round_weight": _weight_lookup(lw, uid),
            }

        if not miners_info:
            return

        wandb_val_log = {
            "miners_info": miners_info,
            "ground_truth": float(ground_truth) if ground_truth is not None else None,
            "timestamp": timestamp if timestamp is not None else None,
        }

        # Flatten metrics for plotting
        for uid, resp, rew in zip(miner_uids, responses, rewards):
            point_pred = getattr(resp, "prediction", None)

            # Always log prediction if available
            if point_pred is not None:
                wandb_val_log[f"miner_{uid}_prediction"] = float(point_pred)

            # Always log reward if available
            if rew is not None:
                wandb_val_log[f"miner_{uid}_reward"] = float(rew)

            # Log error only if possible
            if point_pred is not None and ground_truth is not None:
                error = abs(point_pred - ground_truth)
                wandb_val_log[f"miner_{uid}_error"] = float(error)

        # Debug logging
        if hasattr(bt.logging, "trace"):
            bt.logging.trace(f"Attempting to log data to wandb: {wandb_val_log}")
        else:
            bt.logging.debug(f"Attempting to log data to wandb: {wandb_val_log}")

        # Send to W&B
        wandb.log(wandb_val_log)

    except Exception as e:
        bt.logging.error(f"Failed to log to wandb: {str(e)}")
        bt.logging.error("Full error: ", exc_info=True)
