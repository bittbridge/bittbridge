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
    """
    Log one W&B row per miner, plus one dedicated ground-truth row per slot.

    Previously everything was crammed into a single wandb.log() call, which meant
    each row only had predictions for the subset of miners in that evaluation batch.
    The dashboard therefore could never find a ground_truth value alongside every
    miner's prediction — most prediction cells were null in ground-truth rows and
    vice-versa, causing the two time-series to appear ~6 hours apart on the chart.

    New structure — three separate wandb.log() calls per evaluation round:

      1. Ground-truth row  — keyed by timestamp only; no miner fields.
         The dashboard can always find a clean, complete actual-demand series
         without having to hunt through miner batch rows.

      2. Per-miner rows    — one wandb.log() per miner containing only that
         miner's prediction, reward, MA score, etc., plus the slot timestamp so
         the dashboard can align predictions to the correct point on the X axis.

      3. Aggregate miners_info dict row (kept for backward-compat with any
         existing dashboard panels that read miners_info.* keys).
    """
    try:
        # rewards may be list or numpy array; make it list
        if hasattr(rewards, "tolist"):
            rewards = rewards.tolist()

        lw = last_round_weights if isinstance(last_round_weights, dict) else {}

        def _weight_lookup(weights_by_uid, uid):
            try:
                if isinstance(weights_by_uid, dict):
                    return float(weights_by_uid[uid]) if uid in weights_by_uid else float('nan')
                if isinstance(weights_by_uid, (list, tuple)):
                    return float(weights_by_uid[uid]) if 0 <= uid < len(weights_by_uid) else float('nan')
            except Exception:
                pass
            return float('nan')

        # ── 1. Dedicated ground-truth row ────────────────────────────────────
        # Logged first and alone so the dashboard always has a complete,
        # unambiguous actual-demand series keyed purely by target slot.
        if ground_truth is not None and timestamp is not None:
            gt_row = {
                "ground_truth": float(ground_truth),
                "timestamp": timestamp,
            }
            if hasattr(bt.logging, "trace"):
                bt.logging.trace(f"Logging ground-truth row: {gt_row}")
            else:
                bt.logging.debug(f"Logging ground-truth row: {gt_row}")
            wandb.log(gt_row)

        # ── 2. Per-miner rows ────────────────────────────────────────────────
        # One wandb.log() per miner so every prediction row has exactly one
        # non-null prediction value plus the slot timestamp for X-axis alignment.
        miners_info = {}
        for uid, resp, rew in zip(miner_uids, responses, rewards):
            point_pred = getattr(resp, "prediction", None)
            ma = _weight_lookup(moving_average_scores, uid) if moving_average_scores is not None else float('nan')
            last_round_weight = _weight_lookup(lw, uid)

            # Accumulate for the backward-compat aggregate row (step 3)
            miners_info[str(uid)] = {
                "miner_hotkey": hotkeys.get(uid),
                "miner_point_prediction": point_pred,
                "miner_reward": float(rew) if rew is not None else None,
                "miner_moving_average_score": ma,
                "miner_last_round_weight": last_round_weight,
            }

            # Per-miner row: only this miner's fields + slot timestamp
            miner_row = {
                "timestamp": timestamp,
                f"miners_info.{uid}.miner_hotkey": hotkeys.get(uid),
                f"miners_info.{uid}.miner_moving_average_score": ma,
                f"miners_info.{uid}.miner_last_round_weight": last_round_weight,
            }
            if point_pred is not None:
                miner_row[f"miners_info.{uid}.miner_point_prediction"] = float(point_pred)
                miner_row[f"miner_{uid}_prediction"] = float(point_pred)
            if rew is not None:
                miner_row[f"miners_info.{uid}.miner_reward"] = float(rew)
                miner_row[f"miner_{uid}_reward"] = float(rew)
            if point_pred is not None and ground_truth is not None:
                miner_row[f"miner_{uid}_error"] = abs(float(point_pred) - float(ground_truth))

            if hasattr(bt.logging, "trace"):
                bt.logging.trace(f"Logging miner row uid={uid}: {miner_row}")
            wandb.log(miner_row)

        # ── 3. Backward-compat aggregate row ────────────────────────────────
        # Keeps the miners_info nested dict that any existing W&B panels rely on.
        if miners_info:
            agg_row = {
                "miners_info": miners_info,
                "timestamp": timestamp,
            }
            wandb.log(agg_row)

    except Exception as e:
        bt.logging.error(f"Failed to log to wandb: {str(e)}")
        bt.logging.error("Full error: ", exc_info=True)
