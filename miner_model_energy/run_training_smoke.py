from __future__ import annotations

import argparse

from miner_model_energy.ml_config import load_model_config
from miner_model_energy.pipeline import predict_single_test_row, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="miner_model_energy/model_params.yaml",
        help="Path to model config yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=["linear", "cart", "lstm"],
        help="Model to train in smoke mode.",
    )
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    result = train_model(args.model, cfg)
    one_pred = predict_single_test_row(result)
    print(
        f"model={args.model} rmse={result.metrics['rmse']:.3f} "
        f"mae={result.metrics['mae']:.3f} mape={result.metrics['mape']:.3f}% "
        f"test_pred={one_pred:.3f}"
    )


if __name__ == "__main__":
    main()

