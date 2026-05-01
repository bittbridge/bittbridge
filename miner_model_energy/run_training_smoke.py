from __future__ import annotations

import argparse

from miner_model_energy.ml_config import load_model_config
from miner_model_energy.pipeline import predict_single_test_row, print_actual_vs_predicted_plotext, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="model_params.yaml",
        help="Path to model config yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=["linear", "cart", "rnn", "lstm"],
        help="Model to train in smoke mode.",
    )
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    result = train_model(args.model, cfg)
    if cfg.training.get("show_training_progress", True):
        print_actual_vs_predicted_plotext(result, args.model)
    one_pred = predict_single_test_row(result)
    d = result.durations_sec
    print(
        f"model={args.model} "
        f"total_sec={d.get('total_sec', 0):.3f} "
        f"prepare_sec={d.get('prepare_data_sec', 0):.3f} "
        f"fit_sec={d.get('split_and_fit_sec', 0):.3f} "
        f"train_rmse={result.metrics['train']['rmse']:.3f} "
        f"train_mae={result.metrics['train']['mae']:.3f} "
        f"train_mape={result.metrics['train']['mape']:.3f}% "
        f"train_r2={result.metrics['train']['r2']:.5f} "
        f"val_rmse={result.metrics['validation']['rmse']:.3f} "
        f"val_mae={result.metrics['validation']['mae']:.3f} "
        f"val_mape={result.metrics['validation']['mape']:.3f}% "
        f"val_r2={result.metrics['validation']['r2']:.5f} "
        f"test_pred={one_pred:.3f}"
    )


if __name__ == "__main__":
    main()

