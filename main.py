"""Command-line entry point for fraud precursor analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import FraudSequenceDataset
from src.evaluate import classification_metrics, predict_probabilities
from src.model import FraudPrecursorLSTM
from src.pipeline import prepare_data
from src.predict import predict_sequence
from src.reporting import dataset_summary, write_final_report
from src.train import train_model
from src.utils import ensure_output_dirs, save_json, set_seed, setup_logging
from src.visualization import (
    generate_eda_plots,
    plot_evaluation,
    plot_training_history,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["eda", "train", "evaluate", "predict"], required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", help="CSV containing one prior-transaction sequence")
    return parser.parse_args()


def make_loader(sequence_set, config: Config, shuffle: bool = False) -> DataLoader:
    dataset = FraudSequenceDataset(sequence_set.features, sequence_set.labels)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    paths = ensure_output_dirs(config.outputs_dir)
    setup_logging(paths["logs"] / f"{args.mode}.log")
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    preprocessor_path = paths["models"] / "preprocessor.joblib"
    if args.mode == "predict":
        if not args.input:
            raise ValueError("--input is required in predict mode")
        checkpoint = torch.load(config.model_save_path, map_location=device)
        preprocessor = joblib.load(preprocessor_path)
        probability = predict_sequence(
            args.input, config, preprocessor, checkpoint, device
        )
        label = "fraud-preceding" if probability >= config.threshold else "normal-preceding"
        print(f"Fraud-preceding probability: {probability:.6f}")
        print(f"Sequence classification: {label}")
        return

    if args.mode == "evaluate" and not preprocessor_path.exists():
        raise FileNotFoundError("Training preprocessor not found; run train mode first")
    fitted_preprocessor = (
        joblib.load(preprocessor_path) if args.mode == "evaluate" else None
    )
    prepared = prepare_data(config, fitted_preprocessor)
    summary = dataset_summary(
        prepared.frame,
        config.user_column,
        config.target_column,
        config.sequence_length,
    )
    save_json(summary, paths["reports"] / "dataset_summary.json")
    save_json(prepared.splits.users, paths["reports"] / "split_users.json")

    if args.mode == "eda":
        generate_eda_plots(
            prepared.frame,
            paths["plots"],
            config.user_column,
            config.target_column,
        )
        save_json(
            {
                name: {
                    "total_windows": len(values.labels),
                    "fraud_preceding_windows": int(values.labels.sum()),
                    "normal_preceding_windows": int(len(values.labels) - values.labels.sum()),
                }
                for name, values in prepared.sequences.items()
            },
            paths["reports"] / "sequence_feasibility.json",
        )
        LOGGER.info("EDA outputs saved under %s", config.outputs_dir)
        return

    if args.mode == "train":
        joblib.dump(prepared.preprocessor, preprocessor_path)
        train_sequences = prepared.sequences["train"]
        input_size = train_sequences.features.shape[-1]
        model = FraudPrecursorLSTM(
            input_size,
            config.hidden_size,
            config.num_layers,
            config.dropout,
        )
        model, history, _ = train_model(
            model=model,
            train_loader=make_loader(train_sequences, config, shuffle=True),
            validation_loader=make_loader(prepared.sequences["validation"], config),
            train_labels=train_sequences.labels,
            device=device,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            patience=config.patience,
            gradient_clip=config.gradient_clip,
            checkpoint_path=config.model_save_path,
            checkpoint_metadata={
                "input_size": input_size,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "sequence_length": config.sequence_length,
                "feature_names": prepared.preprocessor.feature_names,
            },
            threshold=config.threshold,
        )
        save_json(history, paths["reports"] / "training_history.json")
        plot_training_history(history, paths["plots"] / "training_validation_loss.png")
    else:
        if not Path(config.model_save_path).exists():
            raise FileNotFoundError("Train the model before running evaluate mode")
        checkpoint = torch.load(config.model_save_path, map_location=device)
        model = FraudPrecursorLSTM(
            checkpoint["input_size"],
            checkpoint["hidden_size"],
            checkpoint["num_layers"],
            checkpoint["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

    all_metrics = {}
    for split_name, sequences in prepared.sequences.items():
        labels, probabilities = predict_probabilities(
            model, make_loader(sequences, config), device
        )
        metrics = classification_metrics(labels, probabilities, config.threshold)
        all_metrics[split_name] = metrics
        plot_evaluation(
            labels,
            probabilities,
            metrics,
            paths["plots"],
            split_name,
            config.threshold,
        )
    save_json(all_metrics, paths["reports"] / "metrics.json")
    write_final_report(
        paths["reports"] / "final_report.md",
        config,
        summary,
        prepared.splits.users,
        all_metrics,
    )
    LOGGER.info("Metrics and report saved under %s", config.outputs_dir)


if __name__ == "__main__":
    main()
