"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neuralops import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralops",
        description="Reproducible log-sequence anomaly detection",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="inspect the local runtime")

    download = subparsers.add_parser("download-hdfs", help="download verified Loghub HDFS v1")
    download.add_argument("--target", type=Path, default=Path("data/raw/hdfs-v1"))

    prepare = subparsers.add_parser("prepare", help="prepare leakage-safe dataset splits")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--raw-dir", type=Path)
    prepare.add_argument("--output-dir", type=Path)

    baseline = subparsers.add_parser("train-baseline", help="train TF-IDF logistic baseline")
    baseline.add_argument("--config", type=Path, required=True)
    baseline.add_argument("--processed-dir", type=Path)
    baseline.add_argument("--artifact", type=Path, required=True)

    train = subparsers.add_parser("train", help="train packed PyTorch GRU")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--processed-dir", type=Path)
    train.add_argument("--artifact", type=Path, required=True)
    train.add_argument("--device")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a locked GRU artifact")
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--processed-dir", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--device", default="auto")

    benchmark = subparsers.add_parser("benchmark", help="measure GRU inference performance")
    benchmark.add_argument("--artifact", type=Path, required=True)
    benchmark.add_argument("--processed-dir", type=Path, required=True)
    benchmark.add_argument("--iterations", type=int, default=100)
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--batch-size", type=int, default=32)
    benchmark.add_argument("--device", default="auto")
    return parser


def _split_ratios(config: dict[str, Any]) -> tuple[float, float, float]:
    from neuralops.config import nested

    return (
        float(nested(config, "data", "split", "train")),
        float(nested(config, "data", "split", "validation")),
        float(nested(config, "data", "split", "test")),
    )


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    from neuralops.config import load_config, nested
    from neuralops.data.hdfs import load_hdfs_records
    from neuralops.data.prepare import prepare_dataset

    config = load_config(args.config)
    source = str(nested(config, "task", "source"))
    if source != "loghub-hdfs-v1":
        raise ValueError("M1 preparation currently supports task.source=loghub-hdfs-v1")
    raw_dir = args.raw_dir or Path(str(nested(config, "data", "raw_dir")))
    output_dir = args.output_dir or Path(str(nested(config, "data", "processed_dir")))
    records = load_hdfs_records(raw_dir)
    return prepare_dataset(
        records,
        output_dir,
        source=source,
        profile=str(nested(config, "task", "profile")),
        seed=int(config["seed"]),
        ratios=_split_ratios(config),
        deduplicate=bool(config["data"].get("deduplicate", True)),
    )


def _train_baseline(args: argparse.Namespace) -> dict[str, Any]:
    from neuralops.baseline.model import train_baseline
    from neuralops.config import load_config, nested

    config = load_config(args.config)
    processed_dir = args.processed_dir or Path(str(nested(config, "data", "processed_dir")))
    minimum_coverage = float(nested(config, "evaluation", "minimum_selective_coverage"))
    return train_baseline(
        processed_dir,
        args.artifact,
        seed=int(config["seed"]),
        minimum_coverage=minimum_coverage,
    )


def _train_gru(args: argparse.Namespace) -> dict[str, Any]:
    from neuralops.config import load_config, nested
    from neuralops.data.io import read_json
    from neuralops.modeling.gru import GRUConfig
    from neuralops.modeling.training import TrainingSettings, train_gru

    config = load_config(args.config)
    processed_dir = args.processed_dir or Path(str(nested(config, "data", "processed_dir")))
    vocabulary = read_json(processed_dir / "vocabulary.json")
    if not isinstance(vocabulary, dict):
        raise ValueError("Prepared vocabulary must be an object")
    model_config = GRUConfig(
        vocabulary_size=len(vocabulary),
        embedding_dim=int(nested(config, "model", "embedding_dim")),
        hidden_dim=int(nested(config, "model", "hidden_dim")),
        layers=int(nested(config, "model", "layers")),
        bidirectional=bool(nested(config, "model", "bidirectional")),
        dropout=float(nested(config, "model", "dropout")),
    )
    settings = TrainingSettings(
        seed=int(config["seed"]),
        batch_size=int(nested(config, "training", "batch_size")),
        epochs=int(nested(config, "training", "epochs")),
        patience=int(nested(config, "training", "patience")),
        learning_rate=float(nested(config, "training", "learning_rate")),
        weight_decay=float(nested(config, "training", "weight_decay")),
        gradient_clip_norm=float(nested(config, "training", "gradient_clip_norm")),
        max_sequence_length=int(nested(config, "model", "max_sequence_length")),
        device=args.device or str(nested(config, "training", "device")),
        minimum_coverage=float(nested(config, "evaluation", "minimum_selective_coverage")),
    )
    return train_gru(
        processed_dir,
        args.artifact,
        model_config=model_config,
        settings=settings,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "doctor":
        from neuralops.device import resolve_device

        print(f"neuralops={__version__} device={resolve_device()}")
        return
    if args.command == "download-hdfs":
        from neuralops.data.hdfs import download_hdfs_v1

        print(download_hdfs_v1(args.target))
        return
    if args.command == "prepare":
        print(json.dumps(_prepare(args), indent=2, sort_keys=True))
        return
    if args.command == "train-baseline":
        print(json.dumps(_train_baseline(args), indent=2, sort_keys=True))
        return
    if args.command == "train":
        print(json.dumps(_train_gru(args), indent=2, sort_keys=True))
        return
    if args.command == "evaluate":
        from neuralops.modeling.training import evaluate_artifact

        result = evaluate_artifact(
            args.artifact,
            args.processed_dir,
            split=args.split,
            requested_device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "benchmark":
        from neuralops.modeling.benchmark import benchmark_artifact

        result = benchmark_artifact(
            args.artifact,
            args.processed_dir,
            iterations=args.iterations,
            warmup=args.warmup,
            batch_size=args.batch_size,
            requested_device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
