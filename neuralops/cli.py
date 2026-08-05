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
    parser.print_help()


if __name__ == "__main__":
    main()
