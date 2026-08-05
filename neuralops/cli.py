"""Command-line entry point."""

from __future__ import annotations

import argparse

from neuralops import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralops",
        description="Reproducible log-sequence anomaly detection",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="inspect the local runtime")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "doctor":
        from neuralops.device import resolve_device

        print(f"neuralops={__version__} device={resolve_device()}")
        return
    parser.print_help()


if __name__ == "__main__":
    main()

