"""Automated pipeline orchestrator: handles end-to-end execution from preprocessing to evaluation."""

import argparse
import subprocess
import sys


def run_command(cmd: str):
    print(f"\n[RUNNING]: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[ERROR]: Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline runner.")
    parser.add_argument(
        "--mode",
        choices=["all", "train", "evaluate", "test", "ism"],
        default="all",
        help="Pipeline phase to execute",
    )
    args = parser.parse_args()

    if args.mode in ["all", "test"]:
        run_command("pytest")

    if args.mode in ["all", "train"]:
        run_command("python scripts/train.py --config configs/config.yaml")

    if args.mode in ["all", "evaluate"]:
        run_command(
            "python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/transformer_best.pt"
        )


if __name__ == "__main__":
    main()
