"""Entry point: python3 -m carbide_modules [--data <path>]

Mirrors carbide.py's original startup order: parse --data, load the
dataset once, then enter the main menu loop.
"""
import argparse

from . import dataset
from .cli import main


def _parse_args():
    parser = argparse.ArgumentParser(description="Carbide Interactive CLI")
    parser.add_argument("--data", type=str, default=None, help="Path to custom dataset file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.data:
        dataset.load_dataset(args.data)
    else:
        dataset.load_dataset()
    main()
