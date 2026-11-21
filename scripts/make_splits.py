"""Create the shared stratified 80/20 train/val split for both tasks.

Run ONCE (one team member), then commit splits/ so everyone uses identical splits.
Usage: python scripts/make_splits.py [--overwrite]
"""
import _path  # noqa: F401
import argparse

from src.common.config import TASKS
from src.common.split import make_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true",
                    help="Recreate splits (invalidates everyone's earlier val results!)")
    args = ap.parse_args()
    for task in TASKS:
        make_split(task, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
