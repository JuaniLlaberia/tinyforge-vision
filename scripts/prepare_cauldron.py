"""
One-time preprocessing of the Cauldron for training: for each of the 45 training subsets, keeps the
first image (resized to 224x224, JPEG) and the first QA pair, and pushes the subset to a private
HF dataset repo (PrepareConfig.target_repo_id). Subsets already in the repo are skipped, so the
script can be rerun after a disconnect.

Requires an HF token with write access (`hf auth login`).

Usage:
    uv run python scripts/prepare_cauldron.py [--subsets ai2d docvqa] [--num-proc 8] [--repo-id user/name]
"""
import argparse
import os
import time

from tfv.config import PrepareConfig
from tfv.data.prepare import finished_subsets, prepare_subset

def parse_args() -> argparse.Namespace:
    """
    Parses command-line overrides of PrepareConfig.

    Returns:
        argparse.Namespace: subsets, num_proc and repo_id.
    """
    cfg = PrepareConfig()
    parser = argparse.ArgumentParser(description="Preprocess the Cauldron and push it to the HF Hub")
    parser.add_argument("--subsets", nargs="+", default=list(cfg.subsets))
    parser.add_argument("--num-proc", type=int, default=cfg.num_proc or os.cpu_count())
    parser.add_argument("--repo-id", type=str, default=cfg.target_repo_id)
    return parser.parse_args()

def main() -> None:
    """
    Processes and pushes every requested subset that is not in the target repo yet.
    """
    args = parse_args()
    cfg = PrepareConfig(target_repo_id=args.repo_id)
    done = finished_subsets(cfg.target_repo_id)
    todo = [name for name in args.subsets if name not in done]
    print(f"Target: {cfg.target_repo_id} | {len(done)} subsets already pushed, {len(todo)} to go | {args.num_proc} processes")

    start = time.monotonic()
    for index, name in enumerate(todo, start=1):
        stats = prepare_subset(name, cfg, args.num_proc)
        elapsed = (time.monotonic() - start) / 60
        print(f"[{index}/{len(todo)}] {name}: {stats['rows']:,} rows, {stats['missing_images']} without image, "
              f"{stats['seconds'] / 60:.1f} min (total {elapsed:.0f} min)")

    print("Done. Train with TrainConfig.train_dataset_path =", cfg.target_repo_id)

if __name__ == "__main__":
    main()
