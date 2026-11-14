"""
Scores a model on MMStar (MMStarDataset, MMStarCollator, test_mmstar; fp32, 5 sampled tokens per
answer, lenient regex check), once per seed. Run it on lusxvr/nanoVLM-222M to reproduce the
published 35.3%.

  --impl v01   The reference VisionLanguageModel in tests/reference (the baseline).
  --impl ours  Our tfv VisionLanguageModel.

Usage:
    uv run python scripts/eval_mmstar.py --impl v01 [--model lusxvr/nanoVLM-222M] [--seeds 0 1 2] [--limit 100]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repo root, so --impl v01 can import tests.reference

import torch

from tfv.config import EvalConfig
from tfv.data.loaders import get_mmstar_loader
from tfv.data.processors import get_tokenizer
from tfv.eval.mmstar import test_mmstar

def parse_args() -> argparse.Namespace:
    """
    Parses command-line overrides of EvalConfig.

    Returns:
        argparse.Namespace: impl, model, seeds, limit, batch_size, output.
    """
    cfg = EvalConfig()
    parser = argparse.ArgumentParser(description="MMStar evaluation")
    parser.add_argument("--impl", choices=["ours", "v01"], default="ours")
    parser.add_argument("--model", default=cfg.model_path, help="HF repo id or local save_pretrained directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.seeds))
    parser.add_argument("--limit", type=int, default=cfg.limit, help="Score only the first N questions")
    parser.add_argument("--batch-size", type=int, default=cfg.mmstar_batch_size)
    parser.add_argument("--output", default=None, help="Write the results to this JSON file")
    return parser.parse_args()

def load_model(impl: str, source: str) -> torch.nn.Module:
    """
    Args:
        impl: "ours" (tfv) or "v01" (reference code in tests/reference).
        source: HF repo id or local directory.
    Returns:
        torch.nn.Module: The VisionLanguageModel, loaded from source.
    """
    if impl == "v01":
        from tests.reference.vision_language_model import VisionLanguageModel
    else:
        from tfv.models.vlm import VisionLanguageModel
    return VisionLanguageModel.from_pretrained(source)

def main() -> None:
    """
    Loads the model and MMStar, scores every seed, and prints (and optionally saves) the results.
    """
    args = parse_args()
    eval_cfg = EvalConfig(model_path=args.model, seeds=tuple(args.seeds), limit=args.limit,
                          mmstar_batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.impl, eval_cfg.model_path).to(device)
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    loader = get_mmstar_loader(eval_cfg, model.cfg, limit=eval_cfg.limit)
    print(f"{args.impl} | {eval_cfg.model_path} | {len(loader.dataset)} questions | seeds {list(eval_cfg.seeds)} | {device}")

    scores = {}
    for seed in eval_cfg.seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        start = time.monotonic()
        scores[seed] = test_mmstar(model, tokenizer, loader, device)
        print(f"  seed {seed}: {scores[seed]:.4f} ({time.monotonic() - start:.0f}s)")

    values = list(scores.values())
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"MMStar: {mean:.4f} ± {std:.4f} over {len(values)} seeds (published nanoVLM-222M: 0.353)")

    if args.output:
        results = {"impl": args.impl, "model": eval_cfg.model_path, "questions": len(loader.dataset),
                   "scores": scores, "mean": mean, "std": std}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
