# TinyForge-Vision

A small VLM (SigLIP-B/16-224 + SmolLM2-135M, 222M params) that reproduces **nanoVLM v0.1**
(tag `v0.1`, commit `6ba9082`; checkpoint `lusxvr/nanoVLM-222M`; 35.3% on MMStar).
`docs/GUIDE.md` is the plan; Part 1 (shape-trace notebook) is skipped.

## Who writes what

This repo works like a course assignment: the owner writes the core VLM blocks, Claude
provides the surrounding code.

| Owner writes (Claude only guides, or writes it if explicitly asked) | Claude writes |
| --- | --- |
| `tfv/models/projector.py`: pixel shuffle + projector | Borrowed nanoVLM v0.1 code (backbones, datasets, processors, eval) |
| `tfv/models/vlm.py`: combine step, `forward`, `generate` | Training loop adapted from the owner's TinyForge (`train/sft.py`) |
| `tfv/train/optim.py`: parameter groups, freezing | Config, tests, scripts, Colab notebook |
| `VQACollator` in `tfv/data/collators.py`: written together | |

Stubs for the owner's pieces have full docstrings and `raise NotImplementedError`. When the
owner asks for help, give guidance and hints first; never refuse to write the code if they
explicitly ask for it.

## Rules

- **Git:** the owner makes all commits and pushes. Never commit or push.
- **The owner reviews every file**, so keep changes focused and explain them.
- **Borrowed files** start with a header naming the repo, tag/commit, and source file. The only
  permitted changes are: one blank line between definitions (nanoVLM uses two), double quotes, added
  docstrings and type hints, and import paths adapted to `tfv.`. No logic changes. The one
  approved exception: the attention branch in `tfv/models/language_model.py`, marked
  `[tinyforge-vision]`, which merges the causal and padding masks on every device (recent PyTorch
  rejects `attn_mask` together with `is_causal=True`, on CUDA too).
- **The eval (`tfv/eval/`) is never changed logically**, so scores stay comparable with 35.3%.
- **`VLMConfig` keeps every v0.1 field name**: `VisionLanguageModel.from_pretrained` loads the
  222M checkpoint with `VLMConfig(**config.json)`. The VLM keeps the attribute names
  `vision_encoder`, `decoder`, `MP` so the checkpoint's `state_dict` loads.
- **All parameters live in `tfv/config.py`** (dataclasses); no magic numbers in code.
- The repo must be self-contained: the v0.1 reference code used by tests lives in
  `tests/reference/`, not in an external clone.

## Style

- 4-space indents, one blank line between top-level definitions, double quotes unless the string
  needs single ones, modern type hints (`list[int]`, `X | None`), Python 3.12.
- Every function, method, and class has a docstring in this format:

```python
def fn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    One-line summary, then any detail worth knowing.

    Args:
        x: What it is, with shape if a tensor, e.g. [B, N, D].
        k: What it controls.
    Returns:
        torch.Tensor: What comes back, with shape.
    """
```

## Commands

- `uv sync`: install (Python 3.12).
- `uv run pytest -m "not slow"`: fast CPU tests. `uv run pytest`: also the tests that download
  the 222M checkpoint or Cauldron samples.
- `uv run python scripts/benchmark_prepare.py`, then `scripts/prepare_cauldron.py`: one-time Cauldron
  preprocessing to `Juanillaberia/tinyforge-vision-cauldron-224` (resumable; needs `hf auth login`).
- `uv run python scripts/eval_mmstar.py --impl v01|ours [--model ...] [--seeds 0 1 2]`: MMStar with
  v0.1's protocol (~4-5 min per seed on the Mac CPU).
- `uv run python -m tfv.train.train --set max_steps=100 --set log_wandb=False`: training from the CLI.
- Training runs on Colab via `notebooks/train_colab.ipynb` (checks, full run, resume, eval).
