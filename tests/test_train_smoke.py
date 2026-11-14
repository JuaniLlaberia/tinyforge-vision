from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from tfv.config import EvalConfig, TrainConfig, VLMConfig
from tfv.data.collators import MMStarCollator
from tfv.data.loaders import ResumableSampler
from tfv.train.train import train

def tiny_train_config(tmp_path: Path, **overrides) -> TrainConfig:
    """
    A TrainConfig for CPU smoke runs: no compile, no W&B, no workers, checkpoints in tmp_path.

    Args:
        tmp_path: Pytest temporary directory.
        **overrides: Any TrainConfig fields to change.
    Returns:
        TrainConfig: The config.
    """
    settings = dict(batch_size=4, micro_batch_size=2, epochs=2, compile=False, log_wandb=False, num_workers=0,
                    eval_interval=2, checkpoint_interval=1, checkpoint_dir=str(tmp_path), run_name="smoke",
                    num_sample_generations=1, sample_max_new_tokens=3, log_interval=1)
    settings.update(overrides)
    return TrainConfig(**settings)

def mmstar_loader(tokenizer: PreTrainedTokenizerBase, samples: list[dict]) -> DataLoader:
    """
    A tiny MMStar-style loader built from the synthetic samples (answer = one letter + EOS).

    Args:
        tokenizer: The tokenizer.
        samples: Synthetic VQA samples reused for their images and questions.
    Returns:
        DataLoader: Batches of MMStarCollator output.
    """
    items = [{"image": s["image"], "text_data": s["text_data"], "answer": "B" + tokenizer.eos_token} for s in samples[:4]]
    return DataLoader(items, batch_size=2, collate_fn=MMStarCollator(tokenizer))

def test_resumable_sampler_replays_the_epoch() -> None:
    """
    Resuming at an offset yields exactly the rest of the same epoch's permutation, and each epoch
    has its own permutation.
    """
    sampler = ResumableSampler(10, seed=0)
    sampler.set_epoch(1)
    full_epoch = list(sampler)

    sampler.set_epoch(1, start_index=4)

    assert list(sampler) == full_epoch[4:]
    assert len(sampler) == 6
    sampler.set_epoch(2)
    assert sorted(sampler) == list(range(10)) and list(sampler) != full_epoch

def test_train_runs_and_resumes(tmp_path: Path, tiny_cfg: VLMConfig, tokenizer: PreTrainedTokenizerBase,
                                vqa_samples: list[dict]) -> None:
    """
    A 2-epoch run (2 steps per epoch, 2 micro-batches per step) finishes, writes its checkpoints,
    and a resumed run with one more epoch continues from step 4 to step 6.

    Args:
        tmp_path: Pytest temporary directory.
        tiny_cfg: Tiny config (see conftest).
        tokenizer: The tokenizer.
        vqa_samples: Synthetic samples (see conftest).
    """
    train_set, val_set = (vqa_samples + vqa_samples)[:8], vqa_samples[:4]
    loader = mmstar_loader(tokenizer, vqa_samples)

    result = train(tiny_train_config(tmp_path), tiny_cfg, EvalConfig(), datasets=(train_set, val_set), mmstar_loader=loader)

    run_dir = Path(result["run_dir"])
    assert result["step"] == 4
    assert (run_dir / "latest.pt").exists() and (run_dir / "final" / "model.safetensors").exists()
    assert torch.isfinite(torch.tensor(result["train_loss"])) and torch.isfinite(torch.tensor(result["val_loss"]))

    resumed = train(tiny_train_config(tmp_path, epochs=3, resume_from=str(run_dir / "latest.pt")), tiny_cfg,
                    EvalConfig(), datasets=(train_set, val_set), mmstar_loader=loader)

    assert resumed["step"] == 6

def test_overfit_one_batch(tmp_path: Path, tiny_cfg: VLMConfig, tokenizer: PreTrainedTokenizerBase,
                           vqa_samples: list[dict]) -> None:
    """
    The guide's first sanity check: repeating one batch, the loss falls far below its starting
    value (about ln(49152) = 10.8). If not, a module or the labels are wrong.

    Args:
        tmp_path: Pytest temporary directory.
        tiny_cfg: Tiny config (see conftest).
        tokenizer: The tokenizer.
        vqa_samples: Synthetic samples (see conftest).
    """
    cfg = tiny_train_config(tmp_path, batch_size=4, micro_batch_size=4, epochs=1000, max_steps=150, eval_interval=0,
                            checkpoint_interval=1000, overfit_batch=True, lr_mp=1e-2, lr_backbones=1e-2, log_interval=50)

    result = train(cfg, tiny_cfg, EvalConfig(), datasets=(vqa_samples[:4], vqa_samples[:4]),
                   mmstar_loader=mmstar_loader(tokenizer, vqa_samples))

    assert result["train_loss"] < 1.0
