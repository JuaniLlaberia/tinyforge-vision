# Training loop adapted from TinyForge (train/sft.py).
# Optimizer, LR schedule and evaluation adapted from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082, train.py.
import argparse
import ast
import dataclasses
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from tfv.config import EvalConfig, TrainConfig, VLMConfig
from tfv.data.loaders import get_mmstar_loader, get_train_val_datasets, get_train_val_loaders
from tfv.data.processors import get_tokenizer
from tfv.eval.mmstar import test_mmstar
from tfv.models.vlm import VisionLanguageModel
from tfv.train.checkpoint import detect_precision, load_checkpoint, resolve_checkpoint_source, save_checkpoint
from tfv.train.optim import build_param_groups
from tfv.train.scheduler import build_scheduler

IGNORE_INDEX = -100

def get_run_name(train_cfg: TrainConfig, num_train_samples: int) -> str:
    """
    Builds a descriptive run name from the config.

    Args:
        train_cfg: Training config.
        num_train_samples: Size of the training split.
    Returns:
        str: e.g. "tfv_1655332samples_bs256_ep5_lr0.0001-0.002_1114".
    """
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    return f"tfv_{num_train_samples}samples_bs{train_cfg.batch_size}_ep{train_cfg.epochs}_{learning_rate}_{time.strftime('%m%d')}"

def loss_sum_from_mean(loss: torch.Tensor, num_targets: int) -> torch.Tensor:
    """
    Turns the model's mean cross-entropy over a micro-batch's targets back into a sum. The caller
    divides by the target count of the whole optimizer step, so every answer token weighs the same
    whichever micro-batch it is in. Reusing the forward's loss avoids a second fp32 copy of the logits.

    Args:
        loss: Mean loss returned by VisionLanguageModel.forward (ignore_index=-100).
        num_targets: Number of non-ignored labels in the micro-batch.
    Returns:
        torch.Tensor: Scalar loss sum (0 if the micro-batch has no targets, where the mean is nan).
    """
    if num_targets == 0:
        return loss.new_zeros(())
    return loss.float() * num_targets

def count_targets(labels: torch.Tensor) -> int:
    """
    Args:
        labels: Shifted labels, [B, T].
    Returns:
        int: Number of positions that contribute to the loss.
    """
    return int((labels != IGNORE_INDEX).sum())

def count_tokens(batch: dict[str, torch.Tensor], vlm_cfg: VLMConfig) -> int:
    """
    Tokens processed in a batch: real text tokens plus image tokens.

    Args:
        batch: Collated batch with "attention_mask" and "image".
        vlm_cfg: Uses vit_patch_size, mp_pixel_shuffle_factor.
    Returns:
        int: Token count.
    """
    images = batch["image"]
    image_tokens = (images.shape[2] // vlm_cfg.vit_patch_size) ** 2 // vlm_cfg.mp_pixel_shuffle_factor ** 2
    return int(batch["attention_mask"].sum()) + images.shape[0] * image_tokens

def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """
    Args:
        batch: Collated batch.
        device: Target device.
    Returns:
        dict[str, torch.Tensor]: The batch on device.
    """
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

@torch.no_grad()
def evaluate_val_loss(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype, amp_enabled: bool
) -> float:
    """
    Token-weighted mean cross-entropy over the whole validation split (under the training autocast).

    Args:
        model: The VLM (compiled or not).
        loader: Validation loader.
        device: Device to run on.
        amp_dtype: Autocast dtype.
        amp_enabled: Whether autocast is on.
    Returns:
        float: Mean loss per answer token (nan if the split has no targets).
    """
    model.eval()

    loss_sum, targets = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        num_targets = count_targets(batch["labels"])

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, loss = model(batch["input_ids"], batch["image"], batch["attention_mask"], targets=batch["labels"])
        loss_sum += loss_sum_from_mean(loss, num_targets).item()
        targets += num_targets
        
    model.train()
    return loss_sum / targets if targets else math.nan

@torch.no_grad()
def sample_generations(
    model: VisionLanguageModel, dataset: Dataset, tokenizer: Any, num_samples: int, max_new_tokens: int, device: torch.device
) -> list[list]:
    """
    Generates answers for the first validation samples, to eyeball progress.

    Args:
        model: The (uncompiled) VLM.
        dataset: Validation dataset (VQADataset items).
        tokenizer: Tokenizer to encode prompts and decode answers.
        num_samples: How many samples to generate for.
        max_new_tokens: Tokens per answer.
        device: Device to run on.
    Returns:
        list[list]: Rows [image tensor, question, generated answer, reference answer].
    """
    model.eval()
    rows = []
    for index in range(min(num_samples, len(dataset))):
        item = dataset[index]
        encoded = tokenizer(item["text_data"], return_tensors="pt")
        generated = model.generate(encoded["input_ids"].to(device), item["image"].unsqueeze(0).to(device),
                                   encoded["attention_mask"].to(device), max_new_tokens=max_new_tokens)
        answer = tokenizer.decode(generated[0], skip_special_tokens=True)
        rows.append([item["image"], item["text_data"], answer, item["answer"].replace(tokenizer.eos_token, "")])
    model.train()
    return rows

def build_model(vlm_cfg: VLMConfig, device: torch.device) -> VisionLanguageModel:
    """
    Args:
        vlm_cfg: Model config; vlm_load_backbone_weights selects pretrained backbones or random init.
        device: Device to put the model on.
    Returns:
        VisionLanguageModel: The model on device.
    """
    model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    print(f"TinyForge-Vision initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model.to(device)

def train(
    train_cfg: TrainConfig,
    vlm_cfg: VLMConfig,
    eval_cfg: EvalConfig,
    datasets: tuple[Dataset, Dataset] | None = None,
    mmstar_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """
    Trains the VLM.

    Args:
        train_cfg: Training config.
        vlm_cfg: Model config.
        eval_cfg: MMStar config (test set and batch size).
        datasets: (train, validation) datasets; None = load the Cauldron per train_cfg. Tests pass
            small synthetic datasets here.
        mmstar_loader: MMStar loader; None = build it from eval_cfg (limited to train_cfg.mmstar_limit).
    Returns:
        dict[str, Any]: {"step", "train_loss", "val_loss", "mmstar", "run_dir"} at the end of the run.
    Raises:
        ValueError: If the training split is smaller than one optimizer step.
    """
    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, needs_scaler = detect_precision()
    amp_enabled = amp_dtype != torch.float32
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    train_dataset, val_dataset = datasets if datasets is not None else get_train_val_datasets(train_cfg, vlm_cfg)
    train_loader, val_loader = get_train_val_loaders(train_dataset, val_dataset, train_cfg, vlm_cfg)
    if mmstar_loader is None:
        mmstar_loader = get_mmstar_loader(eval_cfg, vlm_cfg, limit=train_cfg.mmstar_limit)

    steps_per_epoch = len(train_dataset) // train_cfg.batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"{len(train_dataset)} training samples is less than one step of {train_cfg.batch_size}")
    total_steps = steps_per_epoch * train_cfg.epochs
    if train_cfg.max_steps is not None:
        total_steps = min(total_steps, train_cfg.max_steps)

    model = build_model(vlm_cfg, device)
    optimizer = torch.optim.AdamW(build_param_groups(model, train_cfg), betas=train_cfg.betas,
                                  weight_decay=train_cfg.weight_decay)
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    scaler = torch.amp.GradScaler("cuda") if needs_scaler else None

    step, epoch, step_in_epoch, wandb_run_id, best_mmstar = 0, 0, 0, None, 0.0
    if train_cfg.resume_from:
        bookkeeping = load_checkpoint(resolve_checkpoint_source(train_cfg.resume_from), model, optimizer, scaler)
        step, epoch, step_in_epoch = bookkeeping["step"], bookkeeping["epoch"], bookkeeping["step_in_epoch"]
        wandb_run_id, best_mmstar = bookkeeping["wandb_run_id"], bookkeeping["best_mmstar"] or 0.0
        print(f"Resumed from {train_cfg.resume_from} at step {step} (epoch {epoch + 1}, step {step_in_epoch} in epoch)")

    scheduler = build_scheduler("cosine", optimizer, total_steps, warmup_frac=train_cfg.warmup_frac,
                                min_lr_frac=train_cfg.min_lr_frac, last_epoch=step - 1)
    forward_model = torch.compile(model) if train_cfg.compile else model

    run_name = train_cfg.run_name or (Path(train_cfg.resume_from).parent.name if train_cfg.resume_from
                                      else get_run_name(train_cfg, len(train_dataset)))
    run_dir = Path(train_cfg.checkpoint_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if train_cfg.log_wandb:
        wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity, name=run_name, id=wandb_run_id,
                   resume="allow" if wandb_run_id else None, mode=train_cfg.wandb_mode,
                   config={"VLMConfig": asdict(vlm_cfg), "TrainConfig": asdict(train_cfg), "EvalConfig": asdict(eval_cfg)})
        wandb_run_id = wandb.run.id
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    def log(metrics: dict[str, Any]) -> None:
        """
        Logs metrics to W&B (if enabled) at the current step.

        Args:
            metrics: Metric name -> value.
        """
        if train_cfg.log_wandb:
            wandb.log({"step": step, **metrics})

    def run_eval() -> tuple[float, float]:
        """
        Validation loss, MMStar accuracy (test_mmstar, in fp32) and sample
        generations; saves the model if MMStar improved.

        Returns:
            tuple[float, float]: (val loss, MMStar accuracy).
        """
        nonlocal best_mmstar
        start = time.monotonic()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_loss = evaluate_val_loss(forward_model, val_loader, device, amp_dtype, amp_enabled)
        accuracy = test_mmstar(model, tokenizer, mmstar_loader, device)
        rows = sample_generations(model, val_dataset, tokenizer, train_cfg.num_sample_generations,
                                  train_cfg.sample_max_new_tokens, device)
        improved = accuracy > best_mmstar
        if improved:
            best_mmstar = accuracy
            model.save_pretrained(str(run_dir / "best_mmstar"))
        seconds = time.monotonic() - start

        # tqdm.write prints above the progress bar instead of breaking it
        tqdm.write(f"  eval @ step {step}: val loss {val_loss:.4f} | MMStar {accuracy:.4f}"
                   f"{' (best, saved)' if improved else ''} | {seconds:.0f}s")
        for _, question, answer, reference in rows:
            tqdm.write(f"    {question!r} -> {answer!r} (reference {reference!r})")
        if train_cfg.log_wandb:
            table = wandb.Table(columns=["image", "question", "generated", "reference"],
                                data=[[wandb.Image(image), q, a, r] for image, q, a, r in rows])
            log({"val/loss": val_loss, "val/mmstar": accuracy, "val/best_mmstar": best_mmstar,
                 "val/seconds": seconds, "val/samples": table})
        return val_loss, accuracy

    print(f"{run_name}: {len(train_dataset):,} train / {len(val_dataset):,} val samples | {steps_per_epoch} steps/epoch, "
          f"{total_steps} total | batch {train_cfg.batch_size} = {train_cfg.grad_accum_steps} x {train_cfg.micro_batch_size} "
          f"| {device}, {amp_dtype}")

    val_loss, accuracy, train_loss = math.nan, math.nan, math.nan
    if step == 0 and train_cfg.eval_interval > 0:
        val_loss, accuracy = run_eval()

    forward_model.train()
    optimizer.zero_grad(set_to_none=True)
    max_grad_norm = train_cfg.max_grad_norm if train_cfg.max_grad_norm is not None else math.inf
    cached_batches = None
    start_time = time.monotonic()
    progress = tqdm(total=total_steps, initial=step, desc=run_name, unit="step", dynamic_ncols=True)

    while step < total_steps and epoch < train_cfg.epochs:
        train_loader.sampler.set_epoch(epoch, start_index=step_in_epoch * train_cfg.batch_size)
        micro_batches = iter(train_loader)

        while step_in_epoch < steps_per_epoch and step < total_steps:
            step_start = time.monotonic()
            if train_cfg.overfit_batch:
                cached_batches = cached_batches or [next(micro_batches) for _ in range(train_cfg.grad_accum_steps)]
                batches = cached_batches
            else:
                batches = [next(micro_batches) for _ in range(train_cfg.grad_accum_steps)]
            step_targets = max(1, sum(count_targets(batch["labels"]) for batch in batches))

            loss_sum, step_tokens = 0.0, 0
            for batch in batches:
                step_tokens += count_tokens(batch, vlm_cfg)
                batch = to_device(batch, device)
                num_targets = count_targets(batch["labels"])
                if num_targets == 0:
                    continue  # Every sample truncated: nothing to learn from, and the mean loss is nan
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    _, loss = forward_model(batch["input_ids"], batch["image"], batch["attention_mask"],
                                            targets=batch["labels"])
                micro_loss_sum = loss_sum_from_mean(loss, num_targets)
                loss_to_backward = micro_loss_sum / step_targets
                if scaler is not None:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()
                loss_sum += micro_loss_sum.item()

            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            lrs = [group["lr"] for group in optimizer.param_groups]
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            step_in_epoch += 1

            train_loss = loss_sum / step_targets
            step_seconds = time.monotonic() - step_start
            progress.update(1)
            progress.set_postfix(epoch=epoch + 1, loss=f"{train_loss:.4f}", refresh=False)
            log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/grad_norm": float(grad_norm),
                **{f"train/lr_group{i}": lr for i, lr in enumerate(lrs)},
                "train/tokens_per_second": step_tokens / step_seconds,
                "train/samples_per_second": train_cfg.batch_size / step_seconds,
                "train/peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
            })
            if step % train_cfg.log_interval == 0 or step == total_steps:
                elapsed = time.monotonic() - start_time
                tqdm.write(f"epoch {epoch + 1} | step {step:6d}/{total_steps} | loss {train_loss:.4f} | "
                           f"grad norm {float(grad_norm):.2f} | lr {lrs[0]:.2e} | "
                           f"tok/s {step_tokens / step_seconds:,.0f} | elapsed {elapsed / 60:.1f} min")

            if train_cfg.eval_interval > 0 and step % train_cfg.eval_interval == 0 and step < total_steps:
                val_loss, accuracy = run_eval()
            if step % train_cfg.checkpoint_interval == 0 and step < total_steps:
                save_checkpoint(run_dir / "latest.pt", model, optimizer, step, epoch, step_in_epoch,
                                scaler, wandb_run_id, best_mmstar)

        if step_in_epoch >= steps_per_epoch:
            epoch, step_in_epoch = epoch + 1, 0
    progress.close()

    val_loss, accuracy = run_eval()
    save_checkpoint(run_dir / "latest.pt", model, optimizer, step, epoch, step_in_epoch, scaler, wandb_run_id, best_mmstar)
    model.save_pretrained(str(run_dir / "final"))
    print(f"Done: {step} steps in {(time.monotonic() - start_time) / 60:.1f} min | final val loss {val_loss:.4f} | "
          f"MMStar {accuracy:.4f} | best MMStar {best_mmstar:.4f} | saved to {run_dir}")

    if train_cfg.hf_repo_id:
        model.push_to_hub(train_cfg.hf_repo_id, private=True)
    if train_cfg.log_wandb:
        wandb.summary.update({"final/val_loss": val_loss, "final/mmstar": accuracy, "final/best_mmstar": best_mmstar})
        wandb.finish()

    return {"step": step, "train_loss": train_loss, "val_loss": val_loss, "mmstar": accuracy, "run_dir": str(run_dir)}

def apply_overrides(cfg: Any, overrides: list[str]) -> Any:
    """
    Applies "field=value" overrides to a dataclass config. Values are parsed as Python literals
    (ints, floats, True/False, None, tuples), falling back to plain strings.

    Args:
        cfg: Dataclass config instance.
        overrides: Strings like "batch_size=64".
    Returns:
        Any: A new config with the overrides applied.
    Raises:
        ValueError: If a field does not exist on the config.
    """
    names = {f.name for f in dataclasses.fields(cfg)}
    changes = {}
    for override in overrides:
        key, _, raw = override.partition("=")
        if key not in names:
            raise ValueError(f"{type(cfg).__name__} has no field '{key}'")
        try:
            changes[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            changes[key] = raw
    return dataclasses.replace(cfg, **changes)

def main() -> None:
    """
    Command-line entry point: `uv run python -m tfv.train.train --set max_steps=100 --set log_wandb=False`.
    """
    parser = argparse.ArgumentParser(description="Train TinyForge-Vision")
    parser.add_argument("--set", dest="train_overrides", action="append", default=[], metavar="FIELD=VALUE",
                        help="Override a TrainConfig field (repeatable)")
    parser.add_argument("--model-set", dest="model_overrides", action="append", default=[], metavar="FIELD=VALUE",
                        help="Override a VLMConfig field (repeatable)")
    parser.add_argument("--eval-set", dest="eval_overrides", action="append", default=[], metavar="FIELD=VALUE",
                        help="Override an EvalConfig field (repeatable)")
    args = parser.parse_args()

    train_cfg = apply_overrides(TrainConfig(), args.train_overrides)
    vlm_cfg = apply_overrides(VLMConfig(), args.model_overrides)
    eval_cfg = apply_overrides(EvalConfig(), args.eval_overrides)
    print("--- VLM Config ---", vlm_cfg, "--- Train Config ---", train_cfg, "--- Eval Config ---", eval_cfg, sep="\n")
    train(train_cfg, vlm_cfg, eval_cfg)

if __name__ == "__main__":
    main()
