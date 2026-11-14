# Checkpoint helpers adapted from TinyForge (model/model.py, model/helper/).
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    step_in_epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    wandb_run_id: str | None = None,
    best_mmstar: float = 0.0,
) -> None:
    """
    Saves everything needed to resume training. Writes to a temporary file first and renames it,
    so an interrupted save (e.g. a Colab disconnect) never corrupts the previous checkpoint.

    Args:
        path: Output file, e.g. checkpoints/<run>/latest.pt.
        model: The (uncompiled) model.
        optimizer: The optimizer.
        step: Optimizer steps completed in the run.
        epoch: Current epoch (0-based).
        step_in_epoch: Optimizer steps completed in the current epoch.
        scaler: GradScaler for fp16 training, if any.
        wandb_run_id: W&B run id, to resume the same run.
        best_mmstar: Best MMStar accuracy seen so far.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "wandb_run_id": wandb_run_id,
        "best_mmstar": best_mmstar,
    }
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)

def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    """
    Restores model, optimizer and scaler state in place.

    Args:
        path: A file written by save_checkpoint.
        model: The (uncompiled) model to restore into.
        optimizer: The optimizer to restore into (built with the same param groups).
        scaler: GradScaler to restore into, if training in fp16.
    Returns:
        dict: Bookkeeping: "step", "epoch", "step_in_epoch", "wandb_run_id", "best_mmstar".
    Raises:
        KeyError: If the file is missing required keys.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "step", "epoch", "step_in_epoch"):
        if key not in state:
            raise KeyError(f"Checkpoint {path} is missing '{key}'")

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])

    return {key: state.get(key) for key in ("step", "epoch", "step_in_epoch", "wandb_run_id", "best_mmstar")}

def resolve_checkpoint_source(ref: str) -> Path:
    """
    Local path if it exists on disk, else '<owner>/<repo>/<path/in/repo>' on the HF Hub: the repo id
    is the first two '/'-separated segments, the rest is the file path inside the repo.

    Args:
        ref: Local path or Hub reference.
    Returns:
        Path: Local path to the file (downloaded if needed).
    """
    local_path = Path(ref)
    if local_path.exists():
        return local_path

    parts = ref.split("/")
    repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))

def detect_precision() -> tuple[torch.dtype, bool]:
    """
    Detects the training dtype for the current GPU.

    Returns:
        tuple[torch.dtype, bool]: (dtype, needs_grad_scaler).
            - No CUDA (CPU dev/smoke tests): (float32, False).
            - CUDA with bf16 support (A100, L4, H100): (bfloat16, False).
            - CUDA otherwise (e.g. T4): (float16, True).
    """
    if not torch.cuda.is_available():
        return torch.float32, False

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False

    return torch.float16, True
