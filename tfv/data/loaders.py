# Adapted from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082, file train.py (get_dataloaders).
# Changes: double quotes; split into functions; batches are micro-batches (gradient accumulation); the training
# order comes from ResumableSampler (seeded per epoch, so a run can resume mid-epoch) instead of
# shuffle=True; seed_worker is module-level (picklable for spawn-based workers); num_workers and
# an MMStar sample limit come from the config.
import random

import numpy
import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset, Sampler

from tfv.config import EvalConfig, TrainConfig, VLMConfig
from tfv.data.collators import MMStarCollator, VQACollator
from tfv.data.datasets import MMStarDataset, VQADataset
from tfv.data.processors import get_image_processor, get_tokenizer

class ResumableSampler(Sampler[int]):
    """
    Random order that depends only on (seed, epoch), with an optional starting offset: resuming at
    step k of an epoch replays the same permutation and skips the samples already trained on.
    """
    def __init__(self, num_samples: int, seed: int) -> None:
        """
        Args:
            num_samples: Dataset length.
            seed: Base seed; epoch e uses seed + e.
        """
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        """
        Selects the permutation for an epoch and where in it to start.

        Args:
            epoch: Epoch number (0-based).
            start_index: Number of samples of this epoch to skip (already trained on).
        """
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self):
        """
        Returns:
            Iterator[int]: Sample indices for the current epoch from start_index on.
        """
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(self.num_samples, generator=generator)
        return iter(permutation[self.start_index:].tolist())

    def __len__(self) -> int:
        """
        Returns:
            int: Number of indices the current epoch will yield.
        """
        return self.num_samples - self.start_index

def seed_worker(worker_id: int) -> None:
    """
    Seeds numpy and random in each DataLoader worker from torch's per-worker seed.

    Args:
        worker_id: Worker index (unused; torch.initial_seed() already differs per worker).
    """
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def get_train_val_datasets(train_cfg: TrainConfig, vlm_cfg: VLMConfig) -> tuple[VQADataset, VQADataset]:
    """
    Loads and concatenates the training subsets, shuffles them with seed 0, applies the optional
    cutoff, and holds out the last val_ratio as validation.

    Args:
        train_cfg: Uses train_dataset_path, train_dataset_name, data_cutoff_idx, val_ratio.
        vlm_cfg: Uses vit_img_size, lm_tokenizer.
    Returns:
        tuple[VQADataset, VQADataset]: (train, validation).
    """
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    # Load and combine all training datasets
    combined_train_data = []
    for dataset_name in train_cfg.train_dataset_name:
        train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)
        combined_train_data.append(train_ds["train"])

    train_ds = concatenate_datasets(combined_train_data)
    train_ds = train_ds.shuffle(seed=0) 

    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    train_dataset = VQADataset(train_ds.select(range(train_size)), tokenizer, image_processor)
    val_dataset = VQADataset(train_ds.select(range(train_size, total_samples)), tokenizer, image_processor)
    return train_dataset, val_dataset

def get_train_val_loaders(
    train_dataset: Dataset, val_dataset: Dataset, train_cfg: TrainConfig, vlm_cfg: VLMConfig
) -> tuple[DataLoader, DataLoader]:
    """
    Builds the micro-batch loaders. Call train_loader.sampler.set_epoch(...) before each epoch.

    Args:
        train_dataset: Training samples (VQADataset or any dataset of the same items).
        val_dataset: Validation samples.
        train_cfg: Uses micro_batch_size, num_workers, seed.
        vlm_cfg: Uses lm_tokenizer, lm_max_length.
    Returns:
        tuple[DataLoader, DataLoader]: (train, validation); both drop the last incomplete batch.
    """
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)

    g = torch.Generator()
    g.manual_seed(train_cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.micro_batch_size,
        sampler=ResumableSampler(len(train_dataset), train_cfg.seed),
        collate_fn=vqa_collator,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=False,
        collate_fn=vqa_collator,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, val_loader

def get_mmstar_loader(eval_cfg: EvalConfig, vlm_cfg: VLMConfig, limit: int | None = None) -> DataLoader:
    """
    Builds the MMStar test loader (Lin-Chen/MMStar "val" split, in order, no shuffling).

    Args:
        eval_cfg: Uses test_dataset_path, mmstar_batch_size.
        vlm_cfg: Uses vit_img_size, lm_tokenizer.
        limit: Keep only the first N questions (None = all 1,500).
    Returns:
        DataLoader: Batches of MMStarCollator output.
    """
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    test_ds = load_dataset(eval_cfg.test_dataset_path)["val"]
    if limit is not None:
        test_ds = test_ds.select(range(min(limit, len(test_ds))))

    g = torch.Generator()
    g.manual_seed(0)

    return DataLoader(
        MMStarDataset(test_ds, tokenizer, image_processor),
        batch_size=eval_cfg.mmstar_batch_size,
        shuffle=False,
        collate_fn=MMStarCollator(tokenizer),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
