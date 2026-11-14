from dataclasses import dataclass

CAULDRON_SUBSETS: tuple[str, ...] = (
    "ai2d", "aokvqa", "chart2text", "chartqa", "clevr", "cocoqa", "datikz", "diagram_image_to_text",
    "docvqa", "dvqa", "figureqa", "finqa", "geomverse", "hateful_memes", "hitab", "iam", "iconqa",
    "infographic_vqa", "intergps", "localized_narratives", "mapqa", "multihiertt", "ocrvqa", "plotqa",
    "raven", "rendered_text", "robut_sqa", "robut_wikisql", "robut_wtq", "scienceqa", "screen2words",
    "st_vqa", "tabmwp", "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa", "vistext", "visual7w",
    "visualmrc", "vqarad", "vqav2", "vsr", "websight",
)

PREPARED_CAULDRON_REPO: str = "Juanillaberia/tinyforge-vision-cauldron-224"

@dataclass
class VLMConfig:
    """
    Model architecture: SigLIP-B/16-224 + SmolLM2-135M, 222M parameters.

    Do not rename or remove fields: VisionLanguageModel.from_pretrained builds this class with
    VLMConfig(**config.json), and the lusxvr/nanoVLM-222M config.json uses exactly these names.
    """
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 224
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = "google/siglip-base-patch16-224"

    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_vocab_size: int = 49152
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 128 - 49  # Text tokens; plus 49 image tokens gives a sequence of 128
    lm_use_tokens: bool = False  # False: the LM takes embeddings (needed to prepend image tokens)
    lm_tie_weights: bool = True  # Tie the LM head to the token embedding matrix
    lm_model_type: str = "HuggingFaceTB/SmolLM2-135M"
    lm_tokenizer: str = "HuggingFaceTB/cosmo2-tokenizer"
    lm_eos_token_id: int = 0

    mp_pixel_shuffle_factor: int = 2  # 196 patches -> 49 image tokens

    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = "checkpoints/nanoVLM-222M"

@dataclass
class TrainConfig:
    """
    Training settings for tfv.train.train.
    """
    lr_mp: float = 2e-3
    lr_backbones: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float | None = None  # None = no clipping
    warmup_frac: float = 0.03
    min_lr_frac: float = 0.1
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False

    # Batch. batch_size is the effective batch per optimizer step; each step runs
    # batch_size // micro_batch_size forward/backward passes (gradient accumulation).
    batch_size: int = 256
    micro_batch_size: int = 256
    epochs: int = 5
    max_steps: int | None = None  # Stop early (smoke runs); the LR schedule is compressed to fit
    compile: bool = True

    # Data. The concatenated subsets are shuffled with seed 0 and the last val_ratio is held out.
    train_dataset_path: str = PREPARED_CAULDRON_REPO  # "HuggingFaceM4/the_cauldron" also works
    train_dataset_name: tuple[str, ...] = CAULDRON_SUBSETS
    data_cutoff_idx: int | None = None  # Use only the first N shuffled samples (train + val)
    val_ratio: float = 0.01
    num_workers: int = 8
    seed: int = 0

    # Evaluation during training: val loss + full MMStar every eval_interval steps (including step
    # 0). Each eval costs roughly a minute on an H100: raise the interval or set mmstar_limit to
    # save budget. Evaluation does not change training.
    eval_interval: int = 250
    mmstar_limit: int | None = None  # Score only the first N MMStar samples during training
    num_sample_generations: int = 3  # Validation images whose generated answers get logged
    sample_max_new_tokens: int = 20
    overfit_batch: bool = False  # Repeat the first step's batch every step (sanity check)

    # Checkpoints and logging. checkpoint_dir should be persistent storage on Colab (Google Drive):
    # latest.pt (model + optimizer, for resuming) is overwritten every checkpoint_interval steps;
    # best_mmstar/ and final/ are save_pretrained model directories.
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    resume_from: str | None = None  # Path to a latest.pt to resume from
    hf_repo_id: str | None = None  # Push checkpoints here (created private if missing)
    run_name: str | None = None  # None = generated from the config
    log_interval: int = 100  # Print a log line every N steps (the progress bar and W&B update every step)
    log_wandb: bool = True
    wandb_project: str = "tinyforge-vision"
    wandb_entity: str | None = None
    wandb_mode: str = "online"  # "online", "offline" or "disabled"

    def __post_init__(self) -> None:
        """
        Validates that the effective batch splits evenly into micro-batches.

        Raises:
            ValueError: If batch_size is not a positive multiple of micro_batch_size.
        """
        if self.micro_batch_size <= 0 or self.batch_size % self.micro_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be a multiple of micro_batch_size ({self.micro_batch_size})"
            )

    @property
    def grad_accum_steps(self) -> int:
        """
        Number of micro-batches accumulated per optimizer step.

        Returns:
            int: batch_size // micro_batch_size.
        """
        return self.batch_size // self.micro_batch_size

@dataclass
class EvalConfig:
    """
    MMStar evaluation settings (Lin-Chen/MMStar "val" split).

    The evaluation samples answers instead of decoding greedily, so scores move between runs:
    report the seeds used.
    """
    model_path: str = "lusxvr/nanoVLM-222M"  # HF repo id or local directory
    test_dataset_path: str = "Lin-Chen/MMStar"
    mmstar_batch_size: int = 32
    seeds: tuple[int, ...] = (0, 1, 2)
    limit: int | None = None  # Score only the first N samples (quick checks)

@dataclass
class PrepareConfig:
    """
    Settings for the one-time Cauldron preprocessing (scripts/prepare_cauldron.py).

    Each subset keeps the Cauldron's schema (images: list of images, texts: list of
    user/assistant/source dicts) but only the first image and first QA pair (the only ones
    VQADataset uses), with the image resized to image_size x image_size (the same resize as the
    image processor) and stored as JPEG.
    """
    source_path: str = "HuggingFaceM4/the_cauldron"
    target_repo_id: str = PREPARED_CAULDRON_REPO
    subsets: tuple[str, ...] = CAULDRON_SUBSETS
    image_size: int = 224
    jpeg_quality: int = 95
    num_proc: int | None = None  # None = os.cpu_count()
    work_dir: str = "prepare_cache"  # Per-subset download cache, deleted after each push
    private: bool = True
    max_shard_size: str = "500MB"

    # Timing run (scripts/benchmark_prepare.py): a mix of light and heavy subsets.
    benchmark_subsets: tuple[str, ...] = ("ai2d", "docvqa", "rendered_text", "vqav2", "localized_narratives")
    benchmark_rows: int = 200
