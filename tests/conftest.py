import copy

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from tfv.config import VLMConfig
from tfv.data.processors import get_tokenizer

# Shared by the slow tests that load the published checkpoint.
CHECKPOINT = "lusxvr/nanoVLM-222M"

def tiny_vlm_config() -> VLMConfig:
    """
    A VLM config small enough for fast CPU tests: 32x32 images, 8x8 patches (16 patches, 4 image
    tokens after a factor-2 pixel shuffle), one ViT block and one LM block. The vocabulary stays
    the real one so real tokenizer ids can be fed in.

    Returns:
        VLMConfig: The tiny config.
    """
    return VLMConfig(
        vit_hidden_dim=32, vit_inter_dim=64, vit_patch_size=8, vit_img_size=32, vit_n_heads=2, vit_n_blocks=1,
        lm_hidden_dim=32, lm_inter_dim=64, lm_n_heads=4, lm_n_kv_heads=2, lm_n_blocks=1,
        mp_pixel_shuffle_factor=2, vlm_load_backbone_weights=False,
    )

@pytest.fixture
def tiny_cfg() -> VLMConfig:
    """
    Returns:
        VLMConfig: A fresh tiny config per test (from_pretrained-style code may mutate configs).
    """
    return copy.deepcopy(tiny_vlm_config())

@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerBase:
    """
    The tokenizer (HuggingFaceTB/cosmo2-tokenizer, pad = EOS). Downloads a few MB once.

    Returns:
        PreTrainedTokenizerBase: The tokenizer.
    """
    return get_tokenizer(VLMConfig().lm_tokenizer)

@pytest.fixture
def vqa_samples(tokenizer: PreTrainedTokenizerBase) -> list[dict]:
    """
    Synthetic VQADataset-style samples covering the collator's cases: short and long answers,
    multi-token questions, a one-word answer, and one sample too long for lm_max_length = 79
    (truncated, so all its labels must be -100).

    Args:
        tokenizer: Provides the EOS token appended to answers.
    Returns:
        list[dict]: Samples {"image": [3, 32, 32], "text_data": str, "answer": str}.
    """
    eos = tokenizer.eos_token
    qa_pairs = [
        ("What color is the car?", "Red."),
        ("How many people are in the picture? Count carefully.", "There are three people standing near the door."),
        ("Is this a cat?", "Yes"),
        ("Describe the chart.", "The chart shows revenue growing every quarter, with the largest jump in Q3."),
        ("Read the text in the image. " * 10, "This sample is far too long to fit in seventy-nine tokens. " * 3),
        ("Which option is correct? A. 1 B. 2 C. 3 D. 4", "C"),
    ]
    generator = torch.Generator().manual_seed(0)
    return [
        {"image": torch.rand(3, 32, 32, generator=generator), "text_data": f"Question: {q} Answer:", "answer": a + eos}
        for q, a in qa_pairs
    ]
