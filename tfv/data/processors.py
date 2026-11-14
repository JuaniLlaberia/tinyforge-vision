# Borrowed from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082, file data/processors.py.
# Changes: added docstrings and type hints. No logic changes.
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import torchvision.transforms as transforms

TOKENIZERS_CACHE = {}

def get_tokenizer(name: str) -> PreTrainedTokenizerBase:
    """
    Loads (once, then caches) a fast tokenizer and sets its padding token to EOS.

    Args:
        name: HF tokenizer repo id, e.g. "HuggingFaceTB/cosmo2-tokenizer".
    Returns:
        PreTrainedTokenizerBase: The tokenizer, with pad_token = eos_token.
    """
    if name not in TOKENIZERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
        
    return TOKENIZERS_CACHE[name]

def get_image_processor(img_size: int) -> transforms.Compose:
    """
    Builds the image preprocessing: resize to img_size x img_size, then ToTensor (values in
    [0, 1], no normalization).

    Args:
        img_size: Output height and width in pixels.
    Returns:
        transforms.Compose: Maps a PIL image to a float tensor [3, img_size, img_size].
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
