import pytest
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from tests.reference.collators import VQACollator as RefVQACollator
from tfv.config import VLMConfig
from tfv.data.collators import VQACollator
from tfv.data.datasets import VQADataset
from tfv.data.processors import get_image_processor

MAX_LENGTH = VLMConfig().lm_max_length  # 79

def assert_same_batch(batch: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    """
    Checks every tensor of our batch equals the reference batch exactly.

    Args:
        batch: Our collator's output.
        expected: The reference collator's output on the same samples.
    """
    assert set(batch) == set(expected)
    for key in expected:
        assert torch.equal(batch[key], expected[key]), f"'{key}' differs from the reference"

def test_collator_matches_v01(tokenizer: PreTrainedTokenizerBase, vqa_samples: list[dict]) -> None:
    """
    Same images, input_ids, attention_mask and labels as the reference VQACollator on synthetic samples,
    including one truncated sample.

    Args:
        tokenizer: The tokenizer.
        vqa_samples: Synthetic samples (see conftest).
    """
    expected = RefVQACollator(tokenizer, MAX_LENGTH)(vqa_samples)
    assert torch.all(expected["labels"][4] == -100), "fixture should contain a truncated sample"

    batch = VQACollator(tokenizer, MAX_LENGTH)(vqa_samples)

    assert batch["input_ids"].shape == (len(vqa_samples), MAX_LENGTH)
    assert_same_batch(batch, expected)

def test_labels_are_answer_tokens_only(tokenizer: PreTrainedTokenizerBase, vqa_samples: list[dict]) -> None:
    """
    For a sample that fits, the non-ignored labels decode back to exactly its answer (with EOS).

    Args:
        tokenizer: The tokenizer.
        vqa_samples: Synthetic samples (see conftest).
    """
    batch = VQACollator(tokenizer, MAX_LENGTH)([vqa_samples[1]])
    labels = batch["labels"][0]

    targets = labels[labels != -100]

    assert tokenizer.decode(targets) == vqa_samples[1]["answer"]

@pytest.mark.slow
def test_collator_matches_v01_on_cauldron(tokenizer: PreTrainedTokenizerBase) -> None:
    """
    Same as the reference on 20 real Cauldron samples via VQADataset (ai2d, datikz with long answers that
    get truncated, vqav2), streamed so nothing large is downloaded.

    Args:
        tokenizer: The tokenizer.
    """
    rows = []
    for subset, n in [("ai2d", 7), ("datikz", 7), ("vqav2", 6)]:
        rows += list(load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True).take(n))
    dataset = VQADataset(rows, tokenizer, get_image_processor(VLMConfig().vit_img_size))
    samples = [dataset[i] for i in range(len(dataset))]

    expected = RefVQACollator(tokenizer, MAX_LENGTH)(samples)
    assert any(torch.all(row == -100) for row in expected["labels"]), "expected at least one truncated sample"

    assert_same_batch(VQACollator(tokenizer, MAX_LENGTH)(samples), expected)
