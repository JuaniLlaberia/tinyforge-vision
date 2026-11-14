import pytest
import torch
from transformers import PreTrainedTokenizerBase

from tests.conftest import CHECKPOINT
from tests.reference.collators import VQACollator as RefVQACollator
from tests.reference.vision_language_model import VisionLanguageModel as RefVisionLanguageModel
from tfv.config import VLMConfig
from tfv.models.vlm import VisionLanguageModel

def make_batch(tokenizer: PreTrainedTokenizerBase, samples: list[dict], cfg: VLMConfig) -> dict[str, torch.Tensor]:
    """
    Collates samples with the reference collator (so these tests do not depend on our collator) and
    resizes the images to cfg.vit_img_size.

    Args:
        tokenizer: The tokenizer.
        samples: VQADataset-style samples.
        cfg: Config whose image size the images are resized to.
    Returns:
        dict[str, torch.Tensor]: image, input_ids, attention_mask, labels.
    """
    batch = RefVQACollator(tokenizer, cfg.lm_max_length)(samples)
    size = cfg.vit_img_size
    batch["image"] = torch.nn.functional.interpolate(batch["image"], size=(size, size), mode="bilinear")
    return batch

def assert_same_outputs(model: torch.nn.Module, reference: torch.nn.Module, batch: dict[str, torch.Tensor]) -> None:
    """
    Compares forward with targets (logits + loss), forward without targets, and generate under
    the same seed.

    Args:
        model: Our VisionLanguageModel.
        reference: The reference VisionLanguageModel with the same weights.
        batch: Input batch.
    """
    model.eval()
    reference.eval()
    args = (batch["input_ids"], batch["image"], batch["attention_mask"])
    with torch.no_grad():
        logits, loss = model(*args, targets=batch["labels"])
        expected_logits, expected_loss = reference(*args, targets=batch["labels"])
        assert logits.shape == expected_logits.shape
        assert torch.allclose(logits, expected_logits, atol=1e-5)
        assert torch.allclose(loss, expected_loss, atol=1e-5)

        hidden, no_loss = model(*args)
        expected_hidden, _ = reference(*args)
        assert no_loss is None
        assert torch.allclose(hidden, expected_hidden, atol=1e-5)

    torch.manual_seed(0)
    generated = model.generate(*args, max_new_tokens=5)
    torch.manual_seed(0)
    expected_generated = reference.generate(*args, max_new_tokens=5)
    assert generated.shape == (batch["input_ids"].size(0), 5)
    assert torch.equal(generated, expected_generated), "generate must sample exactly like the reference"

def test_vlm_matches_v01_tiny(tiny_cfg: VLMConfig, tokenizer: PreTrainedTokenizerBase, vqa_samples: list[dict]) -> None:
    """
    Fast check on a tiny random model: our VLM with the reference's weights gives the reference's outputs.

    Args:
        tiny_cfg: Tiny config (see conftest).
        tokenizer: The tokenizer.
        vqa_samples: Synthetic samples (see conftest).
    """
    torch.manual_seed(0)
    reference = RefVisionLanguageModel(tiny_cfg, load_backbone=False)
    model = VisionLanguageModel(tiny_cfg, load_backbone=False)
    model.load_state_dict(reference.state_dict())

    assert_same_outputs(model, reference, make_batch(tokenizer, vqa_samples, tiny_cfg))

@pytest.mark.slow
def test_vlm_matches_v01_checkpoint(tokenizer: PreTrainedTokenizerBase, vqa_samples: list[dict]) -> None:
    """
    Part 2's whole-model check: both load lusxvr/nanoVLM-222M (fp32, CPU) and give the same logits,
    loss and sampled tokens on the same batch.

    Args:
        tokenizer: The tokenizer.
        vqa_samples: Synthetic samples (see conftest).
    """
    reference = RefVisionLanguageModel.from_pretrained(CHECKPOINT)
    model = VisionLanguageModel.from_pretrained(CHECKPOINT)

    assert_same_outputs(model, reference, make_batch(tokenizer, vqa_samples, model.cfg))
