import torch

from tests.reference.vision_language_model import combine_as_in_forward
from tfv.models.vlm import combine_embeddings

def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random image tokens, text embeddings and a left-padded mask with the model's sizes.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: image [2, 49, 576], text [2, 79, 576],
            mask [2, 79] (int64, second row has 30 padding positions).
    """
    image_embd = torch.randn(2, 49, 576)
    token_embd = torch.randn(2, 79, 576)
    attention_mask = torch.ones(2, 79, dtype=torch.long)
    attention_mask[1, :30] = 0
    return image_embd, token_embd, attention_mask

def test_combine_matches_v01() -> None:
    """
    Same combined embeddings and extended mask as the concatenation inside the reference forward.
    """
    image_embd, token_embd, attention_mask = make_inputs()
    expected_embd, expected_mask = combine_as_in_forward(image_embd, token_embd, attention_mask)

    combined, mask = combine_embeddings(image_embd, token_embd, attention_mask)

    assert combined.shape == (2, 49 + 79, 576)
    assert torch.equal(combined, expected_embd)
    assert torch.equal(mask, expected_mask)

def test_combine_puts_image_first_and_keeps_mask_dtype() -> None:
    """
    Image tokens come first, every image position is attended to, and the mask keeps its dtype.
    """
    image_embd, token_embd, attention_mask = make_inputs()

    combined, mask = combine_embeddings(image_embd, token_embd, attention_mask)

    assert torch.equal(combined[:, :49], image_embd)
    assert torch.equal(combined[:, 49:], token_embd)
    assert mask.dtype == attention_mask.dtype
    assert torch.all(mask[:, :49] == 1)
    assert torch.equal(mask[:, 49:], attention_mask)

def test_combine_without_mask() -> None:
    """
    With no mask, embeddings are still combined and the mask stays None.
    """
    image_embd, token_embd, _ = make_inputs()

    combined, mask = combine_embeddings(image_embd, token_embd, None)

    assert combined.shape == (2, 128, 576)
    assert mask is None
