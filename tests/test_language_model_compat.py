import torch

from tfv.config import VLMConfig
from tfv.models.language_model import LanguageModelGroupedQueryAttention, RotaryEmbedding

def test_cpu_attention_branch_matches_v01() -> None:
    """
    The [tinyforge-vision] SDPA branch (merged causal + padding mask, used on every device) gives
    the same outputs on real tokens as the original manual-attention path, has no NaNs even when a
    row is mostly padding, and equals the original causal SDPA call when there is no padding.
    """
    torch.manual_seed(0)
    cfg = VLMConfig()
    attention = LanguageModelGroupedQueryAttention(cfg).eval()
    B, T = 3, 12
    x = torch.randn(B, T, cfg.lm_hidden_dim)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, :4] = 0
    mask[2, :11] = 0  # Only the last token is real
    cos, sin = RotaryEmbedding(cfg)(torch.arange(T).unsqueeze(0).expand(B, -1))

    with torch.no_grad():
        compat = attention(x, cos, sin, mask)
        no_padding = attention(x[:1], cos[:1], sin[:1], None)  # Original is_causal=True call
        attention.sdpa = False
        manual = attention(x, cos, sin, mask)  # Original manual path (zeroes padded positions)

    real = mask.bool()
    assert not torch.isnan(compat).any()
    assert torch.allclose(compat[real], manual[real], atol=1e-5)
    assert torch.allclose(compat[0], no_padding[0], atol=1e-5)
