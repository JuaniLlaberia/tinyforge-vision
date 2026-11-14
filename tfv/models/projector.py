import torch
import torch.nn as nn

from tfv.config import VLMConfig

def pixel_shuffle(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    Merges each scale_factor x scale_factor block of neighbouring patches into one token by
    concatenating their features: fewer tokens, wider features.

    The N patches are a flattened sqrt(N) x sqrt(N) grid in row-major order (patch index =
    row * grid_width + col). Each output token must contain 2D spatial neighbours (for factor 2:
    patches (r, c), (r, c+1), (r+1, c), (r+1, c+1)), not consecutive indices in the flattened
    sequence. Output tokens are again in row-major order over the smaller grid.

    For example [B, 196, 768] -> [B, 49, 3072] with factor 2.

    Args:
        x: Patch features, [B, N, D]. N must be a perfect square whose root is divisible by scale_factor.
        scale_factor: Block side length f.
    Returns:
        torch.Tensor: [B, N / f^2, D * f^2].
    """
    batch, num_patches, dim = x.shape
    side = int(num_patches**0.5)

    assert side * side == num_patches, f"pixel_shuffle needs a square patch grid, got N={num_patches}"
    assert side % scale_factor == 0, f"grid side {side} is not divisible by scale_factor {scale_factor}"

    f, out_side = scale_factor, side // scale_factor

    x = x.view(batch, out_side, f, out_side, f, dim)  # [B, row block, row in block, col block, col in block, D]
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, row block, col block, row in block, col in block, D]

    return x.reshape(batch, out_side * out_side, f * f * dim)

class ModalityProjector(nn.Module):
    """
    Maps ViT patch features into the language model's embedding space: pixel shuffle, then a
    linear projection.
    """
    def __init__(self, cfg: VLMConfig) -> None:
        """
        Args:
            cfg: Model config; uses vit_hidden_dim, lm_hidden_dim, mp_pixel_shuffle_factor.
        """
        super().__init__()
        self.scale_factor = cfg.mp_pixel_shuffle_factor
        self.proj = nn.Linear(cfg.vit_hidden_dim * self.scale_factor**2, cfg.lm_hidden_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ViT patch features, [B, N, vit_hidden_dim], e.g. [B, 196, 768].
        Returns:
            torch.Tensor: Image token embeddings, [B, N / f^2, lm_hidden_dim], e.g. [B, 49, 576].
        """
        return self.proj(pixel_shuffle(x, self.scale_factor))