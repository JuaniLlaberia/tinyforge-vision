# Reference for tests only. Borrowed from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082,
# file models/modality_projector.py. Changes: added docstrings and type hints. No logic changes.
# Modality Projection from Vision to Language
import torch
import torch.nn as nn

from tfv.config import VLMConfig

class ModalityProjector(nn.Module):
    """
    v0.1 projector: pixel shuffle, then Linear(vit_hidden_dim * f^2 -> lm_hidden_dim, bias=False).
    """
    def __init__(self, cfg: VLMConfig) -> None:
        """
        Args:
            cfg: Model config; uses vit_hidden_dim, lm_hidden_dim, mp_pixel_shuffle_factor.
        """
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes the projection with N(0, 0.02).

        Args:
            module: Module visited by self.apply.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D].
        Returns:
            torch.Tensor: [B, N / f^2, D * f^2].
        """
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0 # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, vit_hidden_dim].
        Returns:
            torch.Tensor: [B, N / f^2, lm_hidden_dim].
        """
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x
