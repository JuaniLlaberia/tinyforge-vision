import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from tests.conftest import CHECKPOINT
from tests.reference.modality_projector import ModalityProjector as RefModalityProjector
from tfv.config import VLMConfig
from tfv.models.projector import ModalityProjector, pixel_shuffle

@pytest.mark.parametrize("n_patches,dim,factor", [(196, 768, 2), (64, 16, 2), (64, 16, 4), (16, 8, 1)])
def test_pixel_shuffle_matches_v01(n_patches: int, dim: int, factor: int) -> None:
    """
    Same output as the reference ModalityProjector.pixel_shuffle on random input, for the model's shape and a
    few others (a pure rearrangement, so the match must be exact).

    Args:
        n_patches: Number of input patches N.
        dim: Feature size D.
        factor: Pixel shuffle factor f.
    """
    x = torch.randn(2, n_patches, dim)
    reference = RefModalityProjector(VLMConfig(vit_hidden_dim=dim, mp_pixel_shuffle_factor=factor))
    expected = reference.pixel_shuffle(x)

    out = pixel_shuffle(x, factor)

    assert out.shape == (2, n_patches // factor**2, dim * factor**2)
    assert torch.equal(out, expected)

def test_pixel_shuffle_merges_spatial_neighbours() -> None:
    """
    Each output token must hold a 2x2 block of spatially neighbouring patches of the 14x14 grid,
    not 4 consecutive patches of the flattened sequence (the classic bug).
    """
    grid = 14
    # Feature of every patch = its own index, so we can read back which patches were merged.
    x = torch.arange(grid * grid, dtype=torch.float32).view(1, grid * grid, 1)

    out = pixel_shuffle(x, 2)  # [1, 49, 4]

    assert set(out[0, 0].tolist()) == {0, 1, 14, 15}  # top-left block
    assert set(out[0, 1].tolist()) == {2, 3, 16, 17}  # its right neighbour
    assert set(out[0, 7].tolist()) == {28, 29, 42, 43}  # first block of the second block-row
    assert set(out[0, 48].tolist()) == {180, 181, 194, 195}  # bottom-right block

def test_projector_layout_matches_checkpoint() -> None:
    """
    The 222M checkpoint stores the projector as a single bias-free "proj.weight" of shape
    [576, 3072]; our module must have exactly that state_dict.
    """
    projector = ModalityProjector(VLMConfig())
    state = projector.state_dict()

    assert list(state) == ["proj.weight"]
    assert state["proj.weight"].shape == (576, 3072)

def test_projector_matches_v01_same_weights() -> None:
    """
    With the reference's (random) weights loaded into ours, outputs match on random ViT features.
    """
    cfg = VLMConfig()
    reference = RefModalityProjector(cfg)
    projector = ModalityProjector(cfg)
    projector.load_state_dict(reference.state_dict())
    x = torch.randn(2, 196, cfg.vit_hidden_dim)

    with torch.no_grad():
        out, expected = projector(x), reference(x)

    assert out.shape == (2, 49, cfg.lm_hidden_dim)
    assert torch.allclose(out, expected, atol=1e-5)

@pytest.mark.slow
def test_projector_matches_v01_checkpoint_weights() -> None:
    """
    With the published projector weights (MP.proj.weight from lusxvr/nanoVLM-222M), outputs match the reference.
    """
    weights_path = hf_hub_download(CHECKPOINT, "model.safetensors")
    with safe_open(weights_path, framework="pt") as f:
        state = {"proj.weight": f.get_tensor("MP.proj.weight")}
    cfg = VLMConfig()
    reference, projector = RefModalityProjector(cfg), ModalityProjector(cfg)
    reference.load_state_dict(state)
    projector.load_state_dict(state)
    x = torch.randn(2, 196, cfg.vit_hidden_dim)

    with torch.no_grad():
        assert torch.allclose(projector(x), reference(x), atol=1e-5)
