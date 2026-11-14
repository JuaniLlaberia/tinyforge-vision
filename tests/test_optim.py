import pytest
import torch

from tests.reference.vision_language_model import VisionLanguageModel as RefVisionLanguageModel
from tfv.config import TrainConfig, VLMConfig
from tfv.train.optim import build_param_groups

def lr_by_param(groups: list[dict]) -> dict[int, float]:
    """
    Maps each parameter (by id) to its group's lr, failing if a parameter appears twice.

    Args:
        groups: Parameter groups.
    Returns:
        dict[int, float]: id(param) -> lr.
    """
    lrs = {}
    for group in groups:
        for param in group["params"]:
            assert id(param) not in lrs, "a parameter appears in more than one group"
            lrs[id(param)] = group["lr"]
    return lrs

@pytest.mark.parametrize("freeze_vision,freeze_language", [(False, False), (True, False), (False, True), (True, True)])
def test_param_groups(tiny_cfg: VLMConfig, freeze_vision: bool, freeze_language: bool) -> None:
    """
    Projector at lr_mp, backbones at lr_backbones, frozen parts excluded and not requiring grad,
    every trainable parameter in exactly one group, and AdamW accepts the groups.

    Args:
        tiny_cfg: Tiny config (see conftest).
        freeze_vision: TrainConfig.freeze_vision_encoder.
        freeze_language: TrainConfig.freeze_language_model.
    """
    model = RefVisionLanguageModel(tiny_cfg, load_backbone=False)
    cfg = TrainConfig(freeze_vision_encoder=freeze_vision, freeze_language_model=freeze_language)

    groups = build_param_groups(model, cfg)
    lrs = lr_by_param(groups)

    parts = [(model.MP, False, cfg.lr_mp), (model.vision_encoder, freeze_vision, cfg.lr_backbones),
             (model.decoder, freeze_language, cfg.lr_backbones)]
    for module, frozen, lr in parts:
        for param in module.parameters():
            if frozen:
                assert not param.requires_grad and id(param) not in lrs
            else:
                assert param.requires_grad and lrs[id(param)] == lr

    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(lrs) == trainable
    torch.optim.AdamW(groups, betas=cfg.betas, weight_decay=cfg.weight_decay)
