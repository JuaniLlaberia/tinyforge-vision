import torch

from tfv.config import TrainConfig

def build_param_groups(model: torch.nn.Module, cfg: TrainConfig) -> list[dict]:
    """
    Decides what trains and at which learning rate. The training loop passes the result to
    torch.optim.AdamW(groups, betas=cfg.betas, weight_decay=cfg.weight_decay) and schedules every
    group's lr with the same warmup + cosine multiplier.

    Two groups: the projector (model.MP) at cfg.lr_mp, and the vision encoder plus decoder
    together at cfg.lr_backbones. A part frozen by cfg.freeze_vision_encoder or
    cfg.freeze_language_model gets requires_grad=False and is left out of the groups. Every
    trainable parameter is in exactly one group (the decoder's head weight is tied to its token
    embedding, so it is one parameter).

    Args:
        model: The VisionLanguageModel (attributes vision_encoder, MP, decoder).
        cfg: Training config (lr_mp, lr_backbones, freeze_vision_encoder, freeze_language_model).
    Returns:
        list[dict]: AdamW parameter groups, each {"params": [...], "lr": float} (extra keys allowed,
            e.g. a "name" for logging).
    """
    for module, frozen in ((model.vision_encoder, cfg.freeze_vision_encoder), (model.decoder, cfg.freeze_language_model)):
        module.requires_grad_(not frozen)

    groups = [{"name": "projector", "params": list(model.MP.parameters()), "lr": cfg.lr_mp}]

    backbones = [p for module in (model.vision_encoder, model.decoder) for p in module.parameters() if p.requires_grad]
    if backbones:
        groups.append({"name": "backbones", "params": backbones, "lr": cfg.lr_backbones})
        
    return groups
