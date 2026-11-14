# Reference for tests only. Borrowed from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082,
# file models/vision_language_model.py: __init__, forward, generate and from_pretrained (save/push
# omitted). Changes: blank lines (2 -> 1), added docstrings and type hints, import paths (the
# borrowed backbones from tfv.models, the reference projector from tests.reference). No logic changes.
# combine_as_in_forward is not a v0.1 function: it is the combine step copied verbatim out of
# v0.1's forward (lines 38-48), so the combine step can be tested on its own.
import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model

from tests.reference.modality_projector import ModalityProjector
from tfv.config import VLMConfig
from tfv.models.language_model import LanguageModel
from tfv.models.vision_transformer import ViT

def combine_as_in_forward(
    image_embd: torch.Tensor, token_embd: torch.Tensor, attention_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    v0.1's combine step, extracted from VisionLanguageModel.forward without changes.

    Args:
        image_embd: [B, N_img, D].
        token_embd: [B, T, D].
        attention_mask: [B, T] or None.
    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: ([B, N_img + T, D], [B, N_img + T] or None).
    """
    combined_embd = torch.cat((image_embd, token_embd), dim=1) # Concatenate image embeddings to token embeddings

    # Adjust attention mask to account for image tokens
    if attention_mask is not None:
        # Create mask of 1s for image tokens (all image tokens should be attended to)
        batch_size = image_embd.size(0)
        img_seq_len = image_embd.size(1)
        image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)

        # Combine image and token attention masks
        attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

    return combined_embd, attention_mask

class VisionLanguageModel(nn.Module):
    """
    v0.1's VisionLanguageModel: ViT -> ModalityProjector -> LanguageModel.
    """
    def __init__(self, cfg: VLMConfig, load_backbone: bool = True) -> None:
        """
        Args:
            cfg: Model config.
            load_backbone: Load pretrained SigLIP and SmolLM2 weights, or random init.
        """
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone

    def forward(
        self,
        input_ids: torch.Tensor,
        image: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: [B, T].
            image: [B, 3, H, W].
            attention_mask: [B, T] or None.
            targets: Shifted labels [B, T] or None.
        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: (text logits [B, T, vocab] if targets else
                hidden states [B, N_img + T, D], loss or None).
        """
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)

        token_embd = self.decoder.token_embedding(input_ids)

        combined_embd = torch.cat((image_embd, token_embd), dim=1) # Concatenate image embeddings to token embeddings

        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            batch_size = image_embd.size(0)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)

            # Combine image and token attention masks
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        logits = self.decoder(combined_embd, attention_mask) # Not logits yet, but easier to return like this

        loss = None
        if targets is not None:
            # Only use the token part of the logits for loss computation
            logits = self.decoder.head(logits)
            logits = logits[:, image_embd.size(1):, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        image: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 5,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T].
            image: [B, 3, H, W].
            attention_mask: [B, T] or None.
            max_new_tokens: Tokens to sample.
        Returns:
            torch.Tensor: Sampled token ids, [B, max_new_tokens].
        """
        # Process image through vision encoder and projection
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)

        # Embed initial tokens
        token_embd = self.decoder.token_embedding(input_ids)

        # Concatenate image embeddings with token embeddings
        combined_embd = torch.cat((image_embd, token_embd), dim=1)

        batch_size = image_embd.size(0)
        img_seq_len = image_embd.size(1)
        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        # Generate from combined embeddings using the decoder
        # We need to use the decoder's forward function and not its generate method
        # because we want to keep track of the image prefix
        outputs = combined_embd
        generated_tokens = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)

        #Note: Here you could implement improvements like e.g. KV caching
        for i in range(max_new_tokens):
            model_out = self.decoder(outputs, attention_mask)

            # Get predictions for the last token only (normally this is the embedding, not the logits)
            last_token_logits = model_out[:, -1, :]

            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                last_token_logits = self.decoder.head(last_token_logits)

            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens[:, i] = next_token.squeeze(-1)

            # Convert to embedding and append
            next_embd = self.decoder.token_embedding(next_token)
            outputs = torch.cat((outputs, next_embd), dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)), dim=1)

        return generated_tokens

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.
            revision: Hub revision to load; None = default branch.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model
