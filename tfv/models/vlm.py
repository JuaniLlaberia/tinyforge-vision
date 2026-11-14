# Mixed file. Written for TinyForge-Vision (not borrowed): combine_embeddings, VisionLanguageModel.forward and .generate.
# Borrowed from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082, file
# models/vision_language_model.py: VisionLanguageModel.__init__, from_pretrained, save_pretrained,
# push_to_hub (changes: added docstrings and type hints, import paths, ModalityProjector from
# tfv.models.projector). MODEL_CARD_TEMPLATE and push_to_hub's commit message are rewritten for
# TinyForge-Vision.
import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from tfv.config import VLMConfig
from tfv.models.language_model import LanguageModel
from tfv.models.projector import ModalityProjector
from tfv.models.vision_transformer import ViT

def combine_embeddings(
    image_embd: torch.Tensor, token_embd: torch.Tensor, attention_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Builds the language model's input: image tokens first, then text tokens, and extends the
    attention mask so every image token is attended to.

    The text is left-padded, so the combined sequence looks like [image tokens][padding][text].

    Args:
        image_embd: Projected image tokens, [B, N_img, D], e.g. [B, 49, 576].
        token_embd: Text token embeddings, [B, T, D].
        attention_mask: Text padding mask, [B, T], 1 = real token, 0 = padding; or None.
    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: (combined embeddings [B, N_img + T, D],
            mask [B, N_img + T] with the same dtype and device as attention_mask, or None when
            attention_mask is None).
    """
    combined = torch.cat((image_embd, token_embd), dim=1)

    if attention_mask is None:
        return combined, None
    
    image_mask = attention_mask.new_ones(image_embd.shape[:2])  # Every image token is attended to
    return combined, torch.cat((image_mask, attention_mask), dim=1)

class VisionLanguageModel(nn.Module):
    """
    ViT -> ModalityProjector -> LanguageModel. The attribute names vision_encoder, decoder and MP
    must stay as they are: they are the key prefixes of the lusxvr/nanoVLM-222M state_dict.
    """
    def __init__(self, cfg: VLMConfig, load_backbone: bool = True) -> None:
        """
        Args:
            cfg: Model config.
            load_backbone: Load pretrained SigLIP and SmolLM2 weights (True) or random init (False,
                used when a full VLM checkpoint is loaded afterwards).
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
        Encodes the image, projects it, embeds the text, combines both, and runs the decoder.

        With targets, returns the logits of the text positions only (image positions are dropped)
        and the cross-entropy over them, ignoring -100. targets are already shifted by the
        collator: targets[:, t] is the token after position t. Without targets, returns the
        decoder's hidden states for all positions and None.

        Args:
            input_ids: Text token ids, [B, T].
            image: Images, [B, 3, 224, 224].
            attention_mask: Text padding mask, [B, T], 1 = real token.
            targets: Shifted labels, [B, T], -100 where no loss is taken; or None.
        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: (logits [B, T, vocab] or hidden states, loss
                scalar or None).
        """
        image_embd = self.MP(self.vision_encoder(image))
        token_embd = self.decoder.token_embedding(input_ids)
        embeddings, mask = combine_embeddings(image_embd, token_embd, attention_mask)
        hidden = self.decoder(embeddings, mask)  # [B, N_img + T, D]

        if targets is None:
            return hidden, None

        logits = self.decoder.head(hidden[:, image_embd.size(1):, :])  # Text positions only: [B, T, vocab]
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
        Autoregressively generates max_new_tokens tokens after the prompt, keeping the image prefix.

        test_mmstar calls this, so its behaviour is part of the evaluation: no KV cache (the
        decoder reruns on the whole sequence each step), and the next token is sampled from the
        full softmax with torch.multinomial (no argmax, temperature or top-k). Always produces
        exactly max_new_tokens tokens (no early stop at EOS).

        Args:
            input_ids: Prompt token ids, [B, T] (left-padded).
            image: Images, [B, 3, 224, 224].
            attention_mask: Prompt padding mask, [B, T], 1 = real token.
            max_new_tokens: Number of tokens to generate.
        Returns:
            torch.Tensor: Generated token ids only (not the prompt), [B, max_new_tokens], same dtype
                as input_ids.
        """
        image_embd = self.MP(self.vision_encoder(image))
        token_embd = self.decoder.token_embedding(input_ids)
        embeddings, mask = combine_embeddings(image_embd, token_embd, attention_mask)

        batch_size = input_ids.size(0)
        generated = torch.zeros((batch_size, max_new_tokens), dtype=input_ids.dtype, device=input_ids.device)
        for i in range(max_new_tokens):
            hidden = self.decoder(embeddings, mask)
            logits = self.decoder.head(hidden[:, -1, :])  # [B, vocab]
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)  # [B, 1]
            generated[:, i] = next_token.squeeze(-1)

            embeddings = torch.cat((embeddings, self.decoder.token_embedding(next_token)), dim=1)
            if mask is not None:
                mask = torch.cat((mask, mask.new_ones((batch_size, 1))), dim=1)
                
        return generated

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.
            revision: Hub revision (branch, tag or commit) to load; None = default branch.

        Returns:
            VisionLanguageModel: The loaded model.

        Raises:
            ValueError: If a local directory lacks config.json or model.safetensors.
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

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
            private: Create the repo as private if it does not exist yet.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload TinyForge-Vision using push_to_hub",
            )

MODEL_CARD_TEMPLATE = """
---
library_name: pytorch
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**TinyForge-Vision** is a 222M-parameter vision-language model: a SigLIP-B/16-224 image encoder and
SmolLM2-135M, connected by a pixel-shuffle + linear projector, trained on the Cauldron. It
reproduces nanoVLM v0.1 (https://github.com/huggingface/nanoVLM), whose code it partly borrows.

**Usage** (from the TinyForge-Vision repository):

```python
from tfv.models.vlm import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
