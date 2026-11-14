# Borrowed from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082, file data/datasets.py.
# Changes: blank lines (2 -> 1), added docstrings and type hints, import path (models.config ->
# tfv.config). No logic changes.
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from transformers import PreTrainedTokenizerBase

import tfv.config as cfg

class VQADataset(Dataset):
    """
    Wraps a Cauldron split: returns the first image (preprocessed) and the first QA pair, with the
    question formatted as "Question: {question} Answer:" and EOS appended to the answer.
    """
    def __init__(self, dataset: Any, tokenizer: PreTrainedTokenizerBase, image_processor: Compose) -> None:
        """
        Args:
            dataset: HF dataset with "images" and "texts" columns (Cauldron schema).
            tokenizer: Tokenizer, used for its EOS token.
            image_processor: Maps a PIL image to a [3, H, W] tensor.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self) -> int:
        """
        Returns:
            int: Number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Args:
            idx: Sample index.
        Returns:
            dict[str, Any]: {"image": [3, H, W] tensor, "text_data": formatted question (str),
                "answer": answer + EOS (str)}.
        """
        item = self.dataset[idx]

        image_data = item["images"]
        if isinstance(image_data, list) and len(image_data) > 0:
            image = image_data[0]
        else:
            image = image_data

        # Process the image
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)

        # Process text
        text_data = item["texts"]
        if isinstance(text_data, list) and len(text_data) > 0:
            text = text_data[0]
        else:
            text = text_data

        question = text["user"]
        answer = text["assistant"] + self.tokenizer.eos_token

        formatted_text = f"Question: {question} Answer:"

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }

class MMStarDataset(Dataset):
    """
    Wraps the MMStar split: returns the preprocessed image, the multiple-choice question formatted
    as a prompt, and the answer letter + EOS.
    """
    def __init__(self, dataset: Any, tokenizer: PreTrainedTokenizerBase, image_processor: Compose) -> None:
        """
        Args:
            dataset: HF dataset with "image", "question" and "answer" columns.
            tokenizer: Tokenizer, used for its EOS token.
            image_processor: Maps a PIL image to a [3, H, W] tensor.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self) -> int:
        """
        Returns:
            int: Number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Args:
            idx: Sample index.
        Returns:
            dict[str, Any]: {"image": [3, H, W] tensor, "text_data": formatted question (str),
                "answer": answer letter + EOS (str)}.
        """
        item = self.dataset[idx]

        image = item["image"]

        # Process the image
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)

        question = item["question"]
        answer = item["answer"] + self.tokenizer.eos_token 

        formatted_text = f"Question: {question} \nAnswer only with the letter! \nAnswer:"

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }
