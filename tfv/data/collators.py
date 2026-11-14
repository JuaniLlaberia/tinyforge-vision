# Mixed file. VQACollator: written for TinyForge-Vision.
# MMStarCollator: borrowed from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082, file
# data/collators.py (changes: double quotes, added docstrings and type hints; no logic changes). It is part of the
# MMStar evaluation: never change its logic.
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

class VQACollator:
    """
    Turns VQADataset samples into a training batch with answer-only labels.

    - Each input sequence is the question text followed directly by the answer text (the answer
      already ends with EOS), left-padded to exactly max_length tokens and truncated if longer.
    - Labels are shifted here: labels[:, t] is the token to predict after position t
      (= input_ids[:, t + 1]); the last position has no target (-100).
    - Only answer tokens are targets: every label that predicts a padding or question token is -100.
    - A truncated sample (full sequence longer than max_length) gets all labels -100, so it
      contributes no loss.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        """
        Args:
            tokenizer: Tokenizer with pad_token set (pad = EOS).
            max_length: Text length after padding/truncation (lm_max_length, 79).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: VQADataset samples, each {"image": [3, H, W], "text_data": str, "answer": str}.
        Returns:
            dict[str, torch.Tensor]: {"image": [B, 3, H, W], "input_ids": [B, max_length],
                "attention_mask": [B, max_length], "labels": [B, max_length]}.
        """
        images = torch.stack([item["image"] for item in batch])
        questions = [item["text_data"] for item in batch]
        sequences = [item["text_data"] + item["answer"] for item in batch]

        encoded = self.tokenizer(sequences, padding="max_length", padding_side="left", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]

        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]  # labels[:, t] = token after position t; last position has none

        for i, (question, sequence) in enumerate(zip(questions, sequences)):
            if len(self.tokenizer.encode(sequence)) > self.max_length:
                labels[i] = -100
                continue

            first_real = int(attention_mask[i].nonzero()[0])  # Left padding: first 1 in the mask
            question_len = len(self.tokenizer.encode(question, add_special_tokens=False))
            # Labels before this index predict padding or question tokens (-1 for the shift)
            labels[i, :first_real + question_len - 1] = -100

        return {"image": images, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class MMStarCollator(object):
    """
    Batches MMStarDataset samples for evaluation: left-padded questions as inputs and the
    left-padded answer tokens as labels (decoded back to text by test_mmstar).
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Args:
            tokenizer: Tokenizer with pad_token set (pad = EOS).
        """
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: MMStarDataset samples, each {"image": [3, H, W], "text_data": str, "answer": str}.
        Returns:
            dict[str, torch.Tensor]: {"images": [B, 3, H, W], "input_ids": [B, T_q],
                "attention_mask": [B, T_q], "labels": answer token ids [B, T_a]}.
        """
        images = [item["image"] for item in batch]
        questions = [item["text_data"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Stack images
        images = torch.stack(images)

        encoded_question_sequences = self.tokenizer.batch_encode_plus(
            questions,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )

        encoded_answer_sequences = self.tokenizer.batch_encode_plus(
            answers,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )

        return {
            "images": images,
            "input_ids": encoded_question_sequences["input_ids"],
            "attention_mask": encoded_question_sequences["attention_mask"],
            "labels": encoded_answer_sequences["input_ids"],
        }
