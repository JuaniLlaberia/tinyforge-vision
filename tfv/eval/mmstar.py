# Borrowed from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082, file train.py (test_mmstar).
# Changes: added docstrings and type hints, import path (models.utils -> tfv.eval.utils). No logic changes.
# Never change its logic, or scores stop being comparable with 35.3%.
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

import tfv.eval.utils as utils

def test_mmstar(
    model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, test_loader: DataLoader, device: torch.device
) -> float:
    """
    MMStar accuracy: generates 5 tokens per question by sampling from
    the full softmax (not greedy, so results vary with the seed), then applies the lenient regex
    check. Puts the model back in train mode afterwards.

    Args:
        model: The VLM; must provide generate(input_ids, image, attention_mask).
        tokenizer: Tokenizer used to decode outputs and answers.
        test_loader: DataLoader over MMStarDataset with MMStarCollator.
        device: Device to run on.
    Returns:
        float: Fraction of questions answered correctly.
    """
    model.eval()
    total_examples = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
            image = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            correct_answer = tokenizer.batch_decode(labels, skip_special_tokens=True)

            gen = model.generate(input_ids, image, attention_mask)
            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)

            is_correct = utils.check_multiple_choice_with_regex(model_output, correct_answer)

            total_examples += len(is_correct)
            if is_correct:
                correct_predictions += sum(is_correct)

    model.train()
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    
    return accuracy
