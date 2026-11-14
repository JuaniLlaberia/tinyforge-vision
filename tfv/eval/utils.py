# Borrowed from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082, file models/utils.py.
# Changes: added docstrings and type hints. No logic changes.
# Part of the MMStar evaluation: never change its logic, or scores stop being comparable with 35.3%.
import re

def check_multiple_choice_with_regex(model_outputs: list[str], correct_answers: list[str]) -> list[bool]:
    """
    Lenient multiple-choice check: an output counts as correct if the correct letter appears as a
    word anywhere in it, followed by punctuation, or inside parentheses.

    Args:
        model_outputs: Decoded model answers.
        correct_answers: Decoded correct answer letters, same length.
    Returns:
        list[bool]: Whether each output matches its correct answer.
    """
    results = []
    for model_output, correct_answer in zip(model_outputs, correct_answers):
        correct_answer = correct_answer.upper()

        patterns = [
            rf"\b{correct_answer}\b",  # Word boundary around the answer letter
            rf"\b{correct_answer}[.,)]",  # Answer followed by punctuation
            rf"\(.*{correct_answer}.*\)",  # Answer within parentheses
        ]

        match_found = False
        for pattern in patterns:
            if re.search(pattern, model_output):
                match_found = True
                break 
        results.append(match_found)
        
    return results
