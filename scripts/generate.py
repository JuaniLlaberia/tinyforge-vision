# Adapted from huggingface/nanoVLM (Apache-2.0), tag v0.1 @ 6ba9082, file generate.py.
# Changes: docstrings and type hints, import paths, an --impl flag (reference model or ours),
# and the repo root on sys.path for it. assets/image.png is its example image.
"""
Generates answers for one image and prompt, several times (generation samples, so answers vary).

Usage:
    uv run python scripts/generate.py [--impl ours|v01] [--checkpoint DIR] [--image assets/image.png] [--prompt "What is this?"]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repo root, so --impl v01 can import tests.reference

import torch
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from tfv.data.processors import get_tokenizer, get_image_processor

def parse_args() -> argparse.Namespace:
    """
    Returns:
        argparse.Namespace: impl, checkpoint, hf_model, image, prompt, generations, max_new_tokens.
    """
    parser = argparse.ArgumentParser(
        description="Generate text from an image with TinyForge-Vision")
    parser.add_argument("--impl", choices=["ours", "v01"], default="ours",
                        help="Our tfv model or the reference model in tests/reference")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local save_pretrained directory. If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-222M",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--image", type=str, default="assets/image.png",
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default="What is this?",
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Maximum number of tokens per output")
    return parser.parse_args()

def main() -> None:
    """
    Loads the model, encodes the prompt as "Question: {prompt} Answer:", and prints the generations.
    """
    args = parse_args()
    if args.impl == "v01":
        from tests.reference.vision_language_model import VisionLanguageModel
    else:
        from tfv.models.vlm import VisionLanguageModel

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    template = f"Question: {args.prompt} Answer:"
    encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded["input_ids"].to(device)

    img = Image.open(args.image).convert("RGB")
    img_t = image_processor(img).unsqueeze(0).to(device)

    print("\nInput:\n ", args.prompt, "\n\nOutputs:")
    for i in range(args.generations):
        gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        print(f"  >> Generation {i+1}: {out}")

if __name__ == "__main__":
    main()
