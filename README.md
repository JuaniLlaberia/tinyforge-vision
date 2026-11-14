# TinyForge-Vision

A small vision-language model (222M parameters) that reproduces
[nanoVLM](https://github.com/huggingface/nanoVLM) v0.1: a pretrained SigLIP image encoder and
SmolLM2 language model, connected by a pixel-shuffle projector and trained together on the Cauldron.

## Model

| Part | Choice |
| --- | --- |
| Vision encoder | SigLIP-B/16-224 (`google/siglip-base-patch16-224`): 196 patches x 768 |
| Projector | Pixel shuffle (2x2 patches into one token) + linear layer: 49 image tokens x 576 |
| Language model | SmolLM2-135M (`HuggingFaceTB/SmolLM2-135M`) |
| Prompt | Image tokens, then `Question: {question} Answer:` |
| Training data | The Cauldron, 45 subsets, 1.67M samples, first QA pair per image |
| Training | 5 epochs, batch 256, AdamW (projector lr 2e-3, backbones 1e-4), cosine schedule, bf16 |
| Evaluation | MMStar (1,500 multiple-choice questions), answers sampled, 3 seeds |

## Results

_To be filled in after training._

| Model | MMStar (mean ± std, 3 seeds) | Val loss | Training time |
| --- | --- | --- | --- |
| nanoVLM-222M, published | 35.3% | - | ~6 h on one H100 |
| nanoVLM-222M, scored with this repo's eval | | - | - |
| TinyForge-Vision | | | |

Sample answers:

## Conclusions

_To be filled in after training._

## Repository layout

```
tfv/
  config.py            all settings (model, training, evaluation, data preparation)
  models/              vision encoder, language model, projector, VLM
  data/                Cauldron preprocessing, datasets, collators, loaders
  train/               training loop, optimizer groups, LR scheduler, checkpoints
  eval/                MMStar evaluation
scripts/               data preparation, MMStar evaluation, generation
notebooks/             Colab notebook for training on a GPU
tests/                 tests, with reference code in tests/reference/
```

## Usage

Setup (Python 3.12, [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run pytest -m "not slow"   # fast tests; drop -m to include the ones that download the checkpoint
```

Prepare the data once (needs `hf auth login`; resumable):

```bash
uv run python scripts/prepare_cauldron.py
```

Train: on Colab with `notebooks/train_colab.ipynb`, or from the command line:

```bash
uv run python -m tfv.train.train --set max_steps=100 --set log_wandb=False
```

Evaluate on MMStar, and generate answers for one image:

```bash
uv run python scripts/eval_mmstar.py --model checkpoints/<run>/final --seeds 0 1 2
curl -sL -o assets/image.png https://raw.githubusercontent.com/huggingface/nanoVLM/v0.1/assets/image.png
uv run python scripts/generate.py --checkpoint checkpoints/<run>/final --image assets/image.png
```

## Credits

The vision encoder, language model, datasets, image processing and MMStar evaluation come from
[nanoVLM](https://github.com/huggingface/nanoVLM) (tag `v0.1`, Apache-2.0) by the Hugging Face team,
and the published checkpoint is [lusxvr/nanoVLM-222M](https://huggingface.co/lusxvr/nanoVLM-222M).
Each borrowed file names its source in a header; see [NOTICE](NOTICE) for what was changed. The
training loop is adapted from my earlier project TinyForge.

## License

Apache-2.0, see [LICENSE](LICENSE).
