# Attention

PyTorch implementations of masked softmax, dot-product attention, additive attention, and an attention-based English-to-French sequence-to-sequence model.

## Project structure

- `src/attention.py` — dot-product and additive attention layers
- `src/ops.py` — sequence masking and masked softmax
- `src/mt/` — translation data pipeline, bidirectional GRU encoder, and attention decoder
- `notebooks/Attention.ipynb` — attention mechanisms and visualizations
- `notebooks/Ops.ipynb` — masking operations
- `notebooks/machine_translation.ipynb` — data preparation, training, translation, attention heatmaps, and BLEU evaluation

## Results

| Model | Validation split | Decoding | Metric | Score |
| --- | ---: | --- | --- | ---: |
| Unidirectional GRU baseline with additive attention | 20% | Greedy | Corpus BLEU-4 | **38.02** |

The baseline score uses add-one smoothing and sequences truncated or padded to 20 tokens. Retrain the bidirectional model to measure its result.

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

The machine-translation notebook downloads and caches the French-English corpus under `notebooks/data` automatically.

## Usage

Open a notebook in your preferred Jupyter-compatible editor and select the environment created at `.venv`.

To verify the source and notebooks with BasedPyright:

```bash
uv run basedpyright
```
