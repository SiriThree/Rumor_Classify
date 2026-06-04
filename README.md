# Explainable Rumor Detection

This project targets the 2026 Introduction to AI course assignment: rumor detection with explanations. The final system is designed as a hybrid pipeline:

- A `Transformer` classifier predicts `0/1`.
- A `RAG` retriever finds similar training tweets as evidence.
- An `LLM` generator turns the prediction and retrieved evidence into a faithful explanation.

## Repository Layout

```text
configs/                 default configuration
docs/                    system design and report support docs
outputs/                 checkpoints, metrics, and generated explanations
rumer2026/               assignment dataset
scripts/                 CLI entry points
src/rumor_system/        project source code
```

## Recommended Final Stack

- Classifier: `vinai/bertweet-base` or `microsoft/deberta-v3-base`
- Fine-tuning: `transformers` + `peft`
- Retrieval: lexical retrieval + dense retrieval fusion
- Explanation: SJTU model API with evidence-grounded prompting

The scaffold currently includes a lightweight lexical retriever and fusion logic so the pipeline can be wired up before GPU training is ready.

## Quick Start

1. Create an environment and install dependencies.

```bash
pip install -r requirements.txt
```

2. Put the real API credentials into `.env`.

```bash
copy .env.example .env
```

3. Run a dry pipeline evaluation.

```bash
python scripts/evaluate.py --config configs/default.yaml
```

4. Train the baseline retrieval-backed classifier summary.

```bash
python scripts/train.py --config configs/default.yaml
```

5. Generate predictions and explanations for validation data.

```bash
python scripts/infer.py --config configs/default.yaml --split val
```

## Transformer Training

Use `configs/bertweet.yaml` for the tweet-specialized backbone:

```bash
python scripts/train.py --config configs/bertweet.yaml
python scripts/evaluate.py --config configs/bertweet.yaml
python scripts/infer.py --config configs/bertweet.yaml --split val
```

Use `configs/deberta.yaml` for the stronger general-purpose comparison model:

```bash
python scripts/train.py --config configs/deberta.yaml
```

The evaluation and inference scripts automatically use the trained checkpoint if it exists. If no checkpoint is found yet, the pipeline falls back to retrieval-only prediction.

To overwrite an existing checkpoint, use:

```bash
python scripts/train.py --config configs/bertweet.yaml --force-retrain
```

## Retrieval Baselines

Use `configs/retrieval_hybrid.yaml` to evaluate the hybrid sparse+dense retriever without a Transformer checkpoint:

```bash
python scripts/evaluate.py --config configs/retrieval_hybrid.yaml
```

## Current Implementation Status

- `Implemented now`
  - dataset loading
  - tweet normalization
  - lexical retrieval index
  - dense retrieval with sentence-transformers
  - hybrid retrieval fusion
  - Transformer training and checkpoint loading
  - Transformer + retrieval fusion prediction
  - retrieval-aware decision fusion
  - evidence packaging
  - explanation prompt builder
  - offline explanation fallback

- `To upgrade next`
  - dense embedding retrieval with FAISS
  - reranker for top-k evidence calibration
  - online LLM explanation generation via SJTU API

## Suggested Team Split

- Member A: data pipeline, validation, metrics
- Member B: Transformer fine-tuning and hyperparameter search
- Member C: RAG retrieval, explanation module, SJTU API integration
- Member D: README, experiments, report, deployment instructions
