# Final System Plan

## Goal

Build a system that maximizes two outcomes:

1. Validation-set rumor detection accuracy as high as possible
2. Explanations that are evidence-grounded, specific, and accurate

## Final Architecture

```text
Input tweet
  -> Preprocessing and normalization
  -> Transformer classifier
       outputs: label probability
  -> Hybrid retriever over train.csv
       outputs: top-k similar labeled samples
  -> Fusion decision module
       combines classifier confidence with retrieval vote
  -> Explanation module
       uses retrieved evidence plus prediction to produce rationale
Final output:
  label, confidence, evidence, explanation
```

## Why This Architecture

- The `Transformer` handles semantic understanding and non-trivial paraphrases.
- The `Retriever` exploits same-event and near-duplicate structure in the dataset.
- The `Fusion` layer pushes accuracy higher than a single model.
- The `Explanation` layer is grounded in concrete retrieved samples rather than free-form guessing.

## Recommended High-Score Version

### Stage 1: Strong classifier

- Backbone: `vinai/bertweet-base`
- Backup comparison model: `microsoft/deberta-v3-base`
- Training:
  - weighted cross-entropy
  - early stopping on validation accuracy
  - 3 random seeds
  - model selection by mean accuracy and stability

### Stage 2: Hybrid RAG retrieval

- Sparse retriever:
  - BM25 or TF-IDF for lexical overlap
- Dense retriever:
  - sentence embeddings plus FAISS
- Fusion:
  - reciprocal rank fusion or weighted score fusion
- Output:
  - top-k evidence tweets
  - label distribution
  - similarity scores

### Stage 3: Decision fusion

Use retrieval to improve accuracy, not just explanation quality.

Suggested logic:

1. If a near-duplicate evidence item exists and similar items strongly agree, trust retrieval.
2. Otherwise trust the Transformer when confidence is high.
3. If the Transformer is uncertain, use evidence voting to adjust the final label.

### Stage 4: Faithful explanation generation

Prompt inputs:

- original tweet
- final prediction
- classifier confidence
- retrieved evidence items
- short evidence summary

Prompt requirement:

- explanation must mention concrete reasons
- explanation must not invent unseen facts
- explanation should refer to wording patterns and retrieved support

## Evaluation Plan

### Main metric

- Validation accuracy on `val.csv`

### Supporting metrics

- per-event accuracy
- confusion matrix
- explanation faithfulness spot checks
- explanation coverage: whether a rationale cites retrieved evidence

### Ablation experiments

1. retrieval-only baseline
2. Transformer-only
3. Transformer + sparse retrieval
4. Transformer + sparse + dense retrieval
5. final system with explanation generation

## Implementation Priorities

1. Get retrieval-only pipeline fully runnable
2. Add Transformer training and prediction
3. Add retrieval-classifier fusion
4. Add online explanation generation through SJTU API
5. Tune thresholds on validation performance

## Risks and Mitigations

- `Risk`: same text can have conflicting labels
  - `Mitigation`: use multi-neighbor voting and confidence thresholds
- `Risk`: LLM explanations sound good but drift from evidence
  - `Mitigation`: supply structured evidence and require grounded output
- `Risk`: model overfits one event pattern
  - `Mitigation`: analyze per-event performance and keep event out of final user input

