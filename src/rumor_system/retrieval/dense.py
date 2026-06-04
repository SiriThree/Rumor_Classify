from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from rumor_system.retrieval.lexical import EvidenceItem

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    faiss = None


@dataclass
class DenseRetriever:
    train_df: pd.DataFrame
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64

    def __post_init__(self) -> None:
        self.train_df = self.train_df.copy()
        self.encoder = SentenceTransformer(self.model_name)
        texts = self.train_df["text"].tolist()
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self.embeddings = embeddings
        self.index = None
        if faiss is not None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)

    def search(self, text: str, top_k: int = 5) -> list[EvidenceItem]:
        query = self.encoder.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        if self.index is not None:
            scores, indices = self.index.search(query, top_k)
        else:
            similarities = np.matmul(self.embeddings, query[0])
            top_indices = np.argsort(-similarities)[:top_k]
            scores = np.array([similarities[top_indices]], dtype="float32")
            indices = np.array([top_indices], dtype="int64")
        evidence: list[EvidenceItem] = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.train_df.iloc[int(idx)]
            evidence.append(
                EvidenceItem(
                    tweet_id=str(row["tweet_id"]),
                    text=str(row["text"]),
                    label=int(row["label"]),
                    event=int(row["event"]),
                    score=float(score),
                    source="dense",
                    metadata={"dense_score": float(score)},
                )
            )
        return evidence
