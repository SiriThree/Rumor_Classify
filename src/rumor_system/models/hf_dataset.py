from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None = None


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

