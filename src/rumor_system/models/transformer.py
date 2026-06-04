from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from rumor_system.models.base import BaseClassifier, Prediction
from rumor_system.models.hf_dataset import TextClassificationDataset


@dataclass
class TransformerClassifier(BaseClassifier):
    model_name: str
    checkpoint_dir: str
    num_labels: int = 2
    tokenizer_name: str | None = None
    max_length: int = 128
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 2
    use_class_weights: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        self._device = self._resolve_device(self.device)
        self.tokenizer = None
        self.model = None

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def _checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)

    def _tokenizer_source(self) -> str:
        return self.tokenizer_name or self.model_name

    def is_trained(self) -> bool:
        checkpoint = self._checkpoint_path()
        return checkpoint.exists() and (checkpoint / "config.json").exists()

    def _load_tokenizer(self) -> Any:
        source = self._tokenizer_source()
        try:
            return AutoTokenizer.from_pretrained(source, use_fast=True)
        except Exception:
            return AutoTokenizer.from_pretrained(source, use_fast=False)

    def _load_model(self) -> Any:
        source = self._checkpoint_path() if self.is_trained() else self.model_name
        model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=self.num_labels)
        return model.to(self._device)

    def _ensure_loaded(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = self._load_tokenizer()
        if self.model is None:
            self.model = self._load_model()

    def _build_dataloader(
        self,
        texts: list[str],
        labels: list[int] | None,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        self._ensure_loaded()
        dataset = TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _class_weight_tensor(self, labels: list[int]) -> torch.Tensor:
        counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=self.num_labels).float()
        counts = torch.clamp(counts, min=1.0)
        weights = counts.sum() / (self.num_labels * counts)
        return weights.to(self._device)

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str] | None = None,
        val_labels: list[int] | None = None,
    ) -> dict[str, float]:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_source(), use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_source(), use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        ).to(self._device)

        train_loader = self._build_dataloader(train_texts, train_labels, self.batch_size, shuffle=True)
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_loader = self._build_dataloader(val_texts, val_labels, self.eval_batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = max(1, len(train_loader) * self.num_epochs)
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        class_weights = self._class_weight_tensor(train_labels) if self.use_class_weights else None
        best_val_accuracy = -1.0
        best_epoch = -1
        patience_counter = 0
        history: dict[str, float] = {}

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                batch = {key: value.to(self._device) for key, value in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                if class_weights is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits,
                        batch["labels"],
                        weight=class_weights.to(dtype=logits.dtype),
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                running_loss += float(loss.item())

            history[f"train_loss_epoch_{epoch + 1}"] = running_loss / max(1, len(train_loader))

            if val_loader is None:
                continue

            val_metrics = self.evaluate_loader(val_loader)
            history[f"val_accuracy_epoch_{epoch + 1}"] = val_metrics["accuracy"]
            history[f"val_loss_epoch_{epoch + 1}"] = val_metrics["loss"]

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                best_epoch = epoch + 1
                patience_counter = 0
                self.save()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        if best_epoch == -1:
            self.save()
            best_epoch = self.num_epochs
            best_val_accuracy = history.get(f"val_accuracy_epoch_{self.num_epochs}", -1.0)

        self.load()
        history["best_epoch"] = float(best_epoch)
        history["best_val_accuracy"] = float(best_val_accuracy)
        return history

    @torch.inference_mode()
    def evaluate_loader(self, dataloader: DataLoader) -> dict[str, float]:
        self._ensure_loaded()
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch in dataloader:
            batch = {key: value.to(self._device) for key, value in batch.items()}
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
            preds = logits.argmax(dim=-1)
            total_loss += float(loss.item())
            correct += int((preds == batch["labels"]).sum().item())
            total += int(batch["labels"].size(0))
        return {
            "accuracy": correct / total if total else 0.0,
            "loss": total_loss / max(1, len(dataloader)),
        }

    def save(self) -> None:
        self._ensure_loaded()
        checkpoint = self._checkpoint_path()
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint)
        self.tokenizer.save_pretrained(checkpoint)

    def load(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_source(), use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_source(), use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self._checkpoint_path()).to(self._device)

    @torch.inference_mode()
    def predict_one(self, text: str) -> Prediction:
        self._ensure_loaded()
        self.model.eval()
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        confidence, label = torch.max(probs, dim=-1)
        return Prediction(label=int(label.item()), confidence=float(confidence.item()), source="transformer")
