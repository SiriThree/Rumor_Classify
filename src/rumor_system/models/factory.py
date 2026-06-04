from __future__ import annotations

from rumor_system.models.base import BaseClassifier
from rumor_system.models.retrieval_vote import RetrievalVoteClassifier
from rumor_system.models.transformer import TransformerClassifier
from rumor_system.retrieval.lexical import LexicalRetriever


def build_classifier(config, retriever: LexicalRetriever) -> BaseClassifier:
    classifier_type = config.model["classifier_type"]
    if classifier_type == "transformer":
        return TransformerClassifier(
            model_name=config.model["classifier_name"],
            checkpoint_dir=config.model["checkpoint_dir"],
            num_labels=config.model["num_labels"],
            max_length=config.model["max_length"],
            batch_size=config.model["batch_size"],
            eval_batch_size=config.model["eval_batch_size"],
            learning_rate=float(config.model["learning_rate"]),
            weight_decay=float(config.model["weight_decay"]),
            num_epochs=int(config.model["num_epochs"]),
            warmup_ratio=float(config.model["warmup_ratio"]),
            gradient_clip_norm=float(config.model["gradient_clip_norm"]),
            early_stopping_patience=int(config.model["early_stopping_patience"]),
            use_class_weights=bool(config.model["use_class_weights"]),
            device=config.model["device"],
        )
    if classifier_type == "retrieval_vote":
        return RetrievalVoteClassifier(retriever=retriever, top_k=config.retrieval["top_k"])
    raise ValueError(f"Unsupported classifier_type: {classifier_type}")

