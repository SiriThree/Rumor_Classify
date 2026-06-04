from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rumor_system.data.preprocess import normalize_tweet


@dataclass
class TweetRecord:
    tweet_id: str
    text: str
    label: int
    event: int
    normalized_text: str


def load_split(csv_path: str | Path, config: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    text_col = config["text_column"]
    id_col = config["id_column"]
    label_col = config["label_column"]
    event_col = config["event_column"]

    df = df.rename(
        columns={
            id_col: "tweet_id",
            text_col: "text",
            label_col: "label",
            event_col: "event",
        }
    )
    return df


def add_normalized_text(df: pd.DataFrame, retrieval_cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out["normalized_text"] = out["text"].map(
        lambda x: normalize_tweet(
            x,
            normalize_urls=retrieval_cfg["normalize_urls"],
            normalize_users=retrieval_cfg["normalize_users"],
            normalize_hashtags=retrieval_cfg["normalize_hashtags"],
        )
    )
    return out

