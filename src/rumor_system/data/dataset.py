from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rumor_system.data.preprocess import normalize_tweet


DEFAULT_EVENT_NAME_MAP = {
    0: "gurlitt",
    1: "ferguson",
    2: "essien_ebola",
    3: "prince_toronto",
    4: "germanwings",
    5: "sydneysiege",
    6: "ottawashooting",
}


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


def add_model_input_text(df: pd.DataFrame, raw_config: dict) -> pd.DataFrame:
    out = df.copy()
    model_cfg = raw_config.get("model", {})
    if not model_cfg.get("use_event_prefix", False):
        out["model_input_text"] = out["text"]
        return out

    event_name_map = {
        int(key): str(value)
        for key, value in raw_config.get("event_name_map", DEFAULT_EVENT_NAME_MAP).items()
    }

    def format_with_event(row: pd.Series) -> str:
        event_id = int(row["event"])
        event_name = event_name_map.get(event_id, f"event_{event_id}")
        return f"[event: {event_name}] {row['text']}"

    out["model_input_text"] = out.apply(format_with_event, axis=1)
    return out
