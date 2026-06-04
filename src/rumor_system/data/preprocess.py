from __future__ import annotations

import re


URL_RE = re.compile(r"https?://\S+")
USER_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
SPACE_RE = re.compile(r"\s+")


def normalize_tweet(
    text: str,
    normalize_urls: bool = True,
    normalize_users: bool = True,
    normalize_hashtags: bool = False,
) -> str:
    text = text.strip()
    if normalize_urls:
        text = URL_RE.sub(" URL ", text)
    if normalize_users:
        text = USER_RE.sub(" USER ", text)
    if normalize_hashtags:
        text = HASHTAG_RE.sub(r" \1 ", text)
    text = SPACE_RE.sub(" ", text)
    return text.lower().strip()


def simple_tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9_]+", text.lower()) if token]

