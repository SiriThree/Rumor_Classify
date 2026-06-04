from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests

from rumor_system.explain.prompting import build_explanation_prompt
from rumor_system.models.base import Prediction
from rumor_system.retrieval.lexical import EvidenceItem
from rumor_system.data.preprocess import simple_tokenize
from rumor_system.utils.env import load_local_env


ROLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("一条机构安全通知", ["courses", "exams", "lockdown", "stay indoors", "stay safe", "officially cancelled", "uottawa"]),
    ("一条官方通报或直播更新", ["statement", "news conference", "address", "watch live", "live on", "raw video", "update:", "breaking"]),
    ("一条对未证实指控的谣言式传播", ["what are police hiding", "conceal evidence", "smear the victim", "police lie", "checkthedatestamp", "martial law"]),
    ("一条情绪化或指控式评论", ["disgusted", "worst. police. ever", "not believing", "thoughts", "prayers"]),
]


@dataclass
class ExplanationGenerator:
    mode: str = "offline_template"
    model_name: str = ""

    def generate(
        self,
        text: str,
        prediction: Prediction,
        evidence: list[EvidenceItem],
    ) -> str:
        if self.mode == "sjtu_api":
            return self._generate_online(text, prediction, evidence)
        return self._generate_offline(text, prediction, evidence)

    def _generate_offline(
        self,
        text: str,
        prediction: Prediction,
        evidence: list[EvidenceItem],
    ) -> str:
        dominant = "rumor" if prediction.label == 1 else "non-rumor"
        if not evidence:
            return (
                f"该推文被判定为{label_to_zh(dominant)}，因为其整体表达更接近这一类别，"
                "但当前还没有足够强的相似检索证据，因此主要依据仍然来自文本本身。"
            )

        label_counts = {0: 0, 1: 0}
        for item in evidence:
            label_counts[item.label] += 1
        role = infer_role(text)
        trigger_phrases = extract_trigger_phrases(text)
        phrase_text = ", ".join(f'"{phrase}"' for phrase in trigger_phrases[:3]) if trigger_phrases else "its overall wording"
        overlap_terms = summarize_overlap_terms(text, evidence)
        overlap_text = ", ".join(overlap_terms[:4]) if overlap_terms else "similar wording and event context"
        evidence_majority = "rumor" if label_counts[1] > label_counts[0] else "non-rumor"
        evidence_balance = (
            f"{label_counts[1]} 条谣言样本、{label_counts[0]} 条非谣言样本"
        )
        top_support = (
            "与最终标签一致"
            if evidence[0].label == prediction.label
            else "与最终标签不完全一致，因此系统还结合了整体证据分布和分类器置信度"
        )
        return (
            f"该推文被判定为{label_to_zh(dominant)}，置信度为 {prediction.confidence:.2f}，因为它在传播中更像{role}。"
            f"其中，{phrase_text} 等表述是触发判断的重要文本信号。"
            f"检索到的相似样本中包含 {evidence_balance}，整体证据更偏向{label_to_zh(evidence_majority)}。"
            f"最相近样本与当前文本在 {overlap_text} 等词汇上存在重合，且 top evidence {top_support}。"
        )

    def _generate_online(
        self,
        text: str,
        prediction: Prediction,
        evidence: list[EvidenceItem],
    ) -> str:
        load_local_env()
        api_base = os.getenv("SJTU_API_BASE_URL", "").rstrip("/")
        api_key = os.getenv("SJTU_API_KEY", "")
        model_name = self.model_name or os.getenv("SJTU_API_MODEL", "")
        if not api_base or not api_key or not model_name:
            raise RuntimeError("SJTU API configuration is incomplete. Check your .env file.")

        prompt = build_explanation_prompt(text, prediction, evidence)
        response = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()


def infer_role(text: str) -> str:
    lower = text.lower()
    for role, patterns in ROLE_PATTERNS:
        if any(pattern in lower for pattern in patterns):
            return role
    if "?" in text:
        return "一条带有猜测或争议色彩的说法"
    if any(token in lower for token in ["reported", "reportedly", "breaking", "update"]):
        return "一条事件早期的新闻更新"
    return "一条需要结合措辞和相似样本共同理解立场的文本"


def extract_trigger_phrases(text: str) -> list[str]:
    lower = text.lower()
    phrases: list[str] = []
    candidate_patterns = [
        "officially cancelled",
        "lockdown still in effect",
        "stay indoors",
        "stay safe",
        "raw video",
        "watch live",
        "news conference",
        "what are police hiding",
        "checkthedatestamp",
        "conceal evidence",
        "smear the victim",
        "martial law",
        "police lie",
        "unarmed black kid",
        "not believing this story",
        "breaking",
        "reported",
        "reportedly",
        "thoughts and prayers",
    ]
    for pattern in candidate_patterns:
        if pattern in lower:
            phrases.append(pattern)
    if not phrases:
        short_chunks = re.findall(r"[A-Za-z#@][^,.!?;]{4,40}", text)
        for chunk in short_chunks[:3]:
            cleaned = chunk.strip().strip('"').strip("'")
            if cleaned:
                phrases.append(cleaned[:40])
    deduped: list[str] = []
    for phrase in phrases:
        if phrase not in deduped:
            deduped.append(phrase)
    return deduped[:4]


def summarize_overlap_terms(text: str, evidence: list[EvidenceItem]) -> list[str]:
    query_tokens = set(simple_tokenize(text))
    counts: dict[str, int] = {}
    for item in evidence[:3]:
        overlap = query_tokens & set(simple_tokenize(item.text))
        for token in overlap:
            if len(token) < 4 or token in {"http", "https", "user", "url"}:
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _ in ranked[:5]]


def label_to_zh(label_name: str) -> str:
    return "谣言" if label_name == "rumor" else "非谣言"
