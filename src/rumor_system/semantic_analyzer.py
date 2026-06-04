from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from rumor_system.models.base import Prediction
from rumor_system.retrieval.lexical import EvidenceItem
from rumor_system.utils.env import load_local_env


@dataclass
class SemanticAnalysis:
    claim_type: str
    verification_status: str
    propagation_style: str
    tone: str
    rumor_risk: str
    semantic_label_bias: str
    override_strength: str
    retrieval_reliability: str
    trust_retrieval_more: bool
    short_reason: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "verification_status": self.verification_status,
            "propagation_style": self.propagation_style,
            "tone": self.tone,
            "rumor_risk": self.rumor_risk,
            "semantic_label_bias": self.semantic_label_bias,
            "override_strength": self.override_strength,
            "retrieval_reliability": self.retrieval_reliability,
            "trust_retrieval_more": self.trust_retrieval_more,
            "short_reason": self.short_reason,
            "source": self.source,
        }


def build_semantic_prompt(
    text: str,
    base_prediction: Prediction,
    current_prediction: Prediction,
    evidence: list[EvidenceItem],
) -> str:
    evidence_lines = []
    for idx, item in enumerate(evidence[:3], start=1):
        evidence_lines.append(
            f"{idx}. label={item.label}, score={item.score:.3f}, text={item.text}"
        )
    joined_evidence = "\n".join(evidence_lines) if evidence_lines else "None"
    few_shot_examples = (
        "Few-shot examples under this dataset's annotation style:\n"
        "Example 1:\n"
        'tweet: "#BREAKING: Shooting reported at the War Memorial in Ottawa"\n'
        "gold_label: non-rumor\n"
        "analysis: This is a short breaking-news update about an ongoing event. It reads like an event alert rather than an unverified rumor amplification.\n"
        'json: {"claim_type":"factual_update","verification_status":"verification_unclear","propagation_style":"headline_news","tone":"alarmist","rumor_risk":"low","semantic_label_bias":"support_non_rumor","override_strength":"strong","retrieval_reliability":"medium","trust_retrieval_more":true,"short_reason":"This is a breaking-event update rather than a rumor-style amplification."}\n'
        "Example 2:\n"
        'tweet: "The PM\'s office releases a statement about #sydneysiege."\n'
        "gold_label: rumor\n"
        "analysis: Even though the text sounds official, in this dataset early-event statement reports can still belong to a rumor thread. Do not assume official wording automatically means non-rumor.\n"
        'json: {"claim_type":"official_statement","verification_status":"verification_unclear","propagation_style":"headline_news","tone":"neutral","rumor_risk":"medium","semantic_label_bias":"uncertain","override_strength":"weak","retrieval_reliability":"medium","trust_retrieval_more":false,"short_reason":"Official-sounding wording alone is not enough to force a non-rumor label in this dataset."}\n'
        "Example 3:\n"
        'tweet: "Someone with better eyes than me please #checkthedatestamp on the bottom left photo. Isn\'t that JUNE? #Ferguson"\n'
        "gold_label: rumor\n"
        "analysis: This is speculative commentary that invites doubt and spreads an unverified implication. It is not a neutral factual update.\n"
        'json: {"claim_type":"unverified_claim","verification_status":"likely_unverified","propagation_style":"commentary","tone":"accusatory","rumor_risk":"high","semantic_label_bias":"support_rumor","override_strength":"strong","retrieval_reliability":"medium","trust_retrieval_more":false,"short_reason":"The tweet amplifies an unverified suspicion rather than reporting a confirmed fact."}\n'
        "Example 4:\n"
        'tweet: "Australian prime minister Tony Abbott to hold news conference live on Sky News at 1.30am GMT about ongoing Sydney hostage situation"\n'
        "gold_label: non-rumor\n"
        "analysis: This is a scheduled official/live-update notice. It may mention a crisis, but the communicative role is a routine event update rather than rumor spread.\n"
        'json: {"claim_type":"official_statement","verification_status":"appears_verified","propagation_style":"live_update","tone":"neutral","rumor_risk":"low","semantic_label_bias":"support_non_rumor","override_strength":"strong","retrieval_reliability":"high","trust_retrieval_more":true,"short_reason":"The tweet announces an official live update rather than amplifying an unverified claim."}\n'
        "Example 5:\n"
        'tweet: "Hashtags #4U9525 #GermanWings #A320 all useful for latest info on plane crash in southern France."\n'
        "gold_label: rumor\n"
        "analysis: Although the text looks informational, in this dataset early-event circulation of unverified crash information can still be labeled as rumor. Early informative wording does not guarantee non-rumor.\n"
        'json: {"claim_type":"factual_update","verification_status":"verification_unclear","propagation_style":"headline_news","tone":"neutral","rumor_risk":"medium","semantic_label_bias":"uncertain","override_strength":"weak","retrieval_reliability":"medium","trust_retrieval_more":false,"short_reason":"Informational wording in an early-event thread may still belong to rumor propagation."}\n'
    )
    return (
        "You are analyzing a hard rumor-detection case. Return only valid JSON.\n"
        "Focus on these frequent failure modes:\n"
        "1. headline_false_alarm: breaking/update/news-style text that is actually a factual or official update rather than a rumor.\n"
        "2. opinion_like_rumor_miss: emotional or commentary-like text that still spreads or amplifies an unverified claim.\n"
        "3. evidence_conflict: the tweet semantics disagree with the current prediction, but retrieved evidence may support a different label.\n"
        "4. retrieval_supports_error: retrieved texts are lexically similar but semantically misleading for the current tweet.\n"
        "Do not classify based only on surface words like BREAKING or UPDATE. Distinguish:\n"
        "- official statement / factual update / live coverage\n"
        "- unverified claim / rumor-like amplification / speculative commentary\n"
        "- emotional reaction that does not itself make the tweet a rumor\n"
        "- early-event informative wording that may still belong to a rumor thread in this dataset\n"
        "Important label semantics:\n"
        "- rumor here does NOT simply mean false news.\n"
        "- non-rumor here does NOT simply mean true news.\n"
        "- judge whether the tweet functions like part of an unverified rumor-style propagation thread under this dataset's annotation style.\n"
        "- do NOT rely only on general world knowledge or what later became true.\n"
        f"{few_shot_examples}\n"
        "Use this schema exactly:\n"
        "{"
        '"claim_type": "factual_update|official_statement|unverified_claim|opinion_commentary|eyewitness_report|unclear", '
        '"verification_status": "appears_verified|verification_unclear|likely_unverified", '
        '"propagation_style": "headline_news|live_update|commentary|amplification|discussion|unclear", '
        '"tone": "neutral|emotional|alarmist|accusatory|unclear", '
        '"rumor_risk": "low|medium|high", '
        '"semantic_label_bias": "support_non_rumor|support_rumor|uncertain", '
        '"override_strength": "weak|medium|strong", '
        '"retrieval_reliability": "low|medium|high", '
        '"trust_retrieval_more": true, '
        '"short_reason": "one sentence"'
        "}\n"
        f"Tweet: {text}\n"
        f"Base prediction: label={base_prediction.label}, confidence={base_prediction.confidence:.3f}\n"
        f"Current fused prediction: label={current_prediction.label}, confidence={current_prediction.confidence:.3f}\n"
        f"Retrieved evidence:\n{joined_evidence}\n"
        "Guidelines:\n"
        "- If the tweet mainly reports an official update, scheduled announcement, live coverage, or confirmed factual update, prefer support_non_rumor.\n"
        "- If the tweet amplifies an unverified allegation, speculative claim, or emotionally framed unverified assertion, prefer support_rumor.\n"
        "- If the tweet is merely emotional reaction without spreading a new unverified claim, avoid over-calling it rumor.\n"
        "- If the tweet sounds informational but clearly belongs to an early-event unverified information stream, you may keep it uncertain instead of automatically preferring non-rumor.\n"
        "- If retrieved evidence is semantically close and more trustworthy than the current prediction, trust_retrieval_more can be true.\n"
        "- Use override_strength=strong only when the semantic reading is clear enough to change the label even if the current model is confident.\n"
        "- Use retrieval_reliability=low when the evidence looks lexically similar but semantically mismatched.\n"
        "Return JSON only."
    )


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@dataclass
class SemanticAnalyzer:
    enabled: bool = False
    mode: str = "offline_rule"
    model_name: str = ""
    cache_path: str = ""

    def __post_init__(self) -> None:
        self._cache: dict[str, SemanticAnalysis] = {}
        self._cache_dirty = False
        if self.cache_path:
            self._load_cache()

    def analyze(
        self,
        text: str,
        base_prediction: Prediction,
        current_prediction: Prediction,
        evidence: list[EvidenceItem],
    ) -> SemanticAnalysis | None:
        if not self.enabled:
            return None

        cache_key = self._build_cache_key(text, base_prediction, current_prediction, evidence)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return SemanticAnalysis(
                claim_type=cached.claim_type,
                verification_status=cached.verification_status,
                propagation_style=cached.propagation_style,
                tone=cached.tone,
                rumor_risk=cached.rumor_risk,
                semantic_label_bias=cached.semantic_label_bias,
                override_strength=cached.override_strength,
                retrieval_reliability=cached.retrieval_reliability,
                trust_retrieval_more=cached.trust_retrieval_more,
                short_reason=cached.short_reason,
                source="cache",
            )

        if self.mode == "sjtu_api":
            try:
                result = self._analyze_online(text, base_prediction, current_prediction, evidence)
            except Exception:
                result = self._analyze_offline(text, base_prediction, current_prediction, evidence)
        else:
            result = self._analyze_offline(text, base_prediction, current_prediction, evidence)

        self._cache[cache_key] = result
        self._save_cache_entry(cache_key, result)
        return result

    def _build_cache_key(
        self,
        text: str,
        base_prediction: Prediction,
        current_prediction: Prediction,
        evidence: list[EvidenceItem],
    ) -> str:
        payload = {
            "mode": self.mode,
            "model_name": self.model_name,
            "text": text,
            "base_label": base_prediction.label,
            "base_confidence": round(base_prediction.confidence, 6),
            "current_label": current_prediction.label,
            "current_confidence": round(current_prediction.confidence, 6),
            "evidence": [
                {
                    "label": item.label,
                    "score": round(item.score, 6),
                    "text": item.text,
                }
                for item in evidence[:3]
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_file(self) -> Path | None:
        if not self.cache_path:
            return None
        return Path(self.cache_path)

    def _load_cache(self) -> None:
        cache_file = self._cache_file()
        if cache_file is None or not cache_file.exists():
            return
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
                key = str(row["key"])
                value = row["value"]
                self._cache[key] = SemanticAnalysis(
                    claim_type=str(value.get("claim_type", "unclear")),
                    verification_status=str(value.get("verification_status", "verification_unclear")),
                    propagation_style=str(value.get("propagation_style", "unclear")),
                    tone=str(value.get("tone", "unclear")),
                    rumor_risk=str(value.get("rumor_risk", "medium")),
                    semantic_label_bias=str(value.get("semantic_label_bias", "uncertain")),
                    override_strength=str(value.get("override_strength", "weak")),
                    retrieval_reliability=str(value.get("retrieval_reliability", "medium")),
                    trust_retrieval_more=bool(value.get("trust_retrieval_more", False)),
                    short_reason=str(value.get("short_reason", "")),
                    source=str(value.get("source", "offline_rule")),
                )
            except Exception:
                continue

    def _save_cache_entry(self, cache_key: str, result: SemanticAnalysis) -> None:
        cache_file = self._cache_file()
        if cache_file is None:
            return
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        row = {"key": cache_key, "value": result.to_dict()}
        with cache_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _analyze_offline(
        self,
        text: str,
        base_prediction: Prediction,
        current_prediction: Prediction,
        evidence: list[EvidenceItem],
    ) -> SemanticAnalysis:
        lower = text.lower()
        top_label = evidence[0].label if evidence else current_prediction.label

        if any(token in lower for token in ["breaking", "update", "latest", "reports", "reported", "reportedly"]):
            propagation_style = "headline_news"
        elif any(token in lower for token in ["live", "stream", "video", "footage", "raw video"]):
            propagation_style = "live_update"
        elif any(token in lower for token in ["i think", "i'm", "i am", "please", "clearly", "hate", "disgusting"]):
            propagation_style = "commentary"
        else:
            propagation_style = "discussion"

        if any(token in lower for token in ["statement", "news conference", "officially", "address the nation", "police storm"]):
            claim_type = "official_statement"
        elif propagation_style in {"headline_news", "live_update"} and "http" in lower:
            claim_type = "factual_update"
        elif any(token in lower for token in ["reportedly", "rumor", "unconfirmed", "is it", "?"]):
            claim_type = "unverified_claim"
        elif propagation_style == "commentary":
            claim_type = "opinion_commentary"
        else:
            claim_type = "unclear"

        if any(token in lower for token in ["unconfirmed", "reportedly", "is it", "?"]):
            verification_status = "likely_unverified"
        elif claim_type in {"official_statement", "factual_update"}:
            verification_status = "appears_verified"
        else:
            verification_status = "verification_unclear"

        if any(token in lower for token in ["disgusting", "hate", "smear", "killing", "martial law"]):
            tone = "accusatory"
        elif any(token in lower for token in ["breaking", "urgent", "storm", "hostages", "crash"]):
            tone = "alarmist"
        elif propagation_style == "commentary":
            tone = "emotional"
        else:
            tone = "neutral"

        if claim_type == "unverified_claim":
            rumor_risk = "high"
        elif claim_type == "opinion_commentary" and tone in {"accusatory", "emotional"}:
            rumor_risk = "medium"
        elif claim_type in {"official_statement", "factual_update"}:
            rumor_risk = "low"
        else:
            rumor_risk = "medium"

        if claim_type in {"official_statement", "factual_update"}:
            semantic_label_bias = "support_non_rumor"
            override_strength = "strong" if top_label == 0 else "medium"
        elif claim_type == "unverified_claim":
            semantic_label_bias = "support_rumor"
            override_strength = "strong" if top_label == 1 else "medium"
        elif claim_type == "opinion_commentary" and tone in {"accusatory", "emotional"}:
            semantic_label_bias = "support_rumor"
            override_strength = "medium"
        else:
            semantic_label_bias = "uncertain"
            override_strength = "weak"

        retrieval_reliability = "high" if evidence and top_label != current_prediction.label else "medium"
        trust_retrieval_more = bool(evidence and top_label != current_prediction.label and claim_type in {"official_statement", "factual_update", "unverified_claim"})
        short_reason = (
            "The tweet looks like a structured update or statement."
            if claim_type in {"official_statement", "factual_update"}
            else "The tweet contains unverified or strongly interpretive language."
        )

        return SemanticAnalysis(
            claim_type=claim_type,
            verification_status=verification_status,
            propagation_style=propagation_style,
            tone=tone,
            rumor_risk=rumor_risk,
            semantic_label_bias=semantic_label_bias,
            override_strength=override_strength,
            retrieval_reliability=retrieval_reliability,
            trust_retrieval_more=trust_retrieval_more,
            short_reason=short_reason,
            source="offline_rule",
        )

    def _analyze_online(
        self,
        text: str,
        base_prediction: Prediction,
        current_prediction: Prediction,
        evidence: list[EvidenceItem],
    ) -> SemanticAnalysis:
        load_local_env()
        api_base = os.getenv("SJTU_API_BASE_URL", "").rstrip("/")
        api_key = os.getenv("SJTU_API_KEY", "")
        model_name = self.model_name or os.getenv("SJTU_API_MODEL", "")
        if not api_base or not api_key or not model_name:
            raise RuntimeError("SJTU API configuration is incomplete.")

        prompt = build_semantic_prompt(text, base_prediction, current_prediction, evidence)
        response = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"].strip()
        data = parse_json_object(content)
        return SemanticAnalysis(
            claim_type=str(data.get("claim_type", "unclear")),
            verification_status=str(data.get("verification_status", "verification_unclear")),
            propagation_style=str(data.get("propagation_style", "unclear")),
            tone=str(data.get("tone", "unclear")),
            rumor_risk=str(data.get("rumor_risk", "medium")),
            semantic_label_bias=str(data.get("semantic_label_bias", "uncertain")),
            override_strength=str(data.get("override_strength", "weak")),
            retrieval_reliability=str(data.get("retrieval_reliability", "medium")),
            trust_retrieval_more=bool(data.get("trust_retrieval_more", False)),
            short_reason=str(data.get("short_reason", "")),
            source="sjtu_api",
        )
