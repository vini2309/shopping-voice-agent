from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .multilingual import canonicalize_query


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "for",
    "from",
    "have",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "please",
    "the",
    "to",
    "what",
    "where",
    "with",
    "you",
    "your",
}

TOKEN_ALIASES = {
    "electronics": {"electronics", "electronic", "device", "item", "items"},
    "opened": {"opened", "open", "used"},
    "paper": {"paper"},
    "towels": {"towel", "towels"},
    "dog": {"dog", "dogs"},
    "food": {"food"},
    "shelf": {"shelf", "stock", "topstock", "system", "inventory"},
    "empty": {"empty", "unavailable", "zero", "none"},
    "stock": {"stock", "inventory", "available"},
    "return": {"return", "refund", "exchange", "devolver"},
    "best": {"best", "top", "recommend", "recommended", "rating", "reviews", "mejor"},
    "list": {"list", "show", "inventory", "all", "options"},
}


def _tokens(value: Any) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    clean: list[str] = []
    for token in raw_tokens:
        if token in STOP_WORDS:
            continue
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        clean.append(token)
    return clean


def _expanded_tokens(value: Any) -> set[str]:
    tokens = set(_tokens(value))
    expanded = set(tokens)
    for token in tokens:
        for canonical, aliases in TOKEN_ALIASES.items():
            if token in aliases:
                expanded.add(canonical)
                expanded.update(aliases)
    return expanded


def _edit_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _wer(reference: str, transcript: str) -> float:
    left = _tokens(reference)
    right = _tokens(transcript)
    if not left:
        return 0.0 if not right else 1.0
    return _edit_distance(left, right) / len(left)


def _semantic_similarity(reference: str, transcript: str) -> float:
    left = _expanded_tokens(reference)
    right = _expanded_tokens(transcript)
    if not left:
        return 1.0 if not right else 0.0
    overlap = len(left & right) / max(1, len(left))
    union_jaccard = len(left & right) / max(1, len(left | right))
    sequence = SequenceMatcher(None, " ".join(_tokens(reference)), " ".join(_tokens(transcript))).ratio()
    wer_similarity = max(0.0, 1.0 - min(1.0, _wer(reference, transcript)))
    return round(max(overlap, (0.45 * overlap) + (0.25 * union_jaccard) + (0.20 * sequence) + (0.10 * wer_similarity)), 4)


def _infer_intent(value: str, route: str, expected: dict[str, Any]) -> str:
    tokens = _expanded_tokens(value)
    text = " ".join(_tokens(value))
    if route == "knowledge":
        if {"return", "opened", "electronics"} & tokens and "return" in tokens:
            return "return_policy"
        if {"shelf", "empty", "stock"} <= tokens or ("system" in tokens and "stock" in tokens):
            return "shelf_stock_policy"
        return "knowledge_policy"
    if expected.get("bestItemId") or expected.get("matchType") == "recommendation" or "best" in tokens:
        return "recommendation"
    if "instead" in text and {"dog", "food"} <= tokens:
        return "barge_category_lookup"
    if expected.get("matchType") == "category" or "list" in tokens or "inventory" in tokens or "show" in tokens:
        return "category_lookup"
    if any(term in tokens for term in {"where", "aisle", "find"}) or expected.get("itemIds"):
        return "location_lookup"
    return "inventory_lookup"


def _intent_score(reference_intent: str, transcript_intent: str) -> float:
    if reference_intent == transcript_intent:
        return 1.0
    equivalent_groups = [
        {"category_lookup", "barge_category_lookup", "inventory_lookup"},
        {"location_lookup", "inventory_lookup"},
        {"return_policy", "knowledge_policy"},
        {"shelf_stock_policy", "knowledge_policy"},
    ]
    for group in equivalent_groups:
        if reference_intent in group and transcript_intent in group:
            return 0.86
    return 0.0


def _required_slots(reference: str, route: str, expected: dict[str, Any], entities: list[Any]) -> list[str]:
    text = " ".join([reference, " ".join(str(entity) for entity in entities), str(expected.get("matchType") or ""), str(expected.get("bestItemId") or "")]).lower()
    sources = " ".join(str(source) for source in expected.get("sources") or []).lower()
    slots: list[str] = []
    if "paper" in text and "towel" in text:
        slots.append("paper_towels")
    if "dog" in text and "food" in text:
        slots.append("dog_food")
    if "best" in text or "recommendation" in text or expected.get("bestItemId"):
        slots.append("recommendation_intent")
    if "category" in text or "inventory" in text or expected.get("minItemCount") is not None:
        slots.append("category_intent")
    if ("opened" in text or "open" in text) and ("electronics" in text or "electronic" in text):
        slots.append("opened_electronics")
    if "returns_and_exchanges" in sources:
        slots.append("return_policy")
    if ("shelf" in text and "stock" in text) or "out_of_stock" in sources:
        slots.append("shelf_stock_policy")
    return list(dict.fromkeys(slots))


def _slot_present(slot: str, transcript: str) -> bool:
    tokens = _expanded_tokens(transcript)
    text = " ".join(_tokens(transcript))
    if slot == "paper_towels":
        return "paper" in tokens and "towel" in tokens or "paper towels" in text
    if slot == "dog_food":
        return "dog" in tokens and "food" in tokens or "dog food" in text
    if slot == "recommendation_intent":
        return bool(tokens & TOKEN_ALIASES["best"])
    if slot == "category_intent":
        return bool(tokens & TOKEN_ALIASES["list"]) or "dog_food" in text or ("dog" in tokens and "food" in tokens)
    if slot == "opened_electronics":
        return bool(tokens & TOKEN_ALIASES["opened"]) and bool(tokens & TOKEN_ALIASES["electronics"])
    if slot == "return_policy":
        return "return" in tokens and (bool(tokens & TOKEN_ALIASES["opened"]) or bool(tokens & TOKEN_ALIASES["electronics"]))
    if slot == "shelf_stock_policy":
        return ("system" in tokens or "shelf" in tokens) and "stock" in tokens and ("empty" in tokens or "inventory" in tokens)
    return False


def _slot_score(reference: str, transcript: str, route: str, expected: dict[str, Any], entities: list[Any]) -> dict[str, Any]:
    required = _required_slots(reference, route, expected, entities)
    if not required:
        return {"score": 1.0, "required": [], "preserved": [], "missing": []}
    preserved = [slot for slot in required if _slot_present(slot, transcript)]
    missing = [slot for slot in required if slot not in preserved]
    return {
        "score": round(len(preserved) / len(required), 4),
        "required": required,
        "preserved": preserved,
        "missing": missing,
    }


def score_semantic_transcript(
    *,
    reference: str,
    transcript: str,
    canonical_transcript: str | None,
    route: str,
    expected: dict[str, Any],
    entities: list[Any],
    downstream_passed: bool,
    strict_asr_passed: bool,
    strict_passed: bool,
    wer: float | None,
    entity_recall: float | None,
) -> dict[str, Any]:
    canonical = canonical_transcript or str(canonicalize_query(transcript).get("canonicalText") or transcript)
    reference_canonical = str(canonicalize_query(reference).get("canonicalText") or reference)
    reference_intent = _infer_intent(reference_canonical, route, expected)
    transcript_intent = _infer_intent(canonical, route, expected)
    intent_score = _intent_score(reference_intent, transcript_intent)
    slot = _slot_score(reference_canonical, canonical, route, expected, entities)
    canonical_score = _semantic_similarity(reference_canonical, canonical)
    downstream_score = 1.0 if downstream_passed else 0.0
    entity_score = 1.0 if entity_recall is None else max(0.0, min(1.0, float(entity_recall)))
    score = round((0.30 * intent_score) + (0.30 * float(slot["score"])) + (0.20 * canonical_score) + (0.15 * downstream_score) + (0.05 * entity_score), 4)

    if not transcript.strip():
        label = "true_asr_failure"
        passed = False
        reason = "empty transcript"
    elif strict_passed:
        label = "exact_transcript_pass"
        passed = True
        reason = "strict ASR and downstream checks passed"
    elif downstream_passed and intent_score >= 0.85 and float(slot["score"]) >= 1.0:
        label = "semantic_transcript_pass"
        passed = True
        reason = "intent and required slots are preserved despite strict WER/entity miss"
    elif downstream_passed and intent_score >= 0.85 and float(slot["score"]) >= 0.75:
        label = "task_recovered_asr_miss"
        passed = True
        reason = "downstream task recovered from a partial ASR metric miss"
    elif intent_score <= 0.25 and float(slot["score"]) <= 0.25 and canonical_score < 0.45:
        label = "wrong_prompt"
        passed = False
        reason = "transcript intent and slots do not match the reference prompt"
    elif not downstream_passed and float(slot["score"]) >= 0.75 and intent_score >= 0.85:
        label = "downstream_failure_after_semantic_asr"
        passed = False
        reason = "transcript preserved the request, but downstream tool/RAG checks failed"
    else:
        label = "true_asr_failure"
        passed = False
        reason = "semantic transcript evidence is below the task-preservation threshold"

    return {
        "passed": passed,
        "label": label,
        "score": score,
        "intentScore": round(intent_score, 4),
        "slotScore": slot["score"],
        "canonicalScore": canonical_score,
        "downstreamScore": downstream_score,
        "entityScore": round(entity_score, 4),
        "referenceIntent": reference_intent,
        "transcriptIntent": transcript_intent,
        "requiredSlots": slot["required"],
        "preservedSlots": slot["preserved"],
        "missingSlots": slot["missing"],
        "canonicalReference": reference_canonical,
        "canonicalTranscript": canonical,
        "strictAsrPassed": strict_asr_passed,
        "strictPassed": strict_passed,
        "literalWer": wer,
        "entityRecall": entity_recall,
        "reason": reason,
        "method": "deterministic_intent_slot_canonical_equivalence",
        "thresholds": {
            "semanticPassIntent": 0.85,
            "semanticPassSlots": 1.0,
            "taskRecoverySlots": 0.75,
        },
    }
