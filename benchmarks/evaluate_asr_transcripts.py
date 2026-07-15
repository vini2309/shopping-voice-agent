from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.inventory import load_inventory  # noqa: E402
from benchmarks.evaluate_text_tasks import load_jsonl, score_task  # noqa: E402


DEFAULT_TASKS = Path(__file__).with_name("generated") / "voice_retail_paper_tasks.jsonl"
DEFAULT_OUT = ROOT / "artifacts" / "asr_transcript_eval.json"

STOP_WORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "but",
    "can",
    "check",
    "could",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "should",
    "that",
    "the",
    "their",
    "there",
    "to",
    "what",
    "when",
    "where",
    "why",
    "with",
    "you",
    "your",
}


@dataclass
class TranscriptResult:
    task_id: str
    dataset: str | None
    condition: dict[str, Any]
    reference_text: str
    transcript_text: str
    wer: float
    entity_wer: float | None
    entity_recall: float | None
    reference_tokens: int
    transcript_tokens: int
    reference_entity_tokens: list[str]
    transcript_entity_tokens: list[str]
    task_success: bool
    task_error_type: str | None
    expected_action: str
    predicted_action: str
    tool_name: str | None


def normalize_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def entity_token_candidates(value: str) -> list[str]:
    tokens = normalize_tokens(value)
    return [token for token in tokens if token not in STOP_WORDS]


def edit_distance(left: list[str], right: list[str]) -> int:
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


def word_error_rate(reference: str, transcript: str) -> tuple[float, list[str], list[str]]:
    reference_tokens = normalize_tokens(reference)
    transcript_tokens = normalize_tokens(transcript)
    if not reference_tokens:
        return (0.0 if not transcript_tokens else 1.0, reference_tokens, transcript_tokens)
    return edit_distance(reference_tokens, transcript_tokens) / len(reference_tokens), reference_tokens, transcript_tokens


def inventory_terms_by_sku() -> dict[str, set[str]]:
    terms: dict[str, set[str]] = {}
    for item in load_inventory():
        item_terms: set[str] = set()
        for value in [item.get("name", ""), item.get("department", ""), *item.get("synonyms", [])]:
            item_terms.update(entity_token_candidates(str(value)))
        terms[str(item["sku"])] = item_terms
    return terms


def expected_calls(expected: dict[str, Any]) -> list[dict[str, Any]]:
    calls = expected.get("tool_calls")
    if isinstance(calls, list) and calls:
        return [call for call in calls if isinstance(call, dict)]
    if expected.get("tool_name"):
        return [
            {
                "tool_name": expected.get("tool_name"),
                "query": expected.get("query"),
                "item_ids": expected.get("item_ids", []),
                "source_ids": expected.get("source_ids", []),
                "aisles": expected.get("aisles", []),
            }
        ]
    return []


def critical_entity_terms(task: dict[str, Any], sku_terms: dict[str, set[str]]) -> set[str]:
    expected = task.get("expected") or {}
    terms: set[str] = set()
    for value in [expected.get("query"), *expected.get("aisles", []), *expected.get("source_ids", [])]:
        if value:
            terms.update(entity_token_candidates(str(value).replace("#", " ")))

    for call in expected_calls(expected):
        for value in [call.get("query"), *call.get("aisles", []), *call.get("source_ids", [])]:
            if value:
                terms.update(entity_token_candidates(str(value).replace("#", " ")))
        for sku in call.get("item_ids", []):
            terms.update(sku_terms.get(str(sku), set()))

    for sku in expected.get("item_ids", []):
        terms.update(sku_terms.get(str(sku), set()))

    return terms


def entity_metrics(reference_tokens: list[str], transcript_tokens: list[str], critical_terms: set[str]) -> dict[str, Any]:
    reference_entities = [token for token in reference_tokens if token in critical_terms]
    transcript_entities = [token for token in transcript_tokens if token in critical_terms]
    if not reference_entities:
        return {
            "entity_wer": None,
            "entity_recall": None,
            "reference_entity_tokens": [],
            "transcript_entity_tokens": transcript_entities,
        }

    entity_wer = edit_distance(reference_entities, transcript_entities) / len(reference_entities)
    remaining = transcript_entities.copy()
    matches = 0
    for token in reference_entities:
        if token in remaining:
            matches += 1
            remaining.remove(token)

    return {
        "entity_wer": entity_wer,
        "entity_recall": matches / len(reference_entities),
        "reference_entity_tokens": reference_entities,
        "transcript_entity_tokens": transcript_entities,
    }


def transcript_pairs(task: dict[str, Any], *, include_clean: bool) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    condition = task.get("condition") or {}
    for turn in task.get("turns", []):
        if turn.get("role") != "user":
            continue

        reference = turn.get("reference_text")
        variants = turn.get("transcript_variants")
        if isinstance(reference, str) and isinstance(variants, list) and variants:
            for variant in variants:
                if isinstance(variant, dict) and isinstance(variant.get("text"), str):
                    pairs.append(
                        {
                            "reference_text": reference,
                            "transcript_text": variant["text"],
                            "condition": variant.get("condition") or condition,
                            "variant": variant,
                        }
                    )
            continue

        if include_clean:
            text = str(turn.get("text") or "")
            pairs.append(
                {
                    "reference_text": str(reference or text),
                    "transcript_text": text,
                    "condition": condition,
                    "variant": {"id": "clean_or_existing_transcript", "provider": "task-text"},
                }
            )
    return pairs


def condition_key(condition: dict[str, Any]) -> str:
    return "|".join(
        [
            f"accent={condition.get('accent', 'unknown')}",
            f"noise={condition.get('noise', 'unknown')}",
            f"snr={condition.get('snr_db')}",
            f"barge_in={condition.get('barge_in', False)}",
        ]
    )


def average(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def summarize(results: list[TranscriptResult]) -> dict[str, Any]:
    if not results:
        return {
            "transcriptPairs": 0,
            "averageWer": None,
            "averageEntityWer": None,
            "averageEntityRecall": None,
            "taskSuccessRate": None,
        }

    return {
        "transcriptPairs": len(results),
        "uniqueTasks": len({result.task_id for result in results}),
        "averageWer": average([result.wer for result in results]),
        "averageEntityWer": average([result.entity_wer for result in results]),
        "averageEntityRecall": average([result.entity_recall for result in results]),
        "taskSuccessRate": sum(result.task_success for result in results) / len(results),
        "maxWer": max(result.wer for result in results),
        "maxEntityWer": max((result.entity_wer for result in results if result.entity_wer is not None), default=None),
    }


def summarize_groups(results: list[TranscriptResult], key_fn) -> dict[str, Any]:
    groups: dict[str, list[TranscriptResult]] = {}
    for result in results:
        groups.setdefault(str(key_fn(result)), []).append(result)
    return {key: summarize(group) for key, group in sorted(groups.items())}


def evaluate(tasks: list[dict[str, Any]], *, include_clean: bool) -> dict[str, Any]:
    sku_terms = inventory_terms_by_sku()
    results: list[TranscriptResult] = []
    scored_tasks: dict[str, Any] = {}

    for task in tasks:
        pairs = transcript_pairs(task, include_clean=include_clean)
        if not pairs:
            continue

        task_result = scored_tasks.setdefault(task["task_id"], score_task(task))
        critical_terms = critical_entity_terms(task, sku_terms)
        source = task.get("source") or {}
        expected = task.get("expected") or {}
        for pair in pairs:
            wer, reference_tokens, transcript_tokens = word_error_rate(pair["reference_text"], pair["transcript_text"])
            entity = entity_metrics(reference_tokens, transcript_tokens, critical_terms)
            results.append(
                TranscriptResult(
                    task_id=task["task_id"],
                    dataset=source.get("dataset"),
                    condition=pair["condition"],
                    reference_text=pair["reference_text"],
                    transcript_text=pair["transcript_text"],
                    wer=wer,
                    entity_wer=entity["entity_wer"],
                    entity_recall=entity["entity_recall"],
                    reference_tokens=len(reference_tokens),
                    transcript_tokens=len(transcript_tokens),
                    reference_entity_tokens=entity["reference_entity_tokens"],
                    transcript_entity_tokens=entity["transcript_entity_tokens"],
                    task_success=bool(task_result.success),
                    task_error_type=task_result.error_type,
                    expected_action=str(expected.get("action") or ""),
                    predicted_action=task_result.predicted_action,
                    tool_name=task_result.tool_name,
                )
            )

    return {
        "summary": summarize(results),
        "by_dataset": summarize_groups(results, lambda result: result.dataset or "unknown"),
        "by_condition": summarize_groups(results, lambda result: condition_key(result.condition)),
        "results": [asdict(result) for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ASR transcript robustness for VoiceRetailBench tasks.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="Also include tasks without explicit transcript variants as zero-error transcript pairs.",
    )
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    output = evaluate(tasks, include_clean=args.include_clean)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
