from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.inventory import tool_payload  # noqa: E402
from backend.app.knowledge import search_knowledge  # noqa: E402


@dataclass
class TaskResult:
    task_id: str
    expected_action: str
    predicted_action: str
    action_ok: bool
    item_ok: bool
    aisle_ok: bool
    source_ok: bool
    source_top1_ok: bool
    success: bool
    error_type: str | None
    retrieval_confidence: float | None
    retrieval_margin: float | None
    source_precision_at_k: float | None
    source_recall_at_k: float | None
    source_mrr: float | None
    source_ndcg_at_k: float | None
    dataset: str | None
    condition: dict[str, Any]
    query: str | None
    tool_name: str | None
    predicted_item_ids: list[str]
    expected_item_ids: list[str]
    predicted_aisles: list[str]
    expected_aisles: list[str]
    predicted_source_ids: list[str]
    expected_source_ids: list[str]
    predicted_tool_names: list[str]
    expected_tool_names: list[str]
    payload: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return tasks


def latest_user_text(task: dict[str, Any]) -> str:
    for turn in reversed(task["turns"]):
        if turn["role"] == "user":
            return turn["text"]
    raise ValueError(f"{task['task_id']} has no user turn")


def expected_query(task: dict[str, Any]) -> str:
    expected = task["expected"]
    return str(expected.get("query") or latest_user_text(task)).strip()


def expected_tool_calls(expected: dict[str, Any]) -> list[dict[str, Any]]:
    calls = expected.get("tool_calls")
    if isinstance(calls, list) and calls:
        return [call for call in calls if isinstance(call, dict)]
    tool_name = expected.get("tool_name")
    if tool_name:
        return [
            {
                "tool_name": tool_name,
                "query": expected.get("query"),
                "item_ids": expected.get("item_ids", []),
                "source_ids": expected.get("source_ids", []),
                "aisles": expected.get("aisles", []),
            }
        ]
    return []


def expected_values(expected: dict[str, Any], field: str) -> list[str]:
    values = [str(value) for value in expected.get(field, [])]
    for call in expected_tool_calls(expected):
        values.extend(str(value) for value in call.get(field, []))

    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def expected_tool_names(expected: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for call in expected_tool_calls(expected):
        name = call.get("tool_name")
        if isinstance(name, str) and name:
            names.append(name)
    return list(dict.fromkeys(names))


def retrieval_meta_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    retrieval = payload.get("retrieval")
    if isinstance(retrieval, dict):
        return retrieval
    calls = payload.get("calls")
    if not isinstance(calls, list):
        return {}
    for call in calls:
        call_payload = call.get("payload") if isinstance(call, dict) else None
        if isinstance(call_payload, dict) and isinstance(call_payload.get("retrieval"), dict):
            return call_payload["retrieval"]
    return {}


def dcg(relevances: list[int]) -> float:
    return sum(relevance / math.log2(index + 2) for index, relevance in enumerate(relevances))


def source_ranking_metrics(predicted: list[str], expected: list[str]) -> dict[str, float | None]:
    if not expected:
        return {
            "source_precision_at_k": None,
            "source_recall_at_k": None,
            "source_mrr": None,
            "source_ndcg_at_k": None,
        }

    expected_set = set(expected)
    predicted_set = set(predicted)
    true_positive_count = len(predicted_set & expected_set)
    precision = true_positive_count / len(predicted) if predicted else 0.0
    recall = true_positive_count / len(expected_set)

    mrr = 0.0
    for rank, source_id in enumerate(predicted, start=1):
        if source_id in expected_set:
            mrr = 1.0 / rank
            break

    relevances = [1 if source_id in expected_set else 0 for source_id in predicted]
    ideal_relevances = [1] * min(len(expected_set), len(predicted))
    ideal_value = dcg(ideal_relevances)
    ndcg = dcg(relevances) / ideal_value if ideal_value else 0.0

    return {
        "source_precision_at_k": precision,
        "source_recall_at_k": recall,
        "source_mrr": mrr,
        "source_ndcg_at_k": ndcg,
    }


def classify_error(
    *,
    success: bool,
    expected_action: str,
    tool_name: str | None,
    action_ok: bool,
    item_ok: bool,
    aisle_ok: bool,
    source_ok: bool,
) -> str | None:
    if success:
        return None
    if not action_ok:
        return "MULTI_TOOL_ROUTING_ERROR" if expected_action == "multi_tool" else "TOOL_DECISION_ERROR"
    if expected_action == "cannot_answer":
        return "UNSUPPORTED_ANSWER_ERROR"
    if expected_action == "multi_tool" and not item_ok:
        return "TOOL_RESULT_ERROR"
    if expected_action == "multi_tool" and not aisle_ok:
        return "SLOT_ERROR"
    if expected_action == "multi_tool" and not source_ok:
        return "RAG_RETRIEVAL_ERROR"
    if tool_name == "search_knowledge" and not source_ok:
        return "RAG_RETRIEVAL_ERROR"
    if tool_name == "lookup_inventory" and not item_ok:
        return "TOOL_RESULT_ERROR"
    if tool_name == "lookup_inventory" and not aisle_ok:
        return "SLOT_ERROR"
    return "UNKNOWN_ERROR"


def score_task(task: dict[str, Any]) -> TaskResult:
    expected = task["expected"]
    expected_action = expected["action"]

    payload: dict[str, Any] = {}
    predicted_action = "direct"
    predicted_item_ids: list[str] = []
    predicted_aisles: list[str] = []
    predicted_source_ids: list[str] = []
    predicted_tool_names: list[str] = []
    query = expected.get("query")
    tool_name = expected.get("tool_name")

    if expected_action == "multi_tool":
        call_results: list[dict[str, Any]] = []
        for call in expected_tool_calls(expected):
            call_tool_name = str(call.get("tool_name") or "")
            call_query = str(call.get("query") or latest_user_text(task)).strip()
            if call_tool_name == "search_knowledge":
                result = search_knowledge(call_query)
                predicted_source_ids.extend(str(source) for source in result.get("sources", []))
            elif call_tool_name == "lookup_inventory":
                result = tool_payload(call_query)
                matches = result.get("matches") or ([result["item"]] if result.get("item") else [])
                predicted_item_ids.extend(str(item["sku"]) for item in matches if item.get("sku"))
                predicted_aisles.extend(str(item["aisle"]) for item in matches if item.get("aisle"))
            else:
                result = {"tool": call_tool_name, "query": call_query, "found": False, "error": "unsupported expected tool"}

            if result.get("found"):
                predicted_tool_names.append(call_tool_name)
            call_results.append({"tool": call_tool_name, "query": call_query, "payload": result})

        expected_name_set = set(expected_tool_names(expected))
        predicted_name_set = set(predicted_tool_names)
        if expected_name_set and expected_name_set.issubset(predicted_name_set):
            predicted_action = "multi_tool"
        elif predicted_name_set:
            predicted_action = "tool_call"
        else:
            predicted_action = "cannot_answer"
        tool_name = "multi_tool"
        predicted_aisles = sorted(set(predicted_aisles))
        payload = {
            "tool": "multi_tool",
            "query": query,
            "found": predicted_action == "multi_tool",
            "calls": call_results,
        }
    elif expected_action in {"tool_call", "cannot_answer"}:
        query = expected_query(task)
        if tool_name == "search_knowledge":
            payload = search_knowledge(query)
            predicted_source_ids = [str(source) for source in payload.get("sources", [])]
        else:
            payload = tool_payload(query)
            matches = payload.get("matches") or ([payload["item"]] if payload.get("item") else [])
            predicted_item_ids = [str(item["sku"]) for item in matches if item.get("sku")]
            predicted_aisles = sorted({str(item["aisle"]) for item in matches if item.get("aisle")})
        predicted_action = "tool_call" if payload.get("found") else "cannot_answer"
        if payload.get("found") and isinstance(tool_name, str):
            predicted_tool_names = [tool_name]
    elif expected_action == "request_info":
        predicted_action = "request_info"
    else:
        predicted_action = "direct"

    expected_item_ids = expected_values(expected, "item_ids")
    expected_aisles = expected_values(expected, "aisles")
    expected_source_ids = expected_values(expected, "source_ids")
    expected_tool_name_list = expected_tool_names(expected)

    action_ok = predicted_action == expected_action
    if expected_action == "multi_tool":
        action_ok = action_ok and set(expected_tool_name_list).issubset(set(predicted_tool_names))
    item_ok = set(expected_item_ids).issubset(set(predicted_item_ids))
    aisle_ok = set(expected_aisles).issubset(set(predicted_aisles))
    source_ok = set(expected_source_ids).issubset(set(predicted_source_ids))
    source_top1_ok = not expected_source_ids or (
        bool(predicted_source_ids) and predicted_source_ids[0] in set(expected_source_ids)
    )

    if expected_action == "cannot_answer":
        item_ok = not predicted_item_ids
        aisle_ok = not predicted_aisles
        source_ok = not predicted_source_ids
        source_top1_ok = not predicted_source_ids
    if expected_action in {"direct", "request_info"}:
        item_ok = True
        aisle_ok = True
        source_ok = True
        source_top1_ok = True
    if not expected_source_ids:
        source_ok = True

    success = action_ok and item_ok and aisle_ok and source_ok
    retrieval = retrieval_meta_from_payload(payload) if isinstance(payload, dict) else {}
    retrieval_confidence = retrieval.get("confidence")
    retrieval_margin = retrieval.get("margin")
    ranking_metrics = source_ranking_metrics(predicted_source_ids, expected_source_ids)
    source = task.get("source") or {}
    condition = task.get("condition") or {}
    error_type = classify_error(
        success=success,
        expected_action=expected_action,
        tool_name=tool_name,
        action_ok=action_ok,
        item_ok=item_ok,
        aisle_ok=aisle_ok,
        source_ok=source_ok,
    )

    return TaskResult(
        task_id=task["task_id"],
        expected_action=expected_action,
        predicted_action=predicted_action,
        action_ok=action_ok,
        item_ok=item_ok,
        aisle_ok=aisle_ok,
        source_ok=source_ok,
        source_top1_ok=source_top1_ok,
        success=success,
        error_type=error_type,
        retrieval_confidence=retrieval_confidence,
        retrieval_margin=retrieval_margin,
        source_precision_at_k=ranking_metrics["source_precision_at_k"],
        source_recall_at_k=ranking_metrics["source_recall_at_k"],
        source_mrr=ranking_metrics["source_mrr"],
        source_ndcg_at_k=ranking_metrics["source_ndcg_at_k"],
        dataset=source.get("dataset"),
        condition=condition,
        query=query,
        tool_name=tool_name,
        predicted_item_ids=predicted_item_ids,
        expected_item_ids=expected_item_ids,
        predicted_aisles=predicted_aisles,
        expected_aisles=expected_aisles,
        predicted_source_ids=predicted_source_ids,
        expected_source_ids=expected_source_ids,
        predicted_tool_names=predicted_tool_names,
        expected_tool_names=expected_tool_name_list,
        payload=payload,
    )


def summarize(results: list[TaskResult]) -> dict[str, Any]:
    total = len(results)
    successes = sum(result.success for result in results)
    action_ok = sum(result.action_ok for result in results)
    item_ok = sum(result.item_ok for result in results)
    aisle_ok = sum(result.aisle_ok for result in results)
    source_ok = sum(result.source_ok for result in results)
    rag_results = [result for result in results if result.tool_name == "search_knowledge" or result.expected_source_ids]
    multi_tool_results = [result for result in results if result.expected_action == "multi_tool"]
    retrieval_confidences = [
        result.retrieval_confidence for result in rag_results if result.retrieval_confidence is not None
    ]
    retrieval_margins = [result.retrieval_margin for result in rag_results if result.retrieval_margin is not None]
    source_precisions = [
        result.source_precision_at_k for result in rag_results if result.source_precision_at_k is not None
    ]
    source_recalls = [result.source_recall_at_k for result in rag_results if result.source_recall_at_k is not None]
    source_mrrs = [result.source_mrr for result in rag_results if result.source_mrr is not None]
    source_ndcgs = [result.source_ndcg_at_k for result in rag_results if result.source_ndcg_at_k is not None]
    return {
        "total": total,
        "success": successes,
        "success_rate": successes / total if total else 0,
        "action_accuracy": action_ok / total if total else 0,
        "item_accuracy": item_ok / total if total else 0,
        "aisle_accuracy": aisle_ok / total if total else 0,
        "source_accuracy": source_ok / total if total else 0,
        "rag_total": len(rag_results),
        "multi_tool_total": len(multi_tool_results),
        "multi_tool_success_rate": (
            sum(result.success for result in multi_tool_results) / len(multi_tool_results)
            if multi_tool_results
            else 0
        ),
        "multi_tool_action_accuracy": (
            sum(result.action_ok for result in multi_tool_results) / len(multi_tool_results)
            if multi_tool_results
            else 0
        ),
        "multi_tool_item_accuracy": (
            sum(result.item_ok for result in multi_tool_results) / len(multi_tool_results)
            if multi_tool_results
            else 0
        ),
        "multi_tool_source_accuracy": (
            sum(result.source_ok for result in multi_tool_results) / len(multi_tool_results)
            if multi_tool_results
            else 0
        ),
        "rag_source_top1_accuracy": (
            sum(result.source_top1_ok for result in rag_results) / len(rag_results) if rag_results else 0
        ),
        "avg_retrieval_confidence": (
            sum(retrieval_confidences) / len(retrieval_confidences) if retrieval_confidences else None
        ),
        "avg_retrieval_margin": sum(retrieval_margins) / len(retrieval_margins) if retrieval_margins else None,
        "rag_source_precision_at_k": sum(source_precisions) / len(source_precisions) if source_precisions else None,
        "rag_source_recall_at_k": sum(source_recalls) / len(source_recalls) if source_recalls else None,
        "rag_source_mrr": sum(source_mrrs) / len(source_mrrs) if source_mrrs else None,
        "rag_source_ndcg_at_k": sum(source_ndcgs) / len(source_ndcgs) if source_ndcgs else None,
    }


def condition_bucket(result: TaskResult) -> str:
    condition = result.condition or {}
    return "|".join(
        [
            f"accent={condition.get('accent', 'unknown')}",
            f"noise={condition.get('noise', 'unknown')}",
            f"snr={condition.get('snr_db')}",
            f"barge_in={condition.get('barge_in', False)}",
        ]
    )


def summarize_groups(results: list[TaskResult], key_fn) -> dict[str, Any]:
    groups: dict[str, list[TaskResult]] = {}
    for result in results:
        key = str(key_fn(result))
        groups.setdefault(key, []).append(result)
    return {key: summarize(group_results) for key, group_results in sorted(groups.items())}


def confusion_matrix(results: list[TaskResult]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for result in results:
        expected = result.expected_action
        predicted = result.predicted_action
        matrix.setdefault(expected, {})
        matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1
    return matrix


def error_taxonomy(results: list[TaskResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        label = result.error_type or "SUCCESS"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VoiceRetailBench text tasks against inventory and knowledge tools.")
    parser.add_argument("--tasks", type=Path, default=Path(__file__).with_name("seed_tasks.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=1.0,
        help="Exit successfully when success_rate is at least this value. Use 0 for benchmark reporting.",
    )
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    results = [score_task(task) for task in tasks]
    summary = summarize(results)

    output = {
        "summary": summary,
        "by_dataset": summarize_groups(results, lambda result: result.dataset or "unknown"),
        "by_condition": summarize_groups(results, condition_bucket),
        "by_tool": summarize_groups(results, lambda result: result.tool_name or "none"),
        "confusion_matrix": confusion_matrix(results),
        "error_taxonomy": error_taxonomy(results),
        "results": [asdict(result) for result in results],
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)

    print(json.dumps(output, indent=2))
    return 0 if summary["success_rate"] >= args.min_success_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
