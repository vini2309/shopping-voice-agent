from __future__ import annotations

import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inventory import tool_payload
from .knowledge import generate_evidence_gated_answer
from .pricing import estimate_costs

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CASES_PATH = ROOT_DIR / "benchmarks" / "eval_cases.json"
BENCHMARK_ARTIFACT_DIR = ROOT_DIR / "artifacts" / "benchmarks"
LATEST_JSON_PATH = BENCHMARK_ARTIFACT_DIR / "latest.json"
LATEST_CSV_PATH = BENCHMARK_ARTIFACT_DIR / "latest.csv"

ABSTAIN_TERMS = (
    "do not see",
    "does not cover",
    "not cover",
    "not enough supported",
    "do not have enough supported",
    "not enough",
    "cannot",
    "try another",
    "clarifying",
    "current shopping catalog",
    "knowledge base",
)


def load_benchmark_cases() -> list[dict[str, Any]]:
    with BENCHMARK_CASES_PATH.open("r", encoding="utf-8-sig") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("benchmark case file must contain a JSON list")
    return [case for case in cases if isinstance(case, dict)]


def load_latest_benchmark() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No benchmark run saved yet."}
    with LATEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").split())


def _contains(text: str, term: str) -> bool:
    return _normalize(term) in _normalize(text)


def _observed_item_ids(payload: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    item = payload.get("item") if isinstance(payload.get("item"), dict) else None
    if item and item.get("sku"):
        ids.append(str(item["sku"]))
    for row in payload.get("matches") or []:
        if isinstance(row, dict) and row.get("sku"):
            ids.append(str(row["sku"]))
    best = payload.get("bestOption") if isinstance(payload.get("bestOption"), dict) else None
    best_item = best.get("item") if isinstance(best and best.get("item"), dict) else None
    if best_item and best_item.get("sku"):
        ids.append(str(best_item["sku"]))
    return list(dict.fromkeys(ids))


def _observed_aisles(payload: dict[str, Any]) -> list[str]:
    aisles = [str(value) for value in payload.get("aisles") or [] if value]
    item = payload.get("item") if isinstance(payload.get("item"), dict) else None
    if item and item.get("aisle"):
        aisles.append(str(item["aisle"]))
    for row in payload.get("matches") or []:
        if isinstance(row, dict) and row.get("aisle"):
            aisles.append(str(row["aisle"]))
    return sorted(set(aisles))


def _inventory_required_slot_ok(slot: str, payload: dict[str, Any], answer: str) -> bool:
    item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
    best = payload.get("bestOption") if isinstance(payload.get("bestOption"), dict) else {}
    best_item = best.get("item") if isinstance(best.get("item"), dict) else {}
    if slot == "item":
        return bool(payload.get("item") or payload.get("matches"))
    if slot == "aisle":
        return bool(_observed_aisles(payload)) or "aisle" in _normalize(answer)
    if slot == "bay":
        return bool(item.get("bay") or best_item.get("bay")) or "bay" in _normalize(answer)
    if slot == "stock":
        return "available" in _normalize(answer) or "unit" in _normalize(answer) or item.get("stock") is not None
    if slot == "rating":
        reviews = best_item.get("customerReviewSummary") if isinstance(best_item.get("customerReviewSummary"), dict) else {}
        return reviews.get("rating") is not None or "star" in _normalize(answer)
    if slot == "reviews":
        reviews = best_item.get("customerReviewSummary") if isinstance(best_item.get("customerReviewSummary"), dict) else {}
        return reviews.get("reviewCount") is not None or "review" in _normalize(answer)
    if slot == "cannot_answer":
        return not payload.get("found") and any(term in _normalize(answer) for term in ABSTAIN_TERMS)
    return True


def _check_inventory(payload: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    failures: list[str] = []
    answer = str(payload.get("speechAnswer") or payload.get("message") or "")
    found = bool(payload.get("found"))
    observed_ids = _observed_item_ids(payload)
    observed_aisles = _observed_aisles(payload)
    observed = {
        "found": found,
        "matchType": payload.get("matchType"),
        "itemIds": observed_ids,
        "aisles": observed_aisles,
        "itemCount": payload.get("itemCount") or len(payload.get("matches") or []),
        "answer": answer,
    }

    if "found" in expected and found is not bool(expected["found"]):
        failures.append(f"expected found={expected['found']} observed {found}")
    if expected.get("matchType") and payload.get("matchType") != expected["matchType"]:
        failures.append(f"expected matchType {expected['matchType']} observed {payload.get('matchType')}")
    if expected.get("itemIds"):
        missing = [item_id for item_id in expected["itemIds"] if item_id not in observed_ids]
        if missing:
            failures.append(f"missing item ids {missing}")
    if expected.get("aisles"):
        missing = [aisle for aisle in expected["aisles"] if aisle not in observed_aisles]
        if missing:
            failures.append(f"missing aisles {missing}")
    if expected.get("minItemCount") is not None and int(observed["itemCount"] or 0) < int(expected["minItemCount"]):
        failures.append(f"item count below {expected['minItemCount']}")
    if expected.get("minComplements") is not None and len(payload.get("complements") or []) < int(expected["minComplements"]):
        failures.append(f"complements below {expected['minComplements']}")

    best = payload.get("bestOption") if isinstance(payload.get("bestOption"), dict) else {}
    best_item = best.get("item") if isinstance(best.get("item"), dict) else {}
    if expected.get("bestItemId"):
        observed["bestItemId"] = best_item.get("sku")
        if best_item.get("sku") != expected["bestItemId"]:
            failures.append(f"expected best item {expected['bestItemId']} observed {best_item.get('sku')}")
    if expected.get("minRating") is not None:
        reviews = best_item.get("customerReviewSummary") if isinstance(best_item.get("customerReviewSummary"), dict) else {}
        observed_rating = float(reviews.get("rating") or 0)
        observed["rating"] = observed_rating
        if observed_rating < float(expected["minRating"]):
            failures.append(f"rating below {expected['minRating']}")

    for slot in expected.get("requiredSlots") or []:
        if not _inventory_required_slot_ok(str(slot), payload, answer):
            failures.append(f"missing required slot {slot}")
    for term in expected.get("forbiddenTerms") or []:
        if _contains(answer, str(term)):
            failures.append(f"forbidden term present: {term}")

    return not failures, failures, observed


def _check_knowledge(payload: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    failures: list[str] = []
    gate = payload.get("answerGate") if isinstance(payload.get("answerGate"), dict) else {}
    answer = str(payload.get("speechAnswer") or payload.get("answerGeneration", {}).get("finalAnswer") or "")
    sources = [str(value) for value in payload.get("sources") or []]
    observed = {
        "found": bool(payload.get("found")),
        "answerable": bool(payload.get("answerable")),
        "gateStatus": gate.get("status"),
        "gateAction": gate.get("action"),
        "faithfulnessScore": gate.get("faithfulnessScore") or payload.get("faithfulness", {}).get("faithfulnessScore"),
        "sources": sources,
        "answer": answer,
    }

    if "answerable" in expected and observed["answerable"] is not bool(expected["answerable"]):
        failures.append(f"expected answerable={expected['answerable']} observed {observed['answerable']}")
    if expected.get("gateStatus") and gate.get("status") != expected["gateStatus"]:
        failures.append(f"expected gate {expected['gateStatus']} observed {gate.get('status')}")
    if expected.get("sources"):
        missing = [source for source in expected["sources"] if source not in sources]
        if missing:
            failures.append(f"missing sources {missing}")
    for term in expected.get("requiredTerms") or []:
        if not _contains(answer, str(term)):
            failures.append(f"missing required answer term: {term}")
    for term in expected.get("forbiddenTerms") or []:
        if _contains(answer, str(term)):
            failures.append(f"forbidden term present: {term}")
    if expected.get("answerable") is False and not any(term in _normalize(answer) for term in ABSTAIN_TERMS):
        failures.append("blocked answer did not use abstention language")

    return not failures, failures, observed


def _case_cost(query: str, answer: str, latency_ms: float) -> dict[str, Any]:
    estimated_voice_latency_ms = int(round(latency_ms + 650 + min(1200, len(answer) * 7)))
    call_ms = max(1400, estimated_voice_latency_ms + 450)
    costs = estimate_costs(
        call_ms=call_ms,
        user_text=query,
        answer=answer,
        total_latency_ms=estimated_voice_latency_ms,
    )
    rows = {row["id"]: row for row in costs.get("rows", [])}
    return {
        "estimatedVoiceLatencyMs": estimated_voice_latency_ms,
        "callMs": call_ms,
        "vapiStackCost": rows.get("vapi-stack", {}).get("cost", 0.0),
        "vapiStackPer1000": rows.get("vapi-stack", {}).get("per1000", 0.0),
        "openaiRealtimeCost": rows.get("openai", {}).get("cost", 0.0),
        "geminiLiveCost": rows.get("gemini", {}).get("cost", 0.0),
    }


def _run_inventory_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    payload = tool_payload(str(case.get("query") or ""))
    latency_ms = (time.perf_counter() - started) * 1000
    passed, failures, observed = _check_inventory(payload, case.get("expected") or {})
    answer = str(payload.get("speechAnswer") or payload.get("message") or "")
    return {
        "payload": payload,
        "observed": observed,
        "passed": passed,
        "failures": failures,
        "latencyMs": round(latency_ms, 2),
        "answer": answer,
        "cost": _case_cost(str(case.get("query") or ""), answer, latency_ms),
    }


def _run_knowledge_case(case: dict[str, Any], *, query_key: str = "query", expected_key: str = "expected") -> dict[str, Any]:
    query = str(case.get(query_key) or case.get("query") or "")
    started = time.perf_counter()
    payload = generate_evidence_gated_answer(query, limit=4)
    latency_ms = (time.perf_counter() - started) * 1000
    expected = case.get(expected_key) if isinstance(case.get(expected_key), dict) else case.get("expected") or {}
    passed, failures, observed = _check_knowledge(payload, expected)
    answer = str(payload.get("speechAnswer") or "")
    return {
        "payload": payload,
        "observed": observed,
        "passed": passed,
        "failures": failures,
        "latencyMs": round(latency_ms, 2),
        "answer": answer,
        "cost": _case_cost(query, answer, latency_ms),
    }


def _run_multi_tool_case(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected") or {}
    inventory_case = {
        "query": case.get("inventoryQuery") or case.get("query"),
        "expected": expected.get("inventory") or {},
    }
    knowledge_case = {
        "query": case.get("knowledgeQuery") or case.get("query"),
        "expected": expected.get("knowledge") or {},
    }
    started = time.perf_counter()
    inventory_result = _run_inventory_case(inventory_case)
    knowledge_result = _run_knowledge_case(knowledge_case)
    latency_ms = (time.perf_counter() - started) * 1000
    failures = [f"inventory: {failure}" for failure in inventory_result["failures"]]
    failures.extend(f"knowledge: {failure}" for failure in knowledge_result["failures"])
    answer = " ".join(part for part in [inventory_result["answer"], knowledge_result["answer"]] if part)
    observed = {
        "inventory": inventory_result["observed"],
        "knowledge": knowledge_result["observed"],
    }
    return {
        "payload": {
            "inventory": inventory_result["payload"],
            "knowledge": knowledge_result["payload"],
        },
        "observed": observed,
        "passed": not failures,
        "failures": failures,
        "latencyMs": round(latency_ms, 2),
        "answer": answer,
        "cost": _case_cost(str(case.get("query") or ""), answer, latency_ms),
    }


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    case_type = str(case.get("type") or "inventory")
    if case_type == "inventory":
        result = _run_inventory_case(case)
    elif case_type == "knowledge":
        result = _run_knowledge_case(case)
    elif case_type == "multi_tool":
        result = _run_multi_tool_case(case)
    else:
        result = {
            "payload": {},
            "observed": {},
            "passed": False,
            "failures": [f"unsupported case type {case_type}"],
            "latencyMs": 0.0,
            "answer": "",
            "cost": _case_cost(str(case.get("query") or ""), "", 0.0),
        }

    return {
        "id": case.get("id"),
        "type": case_type,
        "group": case.get("group") or case_type,
        "condition": case.get("condition") or "unspecified",
        "query": case.get("query"),
        "expected": case.get("expected") or {},
        "observed": result["observed"],
        "passed": result["passed"],
        "failures": result["failures"],
        "latencyMs": result["latencyMs"],
        "estimatedVoiceLatencyMs": result["cost"]["estimatedVoiceLatencyMs"],
        "cost": result["cost"],
        "answer": result["answer"],
        "payload": result["payload"],
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def _group_summary(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for result in results:
        value = str(result.get(key) or "unknown")
        bucket = summary.setdefault(value, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += 1 if result.get("passed") else 0
    for bucket in summary.values():
        bucket["passRate"] = _rate(bucket["passed"], bucket["total"])
    return dict(sorted(summary.items()))


def _gate_confusion(results: list[dict[str, Any]]) -> dict[str, int]:
    matrix: dict[str, int] = {}
    for result in results:
        observed = result.get("observed") or {}
        expected = result.get("expected") or {}
        if result.get("type") == "multi_tool":
            observed_gate = (((observed.get("knowledge") or {}).get("gateStatus")) or "none")
            expected_gate = (((expected.get("knowledge") or {}).get("gateStatus")) or "none")
        else:
            observed_gate = observed.get("gateStatus") or "none"
            expected_gate = expected.get("gateStatus") or "none"
        if expected_gate == "none" and observed_gate == "none":
            continue
        key = f"expected:{expected_gate}|observed:{observed_gate}"
        matrix[key] = matrix.get(key, 0) + 1
    return dict(sorted(matrix.items()))


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.get("passed"))
    latencies = [float(result.get("latencyMs") or 0.0) for result in results]
    voice_latencies = [float(result.get("estimatedVoiceLatencyMs") or 0.0) for result in results]
    costs = [float((result.get("cost") or {}).get("vapiStackCost") or 0.0) for result in results]
    openai_costs = [float((result.get("cost") or {}).get("openaiRealtimeCost") or 0.0) for result in results]
    gemini_costs = [float((result.get("cost") or {}).get("geminiLiveCost") or 0.0) for result in results]
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "passRate": _rate(passed, total),
        "byType": _group_summary(results, "type"),
        "byGroup": _group_summary(results, "group"),
        "latency": {
            "runtimeP50Ms": _percentile(latencies, 0.50),
            "runtimeP95Ms": _percentile(latencies, 0.95),
            "voiceP50Ms": _percentile(voice_latencies, 0.50),
            "voiceP95Ms": _percentile(voice_latencies, 0.95),
        },
        "cost": {
            "totalVapiStack": round(sum(costs), 6),
            "avgVapiStack": round(statistics.mean(costs), 6) if costs else 0.0,
            "per1000VapiStack": round((statistics.mean(costs) * 1000), 4) if costs else 0.0,
            "per1000OpenAIRealtime": round((statistics.mean(openai_costs) * 1000), 4) if openai_costs else 0.0,
            "per1000GeminiLive": round((statistics.mean(gemini_costs) * 1000), 4) if gemini_costs else 0.0,
        },
        "gateConfusion": _gate_confusion(results),
        "metrics": [
            "task_success",
            "slot_coverage",
            "retrieval_source_match",
            "evidence_gate_decision",
            "faithfulness_score",
            "runtime_latency_ms",
            "estimated_voice_latency_ms",
            "estimated_cost_usd",
        ],
    }


def _save_csv(payload: dict[str, Any]) -> None:
    BENCHMARK_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rows = payload.get("results") or []
    with LATEST_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "type",
                "group",
                "condition",
                "passed",
                "latencyMs",
                "estimatedVoiceLatencyMs",
                "vapiStackCost",
                "gateStatus",
                "answerable",
                "failures",
                "answer",
            ],
        )
        writer.writeheader()
        for result in rows:
            observed = result.get("observed") or {}
            if result.get("type") == "multi_tool":
                observed_gate = ((observed.get("knowledge") or {}).get("gateStatus"))
                answerable = ((observed.get("knowledge") or {}).get("answerable"))
            else:
                observed_gate = observed.get("gateStatus")
                answerable = observed.get("answerable")
            writer.writerow(
                {
                    "id": result.get("id"),
                    "type": result.get("type"),
                    "group": result.get("group"),
                    "condition": result.get("condition"),
                    "passed": result.get("passed"),
                    "latencyMs": result.get("latencyMs"),
                    "estimatedVoiceLatencyMs": result.get("estimatedVoiceLatencyMs"),
                    "vapiStackCost": (result.get("cost") or {}).get("vapiStackCost"),
                    "gateStatus": observed_gate,
                    "answerable": answerable,
                    "failures": " | ".join(result.get("failures") or []),
                    "answer": result.get("answer"),
                }
            )


def run_benchmark_suite(
    *,
    groups: list[str] | None = None,
    case_ids: list[str] | None = None,
    limit: int | None = None,
    include_payloads: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    selected_groups = {group for group in groups or [] if group}
    selected_ids = {case_id for case_id in case_ids or [] if case_id}
    cases = load_benchmark_cases()
    if selected_groups:
        cases = [case for case in cases if case.get("group") in selected_groups or case.get("type") in selected_groups]
    if selected_ids:
        cases = [case for case in cases if case.get("id") in selected_ids]
    if limit is not None:
        cases = cases[: max(1, int(limit))]

    started = time.perf_counter()
    results = [_run_case(case) for case in cases]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    if not include_payloads:
        for result in results:
            result.pop("payload", None)

    payload = {
        "runId": datetime.now(timezone.utc).strftime("bench-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suite": "voice_retail_evidence_gated_live_suite",
        "caseFile": str(BENCHMARK_CASES_PATH.relative_to(ROOT_DIR)),
        "elapsedMs": elapsed_ms,
        "filters": {
            "groups": sorted(selected_groups),
            "caseIds": sorted(selected_ids),
            "limit": limit,
        },
        "summary": _summarize(results),
        "results": results,
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(LATEST_CSV_PATH.relative_to(ROOT_DIR)),
        },
        "researchBasis": [
            "VoiceBench-style condition slices for content and ASR robustness",
            "RAGAS-style separation of retrieval evidence and faithfulness",
            "CRAG-style evidence gating before answer release",
            "Cost and latency ledger aligned with the live voice stack",
        ],
    }

    if save:
        BENCHMARK_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with LATEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _save_csv(payload)
    return payload
