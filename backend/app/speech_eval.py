from __future__ import annotations

import csv
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inventory import tool_payload
from .knowledge import generate_evidence_gated_answer
from .pricing import estimate_costs

ROOT_DIR = Path(__file__).resolve().parents[2]
SPEECH_CASES_PATH = ROOT_DIR / "benchmarks" / "speech_cases.json"
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "benchmarks"
LATEST_JSON_PATH = ARTIFACT_DIR / "speech_latest.json"
LATEST_CSV_PATH = ARTIFACT_DIR / "speech_latest.csv"

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "but", "can", "do", "for", "from", "go",
    "have", "here", "i", "is", "it", "me", "my", "near", "of", "on", "please", "show",
    "the", "to", "what", "when", "where", "with", "you", "your",
}
ABSTAIN_TERMS = (
    "cannot",
    "cannot override",
    "do not see",
    "do not have enough supported",
    "not enough supported",
    "does not cover",
    "current shopping catalog",
    "knowledge base",
    "clarifying",
    "service desk",
    "supported store evidence",
)


def load_speech_cases() -> list[dict[str, Any]]:
    with SPEECH_CASES_PATH.open("r", encoding="utf-8-sig") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("speech case file must contain a JSON list")
    return [case for case in cases if isinstance(case, dict)]


def load_latest_speech_eval() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No speech robustness run saved yet."}
    with LATEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def _normalize(value: Any) -> str:
    return " ".join(_tokens(value))


def _entity_tokens(value: Any) -> list[str]:
    return [token for token in _tokens(value) if token not in STOP_WORDS]


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


def _wer(reference: str, transcript: str) -> tuple[float, list[str], list[str]]:
    reference_tokens = _tokens(reference)
    transcript_tokens = _tokens(transcript)
    if not reference_tokens:
        return (0.0 if not transcript_tokens else 1.0, reference_tokens, transcript_tokens)
    return _edit_distance(reference_tokens, transcript_tokens) / len(reference_tokens), reference_tokens, transcript_tokens


def _expand_entity_terms(entities: list[Any]) -> set[str]:
    terms: set[str] = set()
    for entity in entities:
        normalized = _normalize(entity)
        if normalized:
            terms.add(normalized)
        terms.update(_entity_tokens(entity))
        if normalized == "ps5":
            terms.update({"ps", "five", "5"})
        if normalized == "curbside":
            terms.update({"curb", "side"})
        if normalized == "wheelchair":
            terms.update({"wheel", "chair"})
        if normalized == "bounty":
            terms.update({"boun", "tee"})
        if normalized == "electronics":
            terms.update({"electric"})
        if normalized == "opened":
            terms.update({"open"})
        if normalized == "twenty":
            terms.update({"20"})
    return terms


def _entity_metrics(reference_tokens: list[str], transcript_tokens: list[str], entities: list[Any]) -> dict[str, Any]:
    terms = _expand_entity_terms(entities)
    reference_entities = [token for token in reference_tokens if token in terms]
    transcript_entities = [token for token in transcript_tokens if token in terms]
    expected_terms = {token for entity in entities for token in _entity_tokens(entity)} or set(reference_entities)
    if not expected_terms:
        return {
            "entityWer": None,
            "entityRecall": None,
            "referenceEntityTokens": reference_entities,
            "transcriptEntityTokens": transcript_entities,
        }
    transcript_set = set(transcript_entities)
    matched = 0
    for term in expected_terms:
        if term in transcript_set:
            matched += 1
            continue
        if term == "ps5" and ({"ps", "five"} <= transcript_set or "5" in transcript_set):
            matched += 1
        elif term == "curbside" and {"curb", "side"} <= transcript_set:
            matched += 1
        elif term == "wheelchair" and {"wheel", "chair"} <= transcript_set:
            matched += 1
        elif term == "bounty" and {"boun", "tee"} <= transcript_set:
            matched += 1
        elif term == "electronics" and "electric" in transcript_set:
            matched += 1
        elif term == "opened" and "open" in transcript_set:
            matched += 1
        elif term == "twenty" and "20" in transcript_set:
            matched += 1
        elif term.endswith("s") and term[:-1] in transcript_set:
            matched += 1
        elif f"{term}s" in transcript_set:
            matched += 1
    entity_wer = None if not reference_entities else _edit_distance(reference_entities, transcript_entities) / len(reference_entities)
    return {
        "entityWer": round(entity_wer, 4) if entity_wer is not None else None,
        "entityRecall": round(matched / len(expected_terms), 4),
        "referenceEntityTokens": reference_entities,
        "transcriptEntityTokens": transcript_entities,
    }


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


def _contains(answer: str, term: Any) -> bool:
    return _normalize(term) in _normalize(answer)


def _check_inventory(payload: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    failures: list[str] = []
    found = bool(payload.get("found"))
    answer = str(payload.get("speechAnswer") or payload.get("message") or "")
    observed_ids = _observed_item_ids(payload)
    observed_aisles = _observed_aisles(payload)
    observed = {
        "found": found,
        "matchType": payload.get("matchType"),
        "itemIds": observed_ids,
        "aisles": observed_aisles,
        "itemCount": payload.get("itemCount") or len(payload.get("matches") or []),
    }
    if "found" in expected and found is not bool(expected["found"]):
        failures.append(f"expected found={expected['found']} observed {found}")
    if expected.get("matchType") and payload.get("matchType") != expected.get("matchType"):
        failures.append(f"expected matchType {expected.get('matchType')} observed {payload.get('matchType')}")
    for item_id in expected.get("itemIds") or []:
        if item_id not in observed_ids:
            failures.append(f"missing item id {item_id}")
    for aisle in expected.get("aisles") or []:
        if aisle not in observed_aisles:
            failures.append(f"missing aisle {aisle}")
    if expected.get("minItemCount") is not None and int(observed["itemCount"] or 0) < int(expected["minItemCount"]):
        failures.append(f"item count below {expected['minItemCount']}")
    if expected.get("bestItemId"):
        best = payload.get("bestOption") if isinstance(payload.get("bestOption"), dict) else {}
        best_item = best.get("item") if isinstance(best.get("item"), dict) else {}
        observed["bestItemId"] = best_item.get("sku")
        if best_item.get("sku") != expected.get("bestItemId"):
            failures.append(f"expected best item {expected.get('bestItemId')} observed {best_item.get('sku')}")
    if expected.get("found") is False and not any(term in _normalize(answer) for term in ABSTAIN_TERMS):
        failures.append("unsupported inventory answer did not abstain")
    return not failures, failures, observed


def _check_knowledge(payload: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    failures: list[str] = []
    gate = payload.get("answerGate") if isinstance(payload.get("answerGate"), dict) else {}
    answer = str(payload.get("speechAnswer") or "")
    sources = [str(value) for value in payload.get("sources") or []]
    observed = {
        "found": bool(payload.get("found")),
        "answerable": bool(payload.get("answerable")),
        "gateStatus": gate.get("status"),
        "sources": sources,
        "faithfulnessScore": gate.get("faithfulnessScore") or payload.get("faithfulness", {}).get("faithfulnessScore"),
    }
    if "answerable" in expected and observed["answerable"] is not bool(expected["answerable"]):
        failures.append(f"expected answerable={expected['answerable']} observed {observed['answerable']}")
    if expected.get("gateStatus") and gate.get("status") != expected.get("gateStatus"):
        failures.append(f"expected gate {expected.get('gateStatus')} observed {gate.get('status')}")
    for source in expected.get("sources") or []:
        if source not in sources:
            failures.append(f"missing source {source}")
    for term in expected.get("forbiddenTerms") or []:
        if _contains(answer, term):
            failures.append(f"forbidden answer term present: {term}")
    if expected.get("answerable") is False and not any(term in _normalize(answer) for term in ABSTAIN_TERMS):
        failures.append("blocked policy answer did not abstain")
    return not failures, failures, observed


def _case_cost(query: str, answer: str, latency_ms: float, wer: float) -> dict[str, Any]:
    estimated_voice_latency_ms = int(round(latency_ms + 700 + min(1300, len(answer) * 7) + min(450, wer * 600)))
    call_ms = max(1500, estimated_voice_latency_ms + 500)
    costs = estimate_costs(call_ms=call_ms, user_text=query, answer=answer, total_latency_ms=estimated_voice_latency_ms)
    rows = {row["id"]: row for row in costs.get("rows", [])}
    return {
        "estimatedVoiceLatencyMs": estimated_voice_latency_ms,
        "callMs": call_ms,
        "vapiStackCost": rows.get("vapi-stack", {}).get("cost", 0.0),
        "vapiStackPer1000": rows.get("vapi-stack", {}).get("per1000", 0.0),
        "openaiRealtimeCost": rows.get("openai", {}).get("cost", 0.0),
        "geminiLiveCost": rows.get("gemini", {}).get("cost", 0.0),
    }


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    reference = str(case.get("referenceText") or "")
    transcript = str(case.get("transcriptText") or reference)
    wer, reference_tokens, transcript_tokens = _wer(reference, transcript)
    entities = _entity_metrics(reference_tokens, transcript_tokens, case.get("entities") or [])
    route = str(case.get("route") or "inventory")
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    started = time.perf_counter()
    if route == "knowledge":
        payload = generate_evidence_gated_answer(transcript, limit=4)
        downstream_passed, failures, observed = _check_knowledge(payload, expected)
        answer = str(payload.get("speechAnswer") or "")
    else:
        payload = tool_payload(transcript)
        downstream_passed, failures, observed = _check_inventory(payload, expected)
        answer = str(payload.get("speechAnswer") or payload.get("message") or "")
    latency_ms = (time.perf_counter() - started) * 1000

    max_wer = expected.get("maxWer")
    min_entity_recall = expected.get("minEntityRecall")
    asr_passed = True
    if max_wer is not None and wer > float(max_wer):
        asr_passed = False
        failures.append(f"WER {wer:.3f} above {float(max_wer):.3f}")
    entity_recall = entities.get("entityRecall")
    if min_entity_recall is not None and entity_recall is not None and float(entity_recall) < float(min_entity_recall):
        asr_passed = False
        failures.append(f"entity recall {float(entity_recall):.3f} below {float(min_entity_recall):.3f}")

    cost = _case_cost(transcript, answer, latency_ms, wer)
    return {
        "id": case.get("id"),
        "route": route,
        "group": case.get("group") or route,
        "condition": case.get("condition") or {},
        "referenceText": reference,
        "transcriptText": transcript,
        "audioUri": case.get("audioUri"),
        "wer": round(wer, 4),
        "entityWer": entities.get("entityWer"),
        "entityRecall": entities.get("entityRecall"),
        "referenceEntityTokens": entities.get("referenceEntityTokens") or [],
        "transcriptEntityTokens": entities.get("transcriptEntityTokens") or [],
        "asrPassed": asr_passed,
        "downstreamPassed": downstream_passed,
        "passed": bool(asr_passed and downstream_passed),
        "failures": failures,
        "observed": observed,
        "answer": answer,
        "latencyMs": round(latency_ms, 2),
        "estimatedVoiceLatencyMs": cost["estimatedVoiceLatencyMs"],
        "cost": cost,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _avg(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.mean(clean), 4) if clean else None


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


def _condition_key(condition: dict[str, Any]) -> str:
    return f"{condition.get('accent', 'unknown')}|{condition.get('noise', 'unknown')}|barge:{bool(condition.get('bargeIn'))}"


def _bucket_summary(results: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for result in results:
        key = str(key_fn(result))
        bucket = buckets.setdefault(key, {"total": 0, "passed": 0, "wer": [], "entityRecall": []})
        bucket["total"] += 1
        bucket["passed"] += 1 if result.get("passed") else 0
        bucket["wer"].append(result.get("wer"))
        bucket["entityRecall"].append(result.get("entityRecall"))
    for bucket in buckets.values():
        bucket["passRate"] = _rate(bucket["passed"], bucket["total"])
        bucket["avgWer"] = _avg(bucket.pop("wer"))
        bucket["avgEntityRecall"] = _avg(bucket.pop("entityRecall"))
    return dict(sorted(buckets.items()))


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.get("passed"))
    asr_passed = sum(1 for result in results if result.get("asrPassed"))
    downstream_passed = sum(1 for result in results if result.get("downstreamPassed"))
    latencies = [float(result.get("latencyMs") or 0.0) for result in results]
    voice_latencies = [float(result.get("estimatedVoiceLatencyMs") or 0.0) for result in results]
    costs = [float((result.get("cost") or {}).get("vapiStackCost") or 0.0) for result in results]
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "passRate": _rate(passed, total),
        "asrPassRate": _rate(asr_passed, total),
        "downstreamTaskSuccess": _rate(downstream_passed, total),
        "avgWer": _avg([result.get("wer") for result in results]),
        "avgEntityWer": _avg([result.get("entityWer") for result in results]),
        "avgEntityRecall": _avg([result.get("entityRecall") for result in results]),
        "byRoute": _bucket_summary(results, lambda result: result.get("route")),
        "byGroup": _bucket_summary(results, lambda result: result.get("group")),
        "byCondition": _bucket_summary(results, lambda result: _condition_key(result.get("condition") or {})),
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
        },
        "metrics": [
            "word_error_rate",
            "entity_word_error_rate",
            "entity_recall",
            "downstream_task_success",
            "evidence_gate_correctness",
            "runtime_latency_ms",
            "estimated_voice_latency_ms",
            "estimated_cost_usd",
        ],
    }


def _save_csv(payload: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with LATEST_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id", "route", "group", "accent", "noise", "bargeIn", "passed", "asrPassed",
                "downstreamPassed", "wer", "entityWer", "entityRecall", "latencyMs",
                "estimatedVoiceLatencyMs", "vapiStackCost", "failures", "referenceText", "transcriptText", "answer",
            ],
        )
        writer.writeheader()
        for result in payload.get("results") or []:
            condition = result.get("condition") or {}
            writer.writerow(
                {
                    "id": result.get("id"),
                    "route": result.get("route"),
                    "group": result.get("group"),
                    "accent": condition.get("accent"),
                    "noise": condition.get("noise"),
                    "bargeIn": condition.get("bargeIn"),
                    "passed": result.get("passed"),
                    "asrPassed": result.get("asrPassed"),
                    "downstreamPassed": result.get("downstreamPassed"),
                    "wer": result.get("wer"),
                    "entityWer": result.get("entityWer"),
                    "entityRecall": result.get("entityRecall"),
                    "latencyMs": result.get("latencyMs"),
                    "estimatedVoiceLatencyMs": result.get("estimatedVoiceLatencyMs"),
                    "vapiStackCost": (result.get("cost") or {}).get("vapiStackCost"),
                    "failures": " | ".join(result.get("failures") or []),
                    "referenceText": result.get("referenceText"),
                    "transcriptText": result.get("transcriptText"),
                    "answer": result.get("answer"),
                }
            )


def run_speech_robustness_suite(
    *,
    groups: list[str] | None = None,
    conditions: list[str] | None = None,
    limit: int | None = None,
    save: bool = True,
) -> dict[str, Any]:
    selected_groups = {group for group in groups or [] if group}
    selected_conditions = {condition for condition in conditions or [] if condition}
    cases = load_speech_cases()
    if selected_groups:
        cases = [case for case in cases if case.get("group") in selected_groups or case.get("route") in selected_groups]
    if selected_conditions:
        cases = [case for case in cases if _condition_key(case.get("condition") or {}) in selected_conditions]
    if limit is not None:
        cases = cases[: max(1, int(limit))]

    started = time.perf_counter()
    results = [_run_case(case) for case in cases]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = {
        "runId": datetime.now(timezone.utc).strftime("speech-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suite": "voice_retail_speech_robustness_proxy_suite",
        "caseFile": str(SPEECH_CASES_PATH.relative_to(ROOT_DIR)),
        "elapsedMs": elapsed_ms,
        "filters": {
            "groups": sorted(selected_groups),
            "conditions": sorted(selected_conditions),
            "limit": limit,
        },
        "summary": _summarize(results),
        "results": results,
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(LATEST_CSV_PATH.relative_to(ROOT_DIR)),
        },
        "transcriptionMode": "reference_transcript_proxy",
        "deepgramReady": True,
        "researchBasis": [
            "VoiceBench-style accent, noise, content, and barge-in condition slices",
            "WER and entity-WER for ASR quality",
            "Downstream task success after ASR perturbation",
            "Evidence-gate correctness for spoken policy questions",
        ],
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with LATEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _save_csv(payload)
    return payload
