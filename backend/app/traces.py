from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .inventory import tool_payload
from .knowledge import search_knowledge


ROOT_DIR = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT_DIR / "artifacts" / "traces"
MAX_TEXT_LENGTH = 1800
MAX_EVENTS = 500

PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED_CARD]"),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
]

SECRET_FIELD_NAMES = {
    "apikey",
    "api_key",
    "api-key",
    "authorization",
    "access_token",
    "refresh_token",
    "auth_token",
    "secret",
    "client_secret",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_trace_id(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-")
    if len(raw) >= 8:
        return raw[:80]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"trace-{stamp}-{uuid4().hex[:8]}"


def _redact_text(value: str) -> str:
    redacted = value
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    if len(redacted) > MAX_TEXT_LENGTH:
        redacted = redacted[: MAX_TEXT_LENGTH - 3].rstrip() + "..."
    return redacted


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_redact(item) for item in value[:MAX_EVENTS]]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lower_key = str(key).lower()
            normalized_key = lower_key.replace("-", "_")
            if lower_key in SECRET_FIELD_NAMES or normalized_key in SECRET_FIELD_NAMES or normalized_key.endswith("_api_key"):
                redacted[key] = "[REDACTED_KEY]"
            else:
                redacted[key] = _redact(item)
        return redacted
    return value


def _events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    events = trace.get("events")
    return events if isinstance(events, list) else []


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    events = _events(trace)
    event_types = [str(event.get("type") or "") for event in events if isinstance(event, dict)]
    transcript_events = [
        event for event in events if event.get("type") == "transcript" and event.get("payload", {}).get("role") == "user"
    ]
    assistant_events = [
        event
        for event in events
        if event.get("type") == "transcript" and event.get("payload", {}).get("role") == "assistant"
    ]
    tool_result_events = [event for event in events if event.get("type") == "tool_result"]
    ledger_events = [event for event in events if event.get("type") == "ledger"]
    max_event_ms = max((int(event.get("relativeMs") or 0) for event in events if isinstance(event, dict)), default=0)
    final_ledger = ledger_events[-1].get("payload", {}) if ledger_events else {}
    rows = final_ledger.get("rows") if isinstance(final_ledger, dict) else []
    live_row = next((row for row in rows if row.get("id") == "vapi-stack"), {}) if isinstance(rows, list) else {}
    rag_results = [
        event
        for event in tool_result_events
        if event.get("payload", {}).get("tool") == "search_knowledge" and event.get("payload", {}).get("found")
    ]
    unsupported_results = [
        event for event in tool_result_events if event.get("payload", {}).get("found") is False
    ]
    latency_values = [
        value
        for value in (
            trace.get("metrics", {}).get("totalMs"),
            live_row.get("latencyMs") if isinstance(live_row, dict) else None,
        )
        if isinstance(value, (int, float))
    ]

    return {
        "traceId": trace.get("traceId"),
        "createdAt": trace.get("createdAt"),
        "savedAt": trace.get("savedAt"),
        "architecture": trace.get("architecture", "vapi-managed-cascade"),
        "status": trace.get("status", "recorded"),
        "durationMs": int(trace.get("durationMs") or max_event_ms),
        "eventCount": len(events),
        "turns": int(trace.get("turnCount") or event_types.count("turn_complete")),
        "userTranscriptCount": len(transcript_events),
        "assistantTranscriptCount": len(assistant_events),
        "toolCalls": event_types.count("tool_call"),
        "toolResults": len(tool_result_events),
        "ragEvidenceEvents": len(rag_results),
        "unsupportedToolResults": len(unsupported_results),
        "estimatedCost": live_row.get("cost") if isinstance(live_row, dict) else None,
        "estimatedCostPer1000": live_row.get("per1000") if isinstance(live_row, dict) else None,
        "latencyMs": latency_values[-1] if latency_values else None,
    }


def _trace_path(trace_id: str) -> Path:
    safe_id = _safe_trace_id(trace_id)
    return TRACE_DIR / f"{safe_id}.json"


def save_trace(payload: dict[str, Any]) -> dict[str, Any]:
    trace = _redact(payload)
    if not isinstance(trace, dict):
        raise ValueError("trace payload must be an object")
    trace_id = _safe_trace_id(trace.get("traceId"))
    trace["traceId"] = trace_id
    trace.setdefault("createdAt", _utc_now())
    trace["savedAt"] = _utc_now()
    trace["events"] = _events(trace)[:MAX_EVENTS]
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    _trace_path(trace_id).write_text(json.dumps(trace, indent=2), encoding="utf-8")
    summary = summarize_trace(trace)
    return {"ok": True, "summary": summary}


def list_traces(limit: int = 20) -> dict[str, Any]:
    if not TRACE_DIR.exists():
        return {"traces": []}
    traces: list[dict[str, Any]] = []
    for path in sorted(TRACE_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
            traces.append(summarize_trace(trace))
        except (OSError, json.JSONDecodeError):
            continue
    return {"traces": traces}


def load_trace(trace_id: str) -> dict[str, Any]:
    path = _trace_path(trace_id)
    if not path.is_file():
        raise FileNotFoundError(trace_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _stored_tool_results(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("type") != "tool_result":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            grouped.setdefault(str(payload.get("query") or ""), []).append(payload)
    return grouped


def replay_trace(trace_id: str) -> dict[str, Any]:
    trace = load_trace(trace_id)
    events = _events(trace)
    stored_results = _stored_tool_results(events)
    replays: list[dict[str, Any]] = []

    for event in events:
        if event.get("type") != "tool_call":
            continue
        payload = event.get("payload") or {}
        tool_name = payload.get("name") or payload.get("tool")
        query = str(payload.get("query") or "")
        if not query or tool_name not in {"lookup_inventory", "search_knowledge"}:
            continue

        current = tool_payload(query) if tool_name == "lookup_inventory" else search_knowledge(query)
        stored = (stored_results.get(query) or [{}])[0]
        current_sources = current.get("sources") or []
        stored_sources = stored.get("sources") or []
        current_item = (current.get("item") or {}).get("sku")
        stored_item = (stored.get("item") or {}).get("sku")
        deterministic_match = bool(
            current.get("found") == stored.get("found")
            and (not stored_sources or set(stored_sources).issubset(set(current_sources)))
            and (not stored_item or current_item == stored_item)
        )
        replays.append(
            {
                "tool": tool_name,
                "query": query,
                "storedFound": stored.get("found"),
                "currentFound": current.get("found"),
                "storedSources": stored_sources,
                "currentSources": current_sources,
                "storedItem": stored_item,
                "currentItem": current_item,
                "deterministicMatch": deterministic_match,
                "currentConfidence": (current.get("retrieval") or {}).get("confidence") or current.get("score"),
            }
        )

    matches = sum(item["deterministicMatch"] for item in replays)
    return {
        "trace": summarize_trace(trace),
        "replay": {
            "toolCalls": len(replays),
            "deterministicMatches": matches,
            "deterministicMatchRate": matches / len(replays) if replays else None,
            "items": replays,
        },
    }
