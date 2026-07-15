from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark_suite import _run_case as _run_benchmark_case
from .benchmark_suite import _summarize as _summarize_benchmark
from .benchmark_suite import load_benchmark_cases
from .case_factory import generate_case_factory, load_latest_case_factory
from .speech_eval import _run_case as _run_speech_case
from .speech_eval import _summarize as _summarize_speech
from .speech_eval import load_speech_cases


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "paper"
LATEST_JSON_PATH = ARTIFACT_DIR / "draft_validation_latest.json"
LATEST_CSV_PATH = ARTIFACT_DIR / "draft_validation_latest.csv"
PROMOTION_JSON_PATH = ARTIFACT_DIR / "promotion_manifest_latest.json"
PROMOTION_CSV_PATH = ARTIFACT_DIR / "promotion_manifest_latest.csv"


def load_latest_draft_validation() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No draft validation saved yet."}
    with LATEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _case_id(case: dict[str, Any]) -> str:
    return str(case.get("id") or "")


def _duplicate_ids(cases: list[dict[str, Any]], official_ids: set[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        case_id = _case_id(case)
        if not case_id:
            continue
        if case_id in seen or case_id in official_ids:
            duplicates.add(case_id)
        seen.add(case_id)
    return duplicates


def _benchmark_schema_errors(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["id", "type", "group", "condition", "query", "expected"]:
        if case.get(key) in (None, "", []):
            errors.append(f"missing {key}")
    if case.get("type") not in {"inventory", "knowledge", "multi_tool"}:
        errors.append(f"unsupported type {case.get('type')}")
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    if not isinstance(case.get("expected"), dict):
        errors.append("expected must be object")
    if case.get("type") == "inventory" and "found" not in expected:
        errors.append("inventory expected missing found")
    if case.get("type") == "knowledge" and "answerable" not in expected:
        errors.append("knowledge expected missing answerable")
    if case.get("type") == "multi_tool":
        if not isinstance(expected.get("inventory"), dict):
            errors.append("multi_tool expected missing inventory object")
        if not isinstance(expected.get("knowledge"), dict):
            errors.append("multi_tool expected missing knowledge object")
    return errors


def _speech_schema_errors(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["id", "route", "group", "condition", "referenceText", "transcriptText", "expected"]:
        if case.get(key) in (None, "", []):
            errors.append(f"missing {key}")
    if case.get("route") not in {"inventory", "knowledge"}:
        errors.append(f"unsupported route {case.get('route')}")
    if not isinstance(case.get("condition"), dict):
        errors.append("condition must be object")
    if not isinstance(case.get("expected"), dict):
        errors.append("expected must be object")
    return errors


def _audio_schema_errors(prompt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["id", "templateId", "referenceText", "recommendedStratum", "recordingInstruction"]:
        if prompt.get(key) in (None, "", []):
            errors.append(f"missing {key}")
    return errors


def _validation_status(*, passed: bool, schema_errors: list[str], duplicate: bool) -> str:
    if duplicate:
        return "duplicate_id"
    if schema_errors:
        return "schema_blocked"
    if not passed:
        return "scoring_blocked"
    return "promotion_ready"


def _benchmark_validation_row(case: dict[str, Any], result: dict[str, Any], schema_errors: list[str], duplicate: bool) -> dict[str, Any]:
    passed = bool(result.get("passed"))
    status = _validation_status(passed=passed, schema_errors=schema_errors, duplicate=duplicate)
    failures = [*schema_errors, *(result.get("failures") or [])]
    return {
        "artifactType": "benchmark_case",
        "id": case.get("id"),
        "group": case.get("group"),
        "route": case.get("type"),
        "status": status,
        "promotionReady": status == "promotion_ready",
        "passed": passed,
        "schemaOk": not schema_errors,
        "duplicateId": duplicate,
        "failureCount": len(failures),
        "failures": failures,
        "query": case.get("query"),
        "answer": result.get("answer"),
        "latencyMs": result.get("latencyMs"),
        "estimatedVoiceLatencyMs": result.get("estimatedVoiceLatencyMs"),
        "vapiStackCost": (result.get("cost") or {}).get("vapiStackCost"),
        "case": case,
        "result": result,
    }


def _speech_validation_row(case: dict[str, Any], result: dict[str, Any], schema_errors: list[str], duplicate: bool) -> dict[str, Any]:
    passed = bool(result.get("passed"))
    status = _validation_status(passed=passed, schema_errors=schema_errors, duplicate=duplicate)
    failures = [*schema_errors, *(result.get("failures") or [])]
    return {
        "artifactType": "speech_case",
        "id": case.get("id"),
        "group": case.get("group"),
        "route": case.get("route"),
        "status": status,
        "promotionReady": status == "promotion_ready",
        "passed": passed,
        "schemaOk": not schema_errors,
        "duplicateId": duplicate,
        "failureCount": len(failures),
        "failures": failures,
        "referenceText": case.get("referenceText"),
        "transcriptText": case.get("transcriptText"),
        "wer": result.get("wer"),
        "entityRecall": result.get("entityRecall"),
        "latencyMs": result.get("latencyMs"),
        "estimatedVoiceLatencyMs": result.get("estimatedVoiceLatencyMs"),
        "vapiStackCost": (result.get("cost") or {}).get("vapiStackCost"),
        "case": case,
        "result": result,
    }


def _audio_validation_row(prompt: dict[str, Any], schema_errors: list[str]) -> dict[str, Any]:
    status = "schema_blocked" if schema_errors else "recording_ready"
    return {
        "artifactType": "audio_recording_prompt",
        "id": prompt.get("id"),
        "group": prompt.get("group"),
        "route": prompt.get("route"),
        "status": status,
        "promotionReady": status == "recording_ready",
        "passed": status == "recording_ready",
        "schemaOk": not schema_errors,
        "duplicateId": False,
        "failureCount": len(schema_errors),
        "failures": schema_errors,
        "referenceText": prompt.get("referenceText"),
        "recommendedStratum": prompt.get("recommendedStratum"),
        "case": prompt,
    }


def _summary(rows: list[dict[str, Any]], benchmark_results: list[dict[str, Any]], speech_results: list[dict[str, Any]]) -> dict[str, Any]:
    benchmark_rows = [row for row in rows if row.get("artifactType") == "benchmark_case"]
    speech_rows = [row for row in rows if row.get("artifactType") == "speech_case"]
    audio_rows = [row for row in rows if row.get("artifactType") == "audio_recording_prompt"]
    ready_rows = [row for row in rows if row.get("promotionReady")]
    blocked_rows = [row for row in rows if not row.get("promotionReady")]
    return {
        "totalDraftArtifacts": len(rows),
        "promotionReady": len(ready_rows),
        "blocked": len(blocked_rows),
        "promotionReadyRate": round(len(ready_rows) / len(rows), 4) if rows else 0.0,
        "benchmarkDrafts": len(benchmark_rows),
        "benchmarkPromotionReady": sum(1 for row in benchmark_rows if row.get("promotionReady")),
        "speechDrafts": len(speech_rows),
        "speechPromotionReady": sum(1 for row in speech_rows if row.get("promotionReady")),
        "audioPrompts": len(audio_rows),
        "audioRecordingReady": sum(1 for row in audio_rows if row.get("promotionReady")),
        "schemaBlocked": sum(1 for row in rows if row.get("status") == "schema_blocked"),
        "scoringBlocked": sum(1 for row in rows if row.get("status") == "scoring_blocked"),
        "duplicateBlocked": sum(1 for row in rows if row.get("status") == "duplicate_id"),
        "benchmarkSummary": _summarize_benchmark(benchmark_results) if benchmark_results else {},
        "speechSummary": _summarize_speech(speech_results) if speech_results else {},
    }


def _promotion_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready_benchmark = [row["case"] for row in rows if row.get("artifactType") == "benchmark_case" and row.get("promotionReady")]
    ready_speech = [row["case"] for row in rows if row.get("artifactType") == "speech_case" and row.get("promotionReady")]
    ready_audio = [row["case"] for row in rows if row.get("artifactType") == "audio_recording_prompt" and row.get("promotionReady")]
    blocked = [
        {
            "artifactType": row.get("artifactType"),
            "id": row.get("id"),
            "status": row.get("status"),
            "failures": row.get("failures"),
        }
        for row in rows
        if not row.get("promotionReady")
    ]
    return {
        "readyBenchmarkCases": ready_benchmark,
        "readySpeechCases": ready_speech,
        "readyAudioRecordingPrompts": ready_audio,
        "blocked": blocked,
        "counts": {
            "readyBenchmarkCases": len(ready_benchmark),
            "readySpeechCases": len(ready_speech),
            "readyAudioRecordingPrompts": len(ready_audio),
            "blocked": len(blocked),
        },
        "promotionNotes": [
            "Promotion manifest is review-only; it does not mutate official benchmark files.",
            "Append readyBenchmarkCases to benchmarks/eval_cases.json only after reviewing expected fields.",
            "Append readySpeechCases to benchmarks/speech_cases.json only after checking transcript perturbations.",
            "Record readyAudioRecordingPrompts through the Real Audio Eval panel rather than adding fake audio URIs.",
        ],
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "artifactType",
        "id",
        "group",
        "route",
        "status",
        "promotionReady",
        "passed",
        "schemaOk",
        "duplicateId",
        "failureCount",
        "failures",
        "query",
        "referenceText",
        "transcriptText",
        "recommendedStratum",
        "wer",
        "entityRecall",
        "latencyMs",
        "estimatedVoiceLatencyMs",
        "vapiStackCost",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "failures": " | ".join(row.get("failures") or [])})


def _write_promotion_csv(manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for case in manifest.get("readyBenchmarkCases") or []:
        rows.append({"artifactType": "benchmark_case", "id": case.get("id"), "promotionTarget": "benchmarks/eval_cases.json", "status": "ready"})
    for case in manifest.get("readySpeechCases") or []:
        rows.append({"artifactType": "speech_case", "id": case.get("id"), "promotionTarget": "benchmarks/speech_cases.json", "status": "ready"})
    for prompt in manifest.get("readyAudioRecordingPrompts") or []:
        rows.append({"artifactType": "audio_recording_prompt", "id": prompt.get("id"), "promotionTarget": "Real Audio Eval recording queue", "status": "record"})
    for row in manifest.get("blocked") or []:
        rows.append({"artifactType": row.get("artifactType"), "id": row.get("id"), "promotionTarget": "-", "status": row.get("status")})
    with PROMOTION_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifactType", "id", "promotionTarget", "status"])
        writer.writeheader()
        writer.writerows(rows)


def run_draft_validation(
    *,
    refresh_factory: bool = False,
    limit: int | None = None,
    include_payloads: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    factory = generate_case_factory(save=True) if refresh_factory else load_latest_case_factory()
    if factory.get("found") is False:
        factory = generate_case_factory(save=True)

    benchmark_cases = [case for case in factory.get("benchmarkCases") or [] if isinstance(case, dict)]
    speech_cases = [case for case in factory.get("speechCases") or [] if isinstance(case, dict)]
    audio_prompts = [case for case in factory.get("audioRecordingPrompts") or [] if isinstance(case, dict)]
    if limit is not None:
        safe_limit = max(1, int(limit))
        benchmark_cases = benchmark_cases[:safe_limit]
        speech_cases = speech_cases[:safe_limit]
        audio_prompts = audio_prompts[:safe_limit]

    official_benchmark_ids = {str(case.get("id")) for case in load_benchmark_cases() if isinstance(case, dict)}
    official_speech_ids = {str(case.get("id")) for case in load_speech_cases() if isinstance(case, dict)}
    duplicate_benchmark = _duplicate_ids(benchmark_cases, official_benchmark_ids)
    duplicate_speech = _duplicate_ids(speech_cases, official_speech_ids)

    started = time.perf_counter()
    benchmark_results: list[dict[str, Any]] = []
    speech_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for case in benchmark_cases:
        schema_errors = _benchmark_schema_errors(case)
        result = _run_benchmark_case(case) if not schema_errors else {"passed": False, "failures": [], "answer": "", "cost": {}}
        if not include_payloads:
            result.pop("payload", None)
        benchmark_results.append(result)
        rows.append(_benchmark_validation_row(case, result, schema_errors, _case_id(case) in duplicate_benchmark))

    for case in speech_cases:
        schema_errors = _speech_schema_errors(case)
        result = _run_speech_case(case) if not schema_errors else {"passed": False, "failures": [], "cost": {}}
        speech_results.append(result)
        rows.append(_speech_validation_row(case, result, schema_errors, _case_id(case) in duplicate_speech))

    for prompt in audio_prompts:
        rows.append(_audio_validation_row(prompt, _audio_schema_errors(prompt)))

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    manifest = _promotion_manifest(rows)
    payload = {
        "found": True,
        "runId": datetime.now(timezone.utc).strftime("draft-val-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suite": "paper_draft_validation_gate",
        "elapsedMs": elapsed_ms,
        "inputs": {
            "caseFactoryRunId": factory.get("runId"),
            "limit": limit,
            "refreshFactory": refresh_factory,
        },
        "summary": _summary(rows, benchmark_results, speech_results),
        "rows": rows,
        "promotionManifest": manifest,
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(LATEST_CSV_PATH.relative_to(ROOT_DIR)),
            "promotionManifestJson": str(PROMOTION_JSON_PATH.relative_to(ROOT_DIR)),
            "promotionManifestCsv": str(PROMOTION_CSV_PATH.relative_to(ROOT_DIR)),
        },
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with LATEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        with PROMOTION_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        _write_csv(rows, LATEST_CSV_PATH)
        _write_promotion_csv(manifest)
    return payload
