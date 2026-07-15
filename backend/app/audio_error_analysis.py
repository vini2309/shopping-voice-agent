from __future__ import annotations

import csv
import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio_accepted import load_latest_audio_accepted_set
from .audio_eval import load_latest_audio_eval
from .audio_quality import load_latest_audio_quality


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "audio_eval"
LATEST_JSON_PATH = ARTIFACT_DIR / "error_analysis_latest.json"
LATEST_CSV_PATH = ARTIFACT_DIR / "error_analysis_latest.csv"
ACTION_CSV_PATH = ARTIFACT_DIR / "error_action_plan_latest.csv"

SPANISH_HINTS = {
    "puede",
    "podria",
    "podrias",
    "cual",
    "inventario",
    "comida",
    "perros",
    "sistema",
    "muestra",
    "existencias",
    "devolver",
    "articulos",
    "electronicos",
    "abiertos",
    "realidad",
    "ensenarme",
    "lugar",
}

STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "is",
    "are",
    "do",
    "you",
    "your",
    "can",
    "me",
    "all",
    "what",
    "where",
    "show",
    "please",
    "actually",
    "instead",
    "available",
}

SEVERITY_SCORE = {"blocker": 4, "high": 3, "medium": 2, "low": 1, "none": 0}


def load_latest_audio_error_analysis() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No audio error analysis saved yet."}
    with LATEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _tokens(value: Any) -> list[str]:
    text = _clean(value).lower()
    text = text.translate(str.maketrans({"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}))
    return re.findall(r"[a-z0-9]+", text)


def _meaningful_tokens(value: Any) -> list[str]:
    return [token for token in _tokens(value) if len(token) > 2 and token not in STOP_WORDS]


def _token_overlap(reference: Any, transcript: Any) -> float:
    left = set(_meaningful_tokens(reference))
    right = set(_meaningful_tokens(transcript))
    if not left:
        return 1.0 if not right else 0.0
    return round(len(left & right) / len(left), 4)


def _token_diff(reference: Any, transcript: Any) -> tuple[list[str], list[str]]:
    left = _meaningful_tokens(reference)
    right = _meaningful_tokens(transcript)
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    missing: list[str] = []
    extra: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            missing.extend(left[i1:i2])
        if tag in {"insert", "replace"}:
            extra.extend(right[j1:j2])
    return list(dict.fromkeys(missing))[:8], list(dict.fromkeys(extra))[:8]


def _spanish_like(text: Any) -> bool:
    raw = str(text or "").lower()
    if any(char in raw for char in "¿¡áéíóúñ"):
        return True
    tokens = set(_tokens(raw))
    return len(tokens & SPANISH_HINTS) >= 2


def _condition(result: dict[str, Any]) -> dict[str, Any]:
    condition = result.get("condition") if isinstance(result.get("condition"), dict) else {}
    metadata = result.get("recordingMetadata") if isinstance(result.get("recordingMetadata"), dict) else {}
    speaker = result.get("speaker") if isinstance(result.get("speaker"), dict) else {}
    return {
        "accent": metadata.get("accent") or speaker.get("accent") or condition.get("accent") or "unknown",
        "noise": metadata.get("noise") or condition.get("noise") or "unknown",
        "device": metadata.get("device") or condition.get("device") or "unknown",
        "bargeIn": bool(condition.get("bargeIn")),
    }


def _quality_by_id() -> dict[str, dict[str, Any]]:
    payload = load_latest_audio_quality()
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    return {str(row.get("id")): row for row in rows if isinstance(row, dict) and row.get("id")}


def _accepted_by_id() -> dict[str, dict[str, Any]]:
    payload = load_latest_audio_accepted_set()
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    return {str(row.get("recordingId")): row for row in rows if isinstance(row, dict) and row.get("recordingId")}


def _accepted_groups() -> list[dict[str, Any]]:
    payload = load_latest_audio_accepted_set()
    groups = payload.get("groups") if isinstance(payload.get("groups"), list) else []
    return [group for group in groups if isinstance(group, dict)]


def _failure_modes(result: dict[str, Any], quality: dict[str, Any] | None) -> list[str]:
    if quality and isinstance(quality.get("failureModes"), list):
        return [str(mode) for mode in quality["failureModes"]]
    modes: list[str] = []
    transcript = _clean(result.get("transcriptText"))
    wer = _number(result.get("wer")) or 0.0
    entity = _number(result.get("entityRecall"))
    if result.get("skipped"):
        modes.append("skipped")
    if not transcript:
        modes.append("empty_transcript")
    if wer > 0.35:
        modes.append("high_wer")
    if entity is not None and entity < 0.66:
        modes.append("entity_miss")
    if result.get("asrPassed") is False:
        modes.append("asr_failed")
    if result.get("downstreamPassed") is False:
        modes.append("downstream_failed")
    if not modes and result.get("passed"):
        modes.append("pass")
    return modes or ["needs_review"]


def _class_payload(
    *,
    primary: str,
    family: str,
    severity: str,
    root_cause: str,
    recommended_action: str,
    paper_action: str,
) -> dict[str, str]:
    return {
        "primaryFailure": primary,
        "failureFamily": family,
        "severity": severity,
        "rootCause": root_cause,
        "recommendedAction": recommended_action,
        "paperAction": paper_action,
    }


def _classify(result: dict[str, Any], quality: dict[str, Any] | None, accepted_row: dict[str, Any] | None) -> dict[str, str]:
    transcript = _clean(result.get("transcriptText"))
    reference = _clean(result.get("referenceText"))
    modes = _failure_modes(result, quality)
    wer = _number(result.get("wer")) or 0.0
    entity = _number(result.get("entityRecall"))
    latency = _number(result.get("transcriptionLatencyMs"))
    confidence = _number(result.get("confidence"))
    downstream_failed = result.get("downstreamPassed") is False
    asr_failed = result.get("asrPassed") is False
    semantic = result.get("semanticTranscript") if isinstance(result.get("semanticTranscript"), dict) else {}
    overlap = _token_overlap(reference, transcript)
    accepted_status = str((accepted_row or {}).get("reviewStatus") or "")

    if result.get("skipped"):
        return _class_payload(
            primary="provider_or_fixture_skipped",
            family="data_collection",
            severity="blocker",
            root_cause="Recording could not be evaluated by the provider pipeline.",
            recommended_action="Check provider credentials and audio file availability before rerunning this fixture.",
            paper_action="exclude_until_evaluated",
        )
    if result.get("passed"):
        if latency and latency > 1000:
            return _class_payload(
                primary="pass_with_tail_latency",
                family="latency",
                severity="low",
                root_cause="Task passed, but ASR latency is above the current tail-latency threshold.",
                recommended_action="Trim leading and trailing silence or measure streaming ASR partial latency separately.",
                paper_action="keep_with_latency_note",
            )
        return _class_payload(
            primary="pass",
            family="pass",
            severity="none",
            root_cause="No blocking failure detected.",
            recommended_action="Keep this fixture for accepted-set or raw archive analysis.",
            paper_action="keep",
        )
    if not transcript:
        return _class_payload(
            primary="empty_transcript",
            family="data_collection",
            severity="blocker",
            root_cause="The recording produced no usable speech transcript.",
            recommended_action="Retake with a closer microphone, visible waveform, and one second of speech padding.",
            paper_action="retake",
        )
    if semantic.get("passed") and not result.get("passed"):
        return _class_payload(
            primary=str(semantic.get("label") or "semantic_transcript_pass"),
            family="semantic_asr",
            severity="low",
            root_cause=str(semantic.get("reason") or "The transcript preserves the task intent and required slots despite strict ASR metric failure."),
            recommended_action="Keep for semantic task-preservation claims; continue reporting literal WER separately.",
            paper_action="keep_for_semantic_transcript_claim",
        )
    if _spanish_like(transcript) and wer >= 0.8:
        return _class_payload(
            primary="language_mismatch_not_accent",
            family="language",
            severity="high",
            root_cause="The fixture appears to be Spanish-language speech while the reference and downstream task expect English.",
            recommended_action="Separate language robustness from accent robustness; either collect English speech with the target accent or add language detection plus translation before retrieval.",
            paper_action="label_as_language_mismatch",
        )
    if "wrong_prompt_or_unintelligible" in modes or (wer >= 0.8 and overlap < 0.25):
        return _class_payload(
            primary="prompt_drift_or_wrong_clip",
            family="data_collection",
            severity="high",
            root_cause="The transcript does not match the requested benchmark prompt.",
            recommended_action="Retake the exact prompt and keep paraphrases in a separate intent-level evaluation split.",
            paper_action="retake_or_move_to_intent_split",
        )
    if asr_failed and result.get("downstreamPassed") is True:
        if entity is not None and entity < 0.66:
            return _class_payload(
                primary="entity_miss_with_task_recovery",
                family="asr",
                severity="medium",
                root_cause="Downstream answer survived, but ASR missed one or more reference entities.",
                recommended_action="Add product, aisle, and policy keyterms; report entity recall separately from task success.",
                paper_action="keep_for_task_success_not_asr_claim",
            )
        return _class_payload(
            primary="asr_metric_miss_task_recovered",
            family="asr",
            severity="medium",
            root_cause="The assistant answered correctly, but the transcript exceeded the strict WER threshold.",
            recommended_action="Keep task success and WER as separate metrics; add semantic transcript scoring for paraphrases.",
            paper_action="keep_for_task_success_not_wer_claim",
        )
    if entity is not None and entity < 0.66:
        return _class_payload(
            primary="entity_miss",
            family="asr",
            severity="high",
            root_cause="The transcript lost key product or policy entities needed by retrieval.",
            recommended_action="Expand Deepgram keyterms with product aliases and add query repair before tool routing.",
            paper_action="pipeline_fix_then_rerun",
        )
    if downstream_failed:
        failures = " ".join(str(item).lower() for item in result.get("failures") or [])
        if "gate" in failures or "source" in failures:
            return _class_payload(
                primary="rag_grounding_miss",
                family="downstream",
                severity="high",
                root_cause="The knowledge answer did not retrieve or cite the expected evidence.",
                recommended_action="Add retrieval reranking and evidence-signature validation for this policy path.",
                paper_action="pipeline_fix_then_rerun",
            )
        return _class_payload(
            primary="inventory_tool_miss",
            family="downstream",
            severity="high",
            root_cause="The inventory lookup failed to map the transcript to the expected product/category.",
            recommended_action="Add alias expansion, category fallback, and product-review ranking for this inventory query.",
            paper_action="pipeline_fix_then_rerun",
        )
    if confidence is not None and confidence < 0.65:
        return _class_payload(
            primary="low_confidence_asr",
            family="asr",
            severity="medium",
            root_cause="Provider confidence is low even though other task checks are inconclusive.",
            recommended_action="Retake with better mic placement and compare against the accent sweep profiles.",
            paper_action="retake_or_provider_sweep",
        )
    if accepted_status in {"needs_retake", "unevaluated"}:
        return _class_payload(
            primary="accepted_group_gap",
            family="coverage",
            severity="medium",
            root_cause="This prompt/accent/noise group still has no passing accepted fixture.",
            recommended_action="Prioritize this group when recording the next audio batch.",
            paper_action="collect_more_data",
        )
    return _class_payload(
        primary="needs_manual_review",
        family="review",
        severity="medium",
        root_cause="The automated checks disagree or do not identify a dominant failure.",
        recommended_action="Inspect the transcript, expected slots, and retrieval evidence manually.",
        paper_action="manual_review",
    )


def _row(result: dict[str, Any], quality: dict[str, Any] | None, accepted_row: dict[str, Any] | None) -> dict[str, Any]:
    condition = _condition(result)
    missing, extra = _token_diff(result.get("referenceText"), result.get("transcriptText"))
    classification = _classify(result, quality, accepted_row)
    semantic = result.get("semanticTranscript") if isinstance(result.get("semanticTranscript"), dict) else {}
    return {
        "id": result.get("id"),
        "templateId": result.get("templateId"),
        "route": result.get("route"),
        "group": result.get("group"),
        "accent": condition["accent"],
        "noise": condition["noise"],
        "device": condition["device"],
        "bargeIn": condition["bargeIn"],
        "passed": bool(result.get("passed")),
        "semanticPassed": bool(semantic.get("passed")),
        "semanticLabel": semantic.get("label"),
        "semanticScore": semantic.get("score"),
        "semanticIntentScore": semantic.get("intentScore"),
        "semanticSlotScore": semantic.get("slotScore"),
        "asrPassed": result.get("asrPassed"),
        "downstreamPassed": result.get("downstreamPassed"),
        "accepted": bool((accepted_row or {}).get("accepted")),
        "reviewStatus": (accepted_row or {}).get("reviewStatus"),
        "qualityPriority": (quality or {}).get("priority"),
        "qualityScore": (quality or {}).get("qualityScore"),
        "failureModes": _failure_modes(result, quality),
        "wer": result.get("wer"),
        "rawWer": result.get("rawWer"),
        "entityRecall": result.get("entityRecall"),
        "rawEntityRecall": result.get("rawEntityRecall"),
        "confidence": result.get("confidence"),
        "transcriptionLatencyMs": result.get("transcriptionLatencyMs"),
        "tokenOverlap": _token_overlap(result.get("referenceText"), result.get("transcriptText")),
        "missingReferenceTerms": missing,
        "extraTranscriptTerms": extra,
        "referenceText": result.get("referenceText"),
        "transcriptText": result.get("transcriptText"),
        "rawTranscriptText": result.get("rawTranscriptText"),
        "failures": result.get("failures") or [],
        **classification,
    }


def _avg(values: list[Any]) -> float | None:
    clean = [_number(value) for value in values]
    numbers = [value for value in clean if value is not None]
    return round(sum(numbers) / len(numbers), 4) if numbers else None


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _aggregate(rows: list[dict[str, Any]], key: str, *, include_pass: bool = True) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not include_pass and row.get("semanticPassed"):
            continue
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    output: list[dict[str, Any]] = []
    for name, group in groups.items():
        total = len(group)
        failed = sum(1 for row in group if not row.get("semanticPassed"))
        severity = max((SEVERITY_SCORE.get(str(row.get("severity")), 0) for row in group), default=0)
        output.append(
            {
                "name": name,
                "total": total,
                "failed": failed,
                "passed": total - failed,
                "failureRate": _rate(failed, total),
                "avgWer": _avg([row.get("wer") for row in group]),
                "avgEntityRecall": _avg([row.get("entityRecall") for row in group]),
                "maxSeverity": next((label for label, score in SEVERITY_SCORE.items() if score == severity), "none"),
            }
        )
    return sorted(output, key=lambda row: (-int(row["failed"]), -float(row["failureRate"]), str(row["name"])))


def _condition_key(row: dict[str, Any]) -> str:
    return f"{row.get('accent')}|{row.get('noise')}|barge:{bool(row.get('bargeIn'))}"


def _condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_condition_key(row), []).append(row)
    output: list[dict[str, Any]] = []
    for name, group in groups.items():
        failures = [row for row in group if not row.get("semanticPassed")]
        top = _aggregate(failures, "primaryFailure", include_pass=False)[:1]
        output.append(
            {
                "condition": name,
                "total": len(group),
                "failed": len(failures),
                "failureRate": _rate(len(failures), len(group)),
                "topFailure": (top[0].get("name") if top else "pass"),
                "avgWer": _avg([row.get("wer") for row in group]),
                "avgEntityRecall": _avg([row.get("entityRecall") for row in group]),
            }
        )
    return sorted(output, key=lambda row: (-float(row["failureRate"]), -int(row["failed"]), str(row["condition"])))


def _action_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [row for row in rows if not row.get("semanticPassed")]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in failures:
        groups.setdefault(str(row.get("primaryFailure") or "needs_manual_review"), []).append(row)
    output: list[dict[str, Any]] = []
    for name, group in groups.items():
        first = group[0]
        severity = max(group, key=lambda row: SEVERITY_SCORE.get(str(row.get("severity")), 0)).get("severity")
        output.append(
            {
                "failure": name,
                "family": first.get("failureFamily"),
                "severity": severity,
                "affectedRecordings": len(group),
                "affectedTemplates": len({str(row.get("templateId")) for row in group}),
                "affectedConditions": len({_condition_key(row) for row in group}),
                "recommendedAction": first.get("recommendedAction"),
                "paperAction": first.get("paperAction"),
                "exampleRecordingId": first.get("id"),
                "exampleReference": first.get("referenceText"),
                "exampleTranscript": first.get("transcriptText"),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -SEVERITY_SCORE.get(str(row.get("severity")), 0),
            -int(row.get("affectedRecordings") or 0),
            str(row.get("failure")),
        ),
    )


def _coverage_gaps(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "selectionGroup": group.get("selectionGroup"),
            "templateId": group.get("templateId"),
            "referenceText": group.get("referenceText"),
            "accent": group.get("accent"),
            "noise": group.get("noise"),
            "bargeIn": group.get("bargeIn"),
            "recordings": group.get("recordings"),
            "status": group.get("status"),
        }
        for group in groups
        if not group.get("accepted")
    ]


def _summary(rows: list[dict[str, Any]], action_plan: list[dict[str, Any]], coverage_gaps: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    strict_passed = sum(1 for row in rows if row.get("passed"))
    semantic_passed = sum(1 for row in rows if row.get("semanticPassed"))
    failed_rows = [row for row in rows if not row.get("semanticPassed")]
    recovered_rows = [row for row in rows if row.get("semanticPassed") and not row.get("passed")]
    asr_only = [
        row
        for row in rows
        if row.get("asrPassed") is False and row.get("downstreamPassed") is True
    ]
    downstream = [row for row in rows if row.get("downstreamPassed") is False]
    language = [row for row in rows if row.get("primaryFailure") == "language_mismatch_not_accent"]
    prompt_drift = [row for row in rows if row.get("primaryFailure") == "prompt_drift_or_wrong_clip"]
    top = sorted(
        action_plan,
        key=lambda row: (-int(row.get("affectedRecordings") or 0), -SEVERITY_SCORE.get(str(row.get("severity")), 0), str(row.get("failure"))),
    )[0] if action_plan else None
    highest_priority = action_plan[0] if action_plan else None
    return {
        "totalRecordings": total,
        "strictPassed": strict_passed,
        "strictFailed": total - strict_passed,
        "semanticPassed": semantic_passed,
        "semanticRecovered": len(recovered_rows),
        "passed": semantic_passed,
        "failed": len(failed_rows),
        "passRate": _rate(semantic_passed, total),
        "strictPassRate": _rate(strict_passed, total),
        "failureBucketCount": len({row.get("primaryFailure") for row in failed_rows}),
        "asrOnlyFailures": len(asr_only),
        "downstreamFailures": len(downstream),
        "languageMismatches": len(language),
        "promptDriftOrWrongClip": len(prompt_drift),
        "coverageGaps": len(coverage_gaps),
        "topFailure": top.get("failure") if top else "none",
        "topRecommendedAction": top.get("recommendedAction") if top else "No blocking failures detected.",
        "highestPriorityFailure": highest_priority.get("failure") if highest_priority else "none",
        "highestPriorityAction": highest_priority.get("recommendedAction") if highest_priority else "No blocking failures detected.",
    }


def _recommendations(summary: dict[str, Any]) -> list[str]:
    recommendations = [
        "Report raw archive metrics separately from accepted benchmark-set metrics.",
        "Use this taxonomy table as the paper's error analysis: it explains whether failures come from data collection, language mismatch, ASR, downstream retrieval, or latency.",
    ]
    if (summary.get("languageMismatches") or 0) > 0:
        recommendations.append("Do not cite Spanish-language clips as accent robustness failures; split them into a multilingual or translation experiment.")
    if (summary.get("asrOnlyFailures") or 0) > 0:
        recommendations.append("Keep downstream task success separate from strict WER, because several clips answer correctly even when the ASR metric fails.")
    if (summary.get("coverageGaps") or 0) > 0:
        recommendations.append("Before the final paper, retake the accepted-set coverage gaps so each prompt/accent/noise group has one passing fixture.")
    return recommendations


def build_audio_error_analysis(*, save: bool = True) -> dict[str, Any]:
    latest = load_latest_audio_eval()
    if latest.get("found") is False:
        return {"found": False, "message": latest.get("message") or "No audio evaluation saved yet."}

    quality_by_id = _quality_by_id()
    accepted_by_id = _accepted_by_id()
    results = [result for result in latest.get("results") or [] if isinstance(result, dict) and not result.get("skipped")]
    rows = [_row(result, quality_by_id.get(str(result.get("id"))), accepted_by_id.get(str(result.get("id")))) for result in results]
    gaps = _coverage_gaps(_accepted_groups())
    actions = _action_plan(rows)
    summary = _summary(rows, actions, gaps)
    payload = {
        "found": True,
        "runId": datetime.now(timezone.utc).strftime("audio-errors-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suite": "real_audio_error_taxonomy",
        "sourceAudioRunId": latest.get("runId"),
        "summary": summary,
        "rows": rows,
        "actionPlan": actions,
        "byFailure": _aggregate(rows, "primaryFailure", include_pass=False),
        "byFamily": _aggregate(rows, "failureFamily", include_pass=True),
        "byAccent": _aggregate(rows, "accent", include_pass=True),
        "byTemplate": _aggregate(rows, "templateId", include_pass=True),
        "byRoute": _aggregate(rows, "route", include_pass=True),
        "conditionRisks": _condition_rows(rows),
        "acceptedCoverageGaps": gaps,
        "recommendations": _recommendations(summary),
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(LATEST_CSV_PATH.relative_to(ROOT_DIR)),
            "actionPlanCsv": str(ACTION_CSV_PATH.relative_to(ROOT_DIR)),
        },
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with LATEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _write_rows_csv(rows, LATEST_CSV_PATH)
        _write_actions_csv(actions, ACTION_CSV_PATH)
    return payload


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"))
    return value


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "id",
        "templateId",
        "route",
        "group",
        "accent",
        "noise",
        "device",
        "bargeIn",
        "passed",
        "semanticPassed",
        "semanticLabel",
        "semanticScore",
        "semanticIntentScore",
        "semanticSlotScore",
        "asrPassed",
        "downstreamPassed",
        "accepted",
        "reviewStatus",
        "qualityPriority",
        "qualityScore",
        "primaryFailure",
        "failureFamily",
        "severity",
        "rootCause",
        "recommendedAction",
        "paperAction",
        "failureModes",
        "wer",
        "rawWer",
        "entityRecall",
        "rawEntityRecall",
        "confidence",
        "transcriptionLatencyMs",
        "tokenOverlap",
        "missingReferenceTerms",
        "extraTranscriptTerms",
        "referenceText",
        "transcriptText",
        "rawTranscriptText",
        "failures",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_actions_csv(actions: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "failure",
        "family",
        "severity",
        "affectedRecordings",
        "affectedTemplates",
        "affectedConditions",
        "recommendedAction",
        "paperAction",
        "exampleRecordingId",
        "exampleReference",
        "exampleTranscript",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(actions)
