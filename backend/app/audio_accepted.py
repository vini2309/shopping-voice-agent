from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio_eval import load_latest_audio_eval, load_recorded_audio_cases
from .speech_eval import _rate


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "audio_eval"
LATEST_JSON_PATH = ARTIFACT_DIR / "accepted_set_latest.json"
LATEST_CSV_PATH = ARTIFACT_DIR / "accepted_set_latest.csv"


def load_latest_audio_accepted_set() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No accepted audio set saved yet."}
    with LATEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _parse_time(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _avg(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
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


def _metadata(recording: dict[str, Any]) -> dict[str, Any]:
    metadata = recording.get("recordingMetadata") if isinstance(recording.get("recordingMetadata"), dict) else {}
    condition = recording.get("condition") if isinstance(recording.get("condition"), dict) else {}
    speaker = recording.get("speaker") if isinstance(recording.get("speaker"), dict) else {}
    return {
        "speakerId": metadata.get("speakerId") or speaker.get("id") or "speaker-1",
        "accent": metadata.get("accent") or speaker.get("accent") or condition.get("accent") or "unknown",
        "noise": metadata.get("noise") or condition.get("noise") or "unknown",
        "device": metadata.get("device") or condition.get("device") or "unknown",
        "environment": metadata.get("environment") or condition.get("environment") or "unspecified",
        "bargeIn": bool(condition.get("bargeIn", False)),
    }


def _augmentation_type(recording: dict[str, Any]) -> str:
    augmentation = recording.get("augmentation") if isinstance(recording.get("augmentation"), dict) else {}
    return str(augmentation.get("type") or "none")


def _group_key(recording: dict[str, Any]) -> str:
    metadata = _metadata(recording)
    return "|".join(
        [
            str(recording.get("templateId") or recording.get("id") or "unknown"),
            str(metadata["accent"]),
            str(metadata["noise"]),
            f"barge:{metadata['bargeIn']}",
            f"aug:{_augmentation_type(recording)}",
        ]
    )


def _result_by_id() -> dict[str, dict[str, Any]]:
    latest = load_latest_audio_eval()
    results = latest.get("results") if isinstance(latest.get("results"), list) else []
    return {str(result.get("id")): result for result in results if isinstance(result, dict) and result.get("id")}


def _result_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [result for result in results if not result.get("skipped")]
    skipped = [result for result in results if result.get("skipped")]
    passed = sum(1 for result in evaluated if result.get("passed"))
    asr_passed = sum(1 for result in evaluated if result.get("asrPassed"))
    downstream_passed = sum(1 for result in evaluated if result.get("downstreamPassed"))
    transcription_latencies = [float(result.get("transcriptionLatencyMs") or 0.0) for result in evaluated]
    voice_latencies = [float(result.get("estimatedVoiceLatencyMs") or 0.0) for result in evaluated]
    costs = [float((result.get("cost") or {}).get("vapiStackCost") or 0.0) for result in evaluated]
    return {
        "total": len(results),
        "evaluated": len(evaluated),
        "skipped": len(skipped),
        "passed": passed,
        "failed": len(evaluated) - passed,
        "passRate": _rate(passed, len(evaluated)),
        "asrPassRate": _rate(asr_passed, len(evaluated)),
        "downstreamTaskSuccess": _rate(downstream_passed, len(evaluated)),
        "avgWer": _avg([result.get("wer") for result in evaluated]),
        "avgRawWer": _avg([result.get("rawWer") for result in evaluated]),
        "avgEntityRecall": _avg([result.get("entityRecall") for result in evaluated]),
        "avgRawEntityRecall": _avg([result.get("rawEntityRecall") for result in evaluated]),
        "latency": {
            "deepgramP50Ms": _percentile(transcription_latencies, 0.50),
            "deepgramP95Ms": _percentile(transcription_latencies, 0.95),
            "voiceP50Ms": _percentile(voice_latencies, 0.50),
            "voiceP95Ms": _percentile(voice_latencies, 0.95),
        },
        "cost": {
            "totalVapiStack": round(sum(costs), 6),
            "avgVapiStack": round(statistics.mean(costs), 6) if costs else 0.0,
            "per1000VapiStack": round(statistics.mean(costs) * 1000, 4) if costs else 0.0,
        },
    }


def _base_row(recording: dict[str, Any], result: dict[str, Any] | None, group_key: str) -> dict[str, Any]:
    metadata = _metadata(recording)
    return {
        "recordingId": recording.get("id"),
        "templateId": recording.get("templateId"),
        "referenceText": recording.get("referenceText"),
        "route": recording.get("route"),
        "group": recording.get("group"),
        "selectionGroup": group_key,
        "recordedAt": recording.get("recordedAt"),
        "audioUri": recording.get("audioUri"),
        "accent": metadata["accent"],
        "noise": metadata["noise"],
        "device": metadata["device"],
        "environment": metadata["environment"],
        "bargeIn": metadata["bargeIn"],
        "augmentationType": _augmentation_type(recording),
        "evaluated": bool(result and not result.get("skipped")),
        "passed": result.get("passed") if result else None,
        "asrPassed": result.get("asrPassed") if result else None,
        "downstreamPassed": result.get("downstreamPassed") if result else None,
        "wer": result.get("wer") if result else None,
        "rawWer": result.get("rawWer") if result else None,
        "entityRecall": result.get("entityRecall") if result else None,
        "rawEntityRecall": result.get("rawEntityRecall") if result else None,
        "transcriptionLatencyMs": result.get("transcriptionLatencyMs") if result else None,
        "confidence": result.get("confidence") if result else None,
        "transcriptText": result.get("transcriptText") if result else None,
        "rawTranscriptText": result.get("rawTranscriptText") if result else None,
        "failures": result.get("failures") if result else ["not evaluated"],
    }


def _accepted_result(row: dict[str, Any], eval_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not row.get("accepted"):
        return None
    return eval_by_id.get(str(row.get("recordingId")))


def build_audio_accepted_set(*, save: bool = True) -> dict[str, Any]:
    latest_audio = load_latest_audio_eval()
    raw_summary = latest_audio.get("summary") if isinstance(latest_audio.get("summary"), dict) else {}
    eval_by_id = _result_by_id()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for recording in load_recorded_audio_cases():
        key = _group_key(recording)
        grouped.setdefault(key, []).append(_base_row(recording, eval_by_id.get(str(recording.get("id"))), key))

    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        group.sort(key=lambda row: _parse_time(row.get("recordedAt")), reverse=True)
        passing = [row for row in group if row.get("passed") is True]
        accepted = passing[0] if passing else None
        newest = group[0] if group else None
        accepted_time = _parse_time(accepted.get("recordedAt")) if accepted else 0.0
        superseded_ids: list[str] = []
        for row in group:
            recording_id = str(row.get("recordingId") or "")
            row["accepted"] = bool(accepted and recording_id == str(accepted.get("recordingId")))
            row["paperAccepted"] = row["accepted"]
            row["supersededBy"] = None
            row["supersedes"] = []
            if row["accepted"]:
                row["reviewStatus"] = "accepted"
                row["selectionReason"] = "latest passing recording for this prompt/accent/noise/barge condition"
            elif accepted and _parse_time(row.get("recordedAt")) <= accepted_time:
                row["reviewStatus"] = "superseded"
                row["supersededBy"] = accepted.get("recordingId")
                row["selectionReason"] = "older attempt superseded by latest passing recording"
                superseded_ids.append(recording_id)
            elif accepted:
                row["reviewStatus"] = "rejected_retake"
                row["selectionReason"] = "newer retake did not pass; latest passing recording remains accepted"
            elif newest and recording_id == str(newest.get("recordingId")):
                row["reviewStatus"] = "needs_retake" if row.get("evaluated") else "unevaluated"
                row["selectionReason"] = "no passing recording exists for this group"
            else:
                row["reviewStatus"] = "archived_attempt"
                row["selectionReason"] = "older attempt retained in raw archive"
            rows.append(row)
        if accepted:
            accepted["supersedes"] = superseded_ids
        group_rows.append(
            {
                "selectionGroup": key,
                "templateId": (newest or {}).get("templateId"),
                "referenceText": (newest or {}).get("referenceText"),
                "accent": (newest or {}).get("accent"),
                "noise": (newest or {}).get("noise"),
                "bargeIn": (newest or {}).get("bargeIn"),
                "recordings": len(group),
                "acceptedRecordingId": accepted.get("recordingId") if accepted else None,
                "accepted": bool(accepted),
                "newestRecordingId": newest.get("recordingId") if newest else None,
                "supersededCount": len(superseded_ids),
                "status": "accepted" if accepted else "needs_retake",
            }
        )

    rows.sort(key=lambda row: (_parse_time(row.get("recordedAt")), str(row.get("recordingId") or "")), reverse=True)
    accepted_results = [
        result
        for result in (_accepted_result(row, eval_by_id) for row in rows)
        if isinstance(result, dict)
    ]
    accepted_summary = _result_summary(accepted_results)
    group_count = len(group_rows)
    accepted_count = sum(1 for row in rows if row.get("accepted"))
    superseded_count = sum(1 for row in rows if row.get("reviewStatus") == "superseded")
    rejected_retake_count = sum(1 for row in rows if row.get("reviewStatus") == "rejected_retake")
    needs_retake_count = sum(1 for row in group_rows if not row.get("accepted"))
    raw_pass_rate = raw_summary.get("passRate")
    accepted_pass_rate = accepted_summary.get("passRate")
    payload = {
        "found": True,
        "runId": datetime.now(timezone.utc).strftime("audio-accepted-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceAudioRunId": latest_audio.get("runId"),
        "policy": {
            "name": "latest_passing_per_prompt_accent_noise_barge",
            "grouping": ["templateId", "accent", "noise", "bargeIn", "augmentationType"],
            "rawArchivePreserved": True,
        },
        "summary": {
            "rawRecordings": int(raw_summary.get("total") or len(rows)),
            "rawEvaluated": int(raw_summary.get("evaluated") or 0),
            "rawPassed": int(raw_summary.get("passed") or 0),
            "rawPassRate": raw_pass_rate,
            "groupCount": group_count,
            "acceptedRecordings": accepted_count,
            "acceptedCoverageRate": _rate(accepted_count, group_count),
            "acceptedPassRate": accepted_pass_rate,
            "acceptedAvgWer": accepted_summary.get("avgWer"),
            "acceptedAvgRawWer": accepted_summary.get("avgRawWer"),
            "acceptedAvgEntityRecall": accepted_summary.get("avgEntityRecall"),
            "acceptedAvgRawEntityRecall": accepted_summary.get("avgRawEntityRecall"),
            "acceptedDeepgramP95Ms": (accepted_summary.get("latency") or {}).get("deepgramP95Ms"),
            "acceptedVoiceP95Ms": (accepted_summary.get("latency") or {}).get("voiceP95Ms"),
            "acceptedCostPer1000Turns": (accepted_summary.get("cost") or {}).get("per1000VapiStack"),
            "supersededRecordings": superseded_count,
            "rejectedRetakes": rejected_retake_count,
            "groupsNeedingRetake": needs_retake_count,
            "passRateLiftVsRaw": round(float(accepted_pass_rate or 0) - float(raw_pass_rate or 0), 4),
        },
        "acceptedSummary": accepted_summary,
        "acceptedResults": accepted_results,
        "groups": group_rows,
        "rows": rows,
        "acceptedRows": [row for row in rows if row.get("accepted")],
        "supersededRows": [row for row in rows if row.get("reviewStatus") == "superseded"],
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(LATEST_CSV_PATH.relative_to(ROOT_DIR)),
        },
        "recommendations": [
            "Use accepted-set metrics for paper headline real-audio claims and raw archive metrics for limitations.",
            "Record new clips for any group with no accepted passing recording.",
            "Keep superseded clips in the archive so retake improvement remains auditable.",
        ],
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with LATEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _save_csv(payload)
    return payload


def _save_csv(payload: dict[str, Any]) -> None:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    LATEST_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recordingId", "templateId", "referenceText", "route", "group", "selectionGroup",
        "recordedAt", "accent", "noise", "device", "environment", "bargeIn", "augmentationType",
        "accepted", "paperAccepted", "reviewStatus", "supersededBy", "supersedes", "selectionReason",
        "evaluated", "passed", "asrPassed", "downstreamPassed", "wer", "rawWer", "entityRecall",
        "rawEntityRecall", "transcriptionLatencyMs", "confidence", "transcriptText", "rawTranscriptText",
        "failures", "audioUri",
    ]
    with LATEST_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key) for key in fieldnames}
            output["supersedes"] = " | ".join(row.get("supersedes") or [])
            output["failures"] = " | ".join(row.get("failures") or [])
            writer.writerow(output)
