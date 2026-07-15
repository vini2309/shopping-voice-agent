from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio_eval import load_latest_audio_eval


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "audio_eval"
LATEST_JSON_PATH = ARTIFACT_DIR / "quality_latest.json"
LATEST_CSV_PATH = ARTIFACT_DIR / "quality_latest.csv"
RETAKE_JSON_PATH = ARTIFACT_DIR / "retake_queue_latest.json"
RETAKE_CSV_PATH = ARTIFACT_DIR / "retake_queue_latest.csv"

EMPTY_TRANSCRIPT_CHARS = 3
WRONG_PROMPT_WER = 0.80
WRONG_PROMPT_ENTITY_RECALL = 0.40
HIGH_WER = 0.35
LOW_ENTITY_RECALL = 0.66
LOW_CONFIDENCE = 0.65
TAIL_LATENCY_MS = 1000.0


def load_latest_audio_quality() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No audio quality gate saved yet."}
    with LATEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _condition(result: dict[str, Any]) -> dict[str, Any]:
    condition = result.get("condition") if isinstance(result.get("condition"), dict) else {}
    metadata = result.get("recordingMetadata") if isinstance(result.get("recordingMetadata"), dict) else {}
    speaker = result.get("speaker") if isinstance(result.get("speaker"), dict) else {}
    return {
        "accent": metadata.get("accent") or speaker.get("accent") or condition.get("accent") or "unknown",
        "noise": metadata.get("noise") or condition.get("noise") or "unknown",
        "device": metadata.get("device") or condition.get("device") or "unknown",
        "speakerId": metadata.get("speakerId") or speaker.get("id") or "speaker-1",
        "bargeIn": bool(condition.get("bargeIn")),
        "micDistanceCm": metadata.get("micDistanceCm") or condition.get("micDistanceCm"),
    }


def _failure_modes(result: dict[str, Any]) -> list[str]:
    transcript = _clean(result.get("transcriptText"))
    wer = float(result.get("wer") or 0.0)
    entity_recall = float(result.get("entityRecall") or 0.0)
    semantic = result.get("semanticTranscript") if isinstance(result.get("semanticTranscript"), dict) else {}
    semantic_passed = bool(semantic.get("passed"))
    confidence = result.get("confidence")
    latency = result.get("transcriptionLatencyMs")
    failures = " | ".join(str(item) for item in result.get("failures") or []).lower()
    modes: list[str] = []

    if result.get("skipped"):
        modes.append("skipped")
    if len(transcript) < EMPTY_TRANSCRIPT_CHARS:
        modes.append("empty_transcript")
    elif wer >= WRONG_PROMPT_WER and entity_recall <= WRONG_PROMPT_ENTITY_RECALL:
        modes.append("wrong_prompt_or_unintelligible")
    if semantic_passed and not result.get("passed"):
        modes.append(str(semantic.get("label") or "semantic_transcript_pass"))
    if not semantic_passed and ("wer" in failures or wer > HIGH_WER):
        modes.append("high_wer")
    if not semantic_passed and ("entity recall" in failures or entity_recall < LOW_ENTITY_RECALL):
        modes.append("entity_miss")
    if result.get("asrPassed") is False and not semantic_passed:
        modes.append("asr_failed")
    if result.get("downstreamPassed") is False:
        modes.append("downstream_failed")
    if isinstance(confidence, (int, float)) and float(confidence) < LOW_CONFIDENCE:
        modes.append("low_confidence")
    if isinstance(latency, (int, float)) and float(latency) > TAIL_LATENCY_MS:
        modes.append("tail_latency")
    if not modes and result.get("passed"):
        modes.append("pass")
    elif not modes:
        modes.append("needs_review")
    return list(dict.fromkeys(modes))


def _quality_score(result: dict[str, Any], modes: list[str]) -> int:
    score = 100
    penalties = {
        "empty_transcript": 65,
        "wrong_prompt_or_unintelligible": 55,
        "asr_failed": 35,
        "downstream_failed": 25,
        "entity_miss": 25,
        "high_wer": 20,
        "low_confidence": 15,
        "tail_latency": 8,
        "skipped": 80,
        "needs_review": 25,
    }
    for mode in modes:
        score -= penalties.get(mode, 0)
    return max(0, min(100, score))


def _priority(result: dict[str, Any], modes: list[str], score: int) -> str:
    semantic = result.get("semanticTranscript") if isinstance(result.get("semanticTranscript"), dict) else {}
    if (result.get("passed") or semantic.get("passed")) and score >= 80:
        return "keep"
    if "empty_transcript" in modes or "wrong_prompt_or_unintelligible" in modes:
        return "urgent_retake"
    if "downstream_failed" in modes or score < 55:
        return "high_retake"
    if score < 80:
        return "medium_retake"
    return "review"


def _instruction(result: dict[str, Any], modes: list[str]) -> str:
    reference = _clean(result.get("referenceText")) or "the prompt"
    if "empty_transcript" in modes:
        return f"Retake '{reference}' in a quiet room, wait one second before speaking, speak closer to the mic, and confirm the waveform is audible."
    if "wrong_prompt_or_unintelligible" in modes:
        return f"Retake exactly this prompt: '{reference}'. Avoid paraphrasing and avoid adding a different shopping request."
    if "entity_miss" in modes:
        return f"Retake '{reference}' and emphasize the product or policy words clearly."
    if "high_wer" in modes:
        return f"Retake '{reference}' at a slower pace with a short pause before and after the sentence."
    if "downstream_failed" in modes:
        return f"Retake '{reference}' and verify the transcript preserves the key item or policy terms."
    if "tail_latency" in modes:
        return f"Keep if transcript is correct; otherwise retake '{reference}' with shorter silence before and after speech."
    return f"Review transcript quality for '{reference}'."


def _row(result: dict[str, Any]) -> dict[str, Any]:
    modes = _failure_modes(result)
    score = _quality_score(result, modes)
    condition = _condition(result)
    priority = _priority(result, modes, score)
    semantic = result.get("semanticTranscript") if isinstance(result.get("semanticTranscript"), dict) else {}
    return {
        "id": result.get("id"),
        "templateId": result.get("templateId"),
        "route": result.get("route"),
        "group": result.get("group"),
        "priority": priority,
        "qualityScore": score,
        "usableForPaper": bool((result.get("passed") or semantic.get("passed")) and score >= 80),
        "passed": bool(result.get("passed")),
        "semanticPassed": bool(semantic.get("passed")),
        "semanticLabel": semantic.get("label"),
        "semanticScore": semantic.get("score"),
        "asrPassed": result.get("asrPassed"),
        "downstreamPassed": result.get("downstreamPassed"),
        "failureModes": modes,
        "failureCount": len([mode for mode in modes if mode != "pass"]),
        "referenceText": result.get("referenceText"),
        "transcriptText": result.get("transcriptText"),
        "canonicalTranscriptText": result.get("canonicalTranscriptText"),
        "wer": result.get("wer"),
        "entityRecall": result.get("entityRecall"),
        "confidence": result.get("confidence"),
        "transcriptionLatencyMs": result.get("transcriptionLatencyMs"),
        "durationMs": result.get("durationMs"),
        "accent": condition["accent"],
        "noise": condition["noise"],
        "device": condition["device"],
        "speakerId": condition["speakerId"],
        "bargeIn": condition["bargeIn"],
        "audioUri": result.get("audioUri"),
        "retakeInstruction": _instruction(result, modes),
        "failures": result.get("failures") or [],
    }


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for mode in row.get("failureModes") or []:
            if mode == "pass":
                continue
            counts[str(mode)] = counts.get(str(mode), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _retake_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority_order = {"urgent_retake": 0, "high_retake": 1, "medium_retake": 2, "review": 3, "keep": 4}
    return sorted(
        [row for row in rows if row.get("priority") != "keep"],
        key=lambda row: (priority_order.get(str(row.get("priority")), 9), row.get("qualityScore") or 100, str(row.get("templateId") or "")),
    )


def _summary(rows: list[dict[str, Any]], retakes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    usable = sum(1 for row in rows if row.get("usableForPaper"))
    empty = sum(1 for row in rows if "empty_transcript" in (row.get("failureModes") or []))
    wrong = sum(1 for row in rows if "wrong_prompt_or_unintelligible" in (row.get("failureModes") or []))
    return {
        "totalRecordings": total,
        "usableForPaper": usable,
        "usableRate": round(usable / total, 4) if total else 0.0,
        "retakeNeeded": len(retakes),
        "retakeRate": round(len(retakes) / total, 4) if total else 0.0,
        "urgentRetakes": sum(1 for row in retakes if row.get("priority") == "urgent_retake"),
        "highRetakes": sum(1 for row in retakes if row.get("priority") == "high_retake"),
        "emptyTranscripts": empty,
        "wrongPromptOrUnintelligible": wrong,
        "avgQualityScore": round(sum(float(row.get("qualityScore") or 0) for row in rows) / total, 2) if total else 0.0,
        "byPriority": _counts(rows, "priority"),
        "byFailureMode": _mode_counts(rows),
        "byAccent": _counts(rows, "accent"),
        "byTemplate": _counts(rows, "templateId"),
    }


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "templateId",
        "route",
        "group",
        "priority",
        "qualityScore",
        "usableForPaper",
        "passed",
        "semanticPassed",
        "semanticLabel",
        "semanticScore",
        "asrPassed",
        "downstreamPassed",
        "failureModes",
        "referenceText",
        "transcriptText",
        "canonicalTranscriptText",
        "wer",
        "entityRecall",
        "confidence",
        "transcriptionLatencyMs",
        "durationMs",
        "accent",
        "noise",
        "device",
        "speakerId",
        "bargeIn",
        "audioUri",
        "retakeInstruction",
        "failures",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "failureModes": " | ".join(row.get("failureModes") or []),
                    "failures": " | ".join(row.get("failures") or []),
                }
            )


def run_audio_quality_gate(*, save: bool = True) -> dict[str, Any]:
    latest = load_latest_audio_eval()
    if latest.get("found") is False:
        return {"found": False, "message": latest.get("message") or "No audio evaluation saved yet."}
    results = [result for result in latest.get("results") or [] if isinstance(result, dict) and not result.get("skipped")]
    rows = [_row(result) for result in results]
    retakes = _retake_rows(rows)
    payload = {
        "found": True,
        "runId": datetime.now(timezone.utc).strftime("audio-qa-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceAudioRunId": latest.get("runId"),
        "summary": _summary(rows, retakes),
        "rows": rows,
        "retakeQueue": retakes,
        "thresholds": {
            "emptyTranscriptChars": EMPTY_TRANSCRIPT_CHARS,
            "wrongPromptWer": WRONG_PROMPT_WER,
            "wrongPromptEntityRecall": WRONG_PROMPT_ENTITY_RECALL,
            "highWer": HIGH_WER,
            "lowEntityRecall": LOW_ENTITY_RECALL,
            "lowConfidence": LOW_CONFIDENCE,
            "tailLatencyMs": TAIL_LATENCY_MS,
        },
        "recommendations": [
            "Record one prompt per clip and read the reference text exactly for benchmark fixtures.",
            "Leave about one second of room tone before speech and stop recording one second after speech.",
            "Prefer a headset or close browser mic for accent/noise stress recordings before adding harder conditions.",
            "Use the retake queue before citing real-audio pass-rate or WER claims.",
        ],
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(LATEST_CSV_PATH.relative_to(ROOT_DIR)),
            "retakeQueueJson": str(RETAKE_JSON_PATH.relative_to(ROOT_DIR)),
            "retakeQueueCsv": str(RETAKE_CSV_PATH.relative_to(ROOT_DIR)),
        },
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with LATEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        with RETAKE_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump({"sourceAudioRunId": latest.get("runId"), "retakeQueue": retakes}, handle, indent=2)
        _write_rows_csv(rows, LATEST_CSV_PATH)
        _write_rows_csv(retakes, RETAKE_CSV_PATH)
    return payload
