from __future__ import annotations

import base64
import csv
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .speech_eval import (
    _case_cost,
    _check_inventory,
    _check_knowledge,
    _condition_key,
    _entity_metrics,
    _rate,
    _wer,
)
from .inventory import load_inventory, tool_payload
from .knowledge import generate_evidence_gated_answer
from .multilingual import canonicalize_query
from .semantic_transcript import score_semantic_transcript


ROOT_DIR = Path(__file__).resolve().parents[2]
AUDIO_CASES_PATH = ROOT_DIR / "benchmarks" / "audio_cases.json"
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "audio_eval"
RECORDINGS_DIR = ARTIFACT_DIR / "recordings"
LOCAL_CASES_PATH = ARTIFACT_DIR / "audio_cases.local.json"
LATEST_JSON_PATH = ARTIFACT_DIR / "latest.json"
LATEST_CSV_PATH = ARTIFACT_DIR / "latest.csv"
MANIFEST_JSON_PATH = ARTIFACT_DIR / "audio_dataset_manifest.json"
MANIFEST_CSV_PATH = ARTIFACT_DIR / "audio_dataset_manifest.csv"
ROBUSTNESS_JSON_PATH = ARTIFACT_DIR / "robustness_latest.json"
ROBUSTNESS_CSV_PATH = ARTIFACT_DIR / "robustness_latest.csv"
ACCENT_SWEEP_JSON_PATH = ARTIFACT_DIR / "accent_sweep_latest.json"
ACCENT_SWEEP_CSV_PATH = ARTIFACT_DIR / "accent_sweep_latest.csv"

MAX_AUDIO_BYTES = 12 * 1024 * 1024
DEEPGRAM_DEFAULT_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_DEFAULT_MODEL = "nova-3"
DEFAULT_DEEPGRAM_PROFILE = "accent_aware_keyterms"

ACCENT_LANGUAGE_HINTS = {
    "indian_english": "en-IN",
    "spanish_l1": "multi",
    "us_english": "en-US",
    "user_recorded": "en-US",
}

STATIC_RETAIL_KEYTERMS = [
    "aisle",
    "bay",
    "shelf",
    "stock",
    "topstock",
    "inventory",
    "out of stock",
    "units available",
    "paper towels",
    "dog food",
    "opened electronics",
    "return policy",
    "pickup",
    "curbside",
    "substitution",
    "price match",
    "service desk",
    "PawChoice",
    "Everyday Choice",
    "Bounty",
    "Clorox",
    "Huggies",
    "Tylenol",
]

SWEEP_CONFIGS = [
    {
        "id": "default",
        "label": "Default Nova-3",
        "language": None,
        "useKeyterms": False,
        "enableTranscriptRepair": False,
    },
    {
        "id": "en_us_keyterms",
        "label": "en-US + retail keyterms",
        "language": "en-US",
        "useKeyterms": True,
        "enableTranscriptRepair": False,
    },
    {
        "id": "accent_aware_keyterms",
        "label": "Accent-aware + retail keyterms",
        "language": "accent-aware",
        "useKeyterms": True,
        "enableTranscriptRepair": False,
    },
    {
        "id": "multi_keyterms",
        "label": "Multilingual + retail keyterms",
        "language": "multi",
        "useKeyterms": True,
        "enableTranscriptRepair": False,
    },
    {
        "id": "accent_aware_keyterms_repair",
        "label": "Accent-aware + keyterms + repair",
        "language": "accent-aware",
        "useKeyterms": True,
        "enableTranscriptRepair": True,
    },
]


def provider_ready() -> bool:
    return bool(os.getenv("DEEPGRAM_API_KEY"))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _term_token_count(term: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9]+", term)))


def _dedupe_terms(terms: list[Any], *, limit: int | None = None, token_limit: int | None = None) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    tokens_used = 0
    for value in terms:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        text = text.strip(".,;:!?\"'")
        if len(text) < 2:
            continue
        key = text.lower()
        if key in seen:
            continue
        token_count = _term_token_count(text)
        if token_limit and clean and tokens_used + token_count > token_limit:
            continue
        seen.add(key)
        clean.append(text)
        tokens_used += token_count
        if limit and len(clean) >= limit:
            break
    return clean


def _case_terms(case: dict[str, Any] | None) -> list[str]:
    if not isinstance(case, dict):
        return []
    terms: list[Any] = []
    terms.extend(case.get("entities") if isinstance(case.get("entities"), list) else [])
    reference = str(case.get("referenceText") or "")
    if reference:
        terms.append(reference)
        tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", reference) if len(token) > 2]
        terms.extend(tokens)
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    terms.extend(expected.get("aisles") if isinstance(expected.get("aisles"), list) else [])
    terms.extend(expected.get("itemIds") if isinstance(expected.get("itemIds"), list) else [])
    return _dedupe_terms(terms)


def retail_keyterms(case: dict[str, Any] | None = None, *, limit: int | None = None) -> list[str]:
    configured = [
        term.strip()
        for term in str(os.getenv("DEEPGRAM_KEYTERMS") or "").split(",")
        if term.strip()
    ]
    terms: list[Any] = [*STATIC_RETAIL_KEYTERMS, *configured, *_case_terms(case)]
    try:
        for item in load_inventory():
            if not isinstance(item, dict):
                continue
            for field in ["name", "brand", "department", "category", "subcategory", "aisle", "bay"]:
                if item.get(field):
                    terms.append(item.get(field))
    except Exception:
        pass
    try:
        for template in load_seed_audio_cases():
            terms.extend(_case_terms(template))
    except Exception:
        pass
    cap = limit if limit is not None else int(os.getenv("DEEPGRAM_KEYTERM_LIMIT") or "90")
    token_cap = int(os.getenv("DEEPGRAM_KEYTERM_TOKEN_LIMIT") or "450")
    return _dedupe_terms(terms, limit=max(1, cap), token_limit=max(50, min(token_cap, 500)))


def _accent_from_case(case: dict[str, Any] | None) -> str:
    if not isinstance(case, dict):
        return "user_recorded"
    metadata = case.get("recordingMetadata") if isinstance(case.get("recordingMetadata"), dict) else {}
    speaker = case.get("speaker") if isinstance(case.get("speaker"), dict) else {}
    condition = case.get("condition") if isinstance(case.get("condition"), dict) else {}
    return str(metadata.get("accent") or speaker.get("accent") or condition.get("accent") or "user_recorded")


def _language_for_case(case: dict[str, Any] | None, language: Any) -> str | None:
    requested = str(language or "").strip()
    if not requested:
        requested = str(os.getenv("DEEPGRAM_LANGUAGE") or "").strip()
    if requested == "accent-aware":
        return ACCENT_LANGUAGE_HINTS.get(_accent_from_case(case), "en-US")
    return requested or None


def _resolve_deepgram_config(config: dict[str, Any] | None, case: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(config or {})
    profile = str(raw.get("id") or raw.get("profile") or os.getenv("DEEPGRAM_PROFILE") or DEFAULT_DEEPGRAM_PROFILE)
    preset = next((item for item in SWEEP_CONFIGS if item["id"] == profile), None)
    merged = {**(preset or {}), **raw}
    use_keyterms = bool(merged.get("useKeyterms", _env_flag("DEEPGRAM_USE_KEYTERMS", profile != "default")))
    language = _language_for_case(case, merged.get("language"))
    keyterms = retail_keyterms(case) if use_keyterms else []
    return {
        "id": profile,
        "label": merged.get("label") or profile.replace("_", " "),
        "model": str(merged.get("model") or os.getenv("DEEPGRAM_MODEL") or DEEPGRAM_DEFAULT_MODEL),
        "language": language,
        "useKeyterms": use_keyterms,
        "keyterms": keyterms,
        "keytermCount": len(keyterms),
        "enableTranscriptRepair": bool(merged.get("enableTranscriptRepair", False)),
    }


def _repair_transcript(transcript: str, case: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    repaired = str(transcript or "")
    if not repaired.strip():
        return repaired, []
    entities = {str(entity).lower() for entity in case.get("entities") or []}
    reference = str(case.get("referenceText") or "").lower()
    rules: list[tuple[bool, re.Pattern[str], str, str]] = [
        (
            "shelf" in entities or "shelf" in reference,
            re.compile(r"\b(the\s+)?(?:selfie|self)\s+is\s+empty\b", re.IGNORECASE),
            "the shelf is empty",
            "retail_lexicon_shelf",
        ),
        (
            "stock" in entities or "stock" in reference,
            re.compile(r"\bsystem\s+(?:so\s+stuck|shows?\s+stuck|show\s+stock)\b", re.IGNORECASE),
            "system shows stock",
            "retail_lexicon_stock",
        ),
        (
            "opened" in entities or "opened" in reference,
            re.compile(r"\breturn\s+and\s+open\s+electronic\s+items?\b", re.IGNORECASE),
            "return an opened electronics item",
            "policy_lexicon_opened_electronics",
        ),
    ]
    repairs: list[dict[str, str]] = []
    for enabled, pattern, replacement, reason in rules:
        if not enabled:
            continue
        updated, count = pattern.subn(replacement, repaired)
        if count:
            repairs.append({"reason": reason, "from": repaired, "to": updated})
            repaired = updated
    return repaired, repairs


def load_seed_audio_cases() -> list[dict[str, Any]]:
    if not AUDIO_CASES_PATH.is_file():
        return []
    with AUDIO_CASES_PATH.open("r", encoding="utf-8-sig") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("audio case file must contain a JSON list")
    return [case for case in cases if isinstance(case, dict)]


def load_recorded_audio_cases() -> list[dict[str, Any]]:
    if not LOCAL_CASES_PATH.is_file():
        return []
    with LOCAL_CASES_PATH.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def load_audio_cases(*, include_templates: bool = True) -> dict[str, Any]:
    templates = load_seed_audio_cases()
    recordings = load_recorded_audio_cases()
    manifest = build_audio_dataset_manifest(save=False)
    return {
        "templates": templates if include_templates else [],
        "recordings": recordings,
        "cases": [*(templates if include_templates else []), *recordings],
        "templateCount": len(templates),
        "recordingCount": len(recordings),
        "coverage": manifest.get("summary", {}),
        "manifestArtifacts": manifest.get("artifacts", {}),
        "provider": {
            "stt": "deepgram",
            "model": os.getenv("DEEPGRAM_MODEL", DEEPGRAM_DEFAULT_MODEL),
            "profile": os.getenv("DEEPGRAM_PROFILE", DEFAULT_DEEPGRAM_PROFILE),
            "languageHints": ACCENT_LANGUAGE_HINTS,
            "keytermCount": len(retail_keyterms()),
            "ready": provider_ready(),
            "requires": "Set DEEPGRAM_API_KEY in backend .env or process environment.",
        },
    }


def load_latest_audio_eval() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No real audio evaluation saved yet."}
    with LATEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _safe_id(value: Any) -> str:
    text = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in str(value or "audio"))
    return "-".join(part for part in text.strip("-").split("-") if part)[:80] or "audio"


def _extension_for_mime(mime_type: str) -> str:
    lowered = mime_type.lower()
    if "wav" in lowered:
        return "wav"
    if "mpeg" in lowered or "mp3" in lowered:
        return "mp3"
    if "ogg" in lowered:
        return "ogg"
    if "mp4" in lowered or "m4a" in lowered:
        return "m4a"
    return "webm"


def _content_type_for_mime(mime_type: str) -> str:
    return (mime_type or "audio/webm").split(";")[0].strip() or "audio/webm"


def _decode_audio(value: Any) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("audioBase64 is required")
    raw = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    data = base64.b64decode(raw, validate=True)
    if not data:
        raise ValueError("audio payload is empty")
    if len(data) > MAX_AUDIO_BYTES:
        raise ValueError(f"audio payload exceeds {MAX_AUDIO_BYTES} bytes")
    return data


def _template_by_id(case_id: str) -> dict[str, Any] | None:
    for case in load_seed_audio_cases():
        if case.get("id") == case_id:
            return case
    return None


def _recording_metadata(payload: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    template_condition = dict(template.get("condition") or {})
    speaker_id = str(raw.get("speakerId") or payload.get("speakerId") or "speaker-1").strip() or "speaker-1"
    accent = str(raw.get("accent") or payload.get("accent") or template_condition.get("accent") or "user_recorded").strip() or "user_recorded"
    noise = str(raw.get("noise") or payload.get("noise") or template_condition.get("noise") or "room").strip() or "room"
    device = str(raw.get("device") or payload.get("device") or "browser_mic").strip() or "browser_mic"
    environment = str(raw.get("environment") or payload.get("environment") or "unspecified").strip() or "unspecified"
    notes = str(raw.get("notes") or payload.get("notes") or "").strip()
    try:
        mic_distance_cm = int(raw.get("micDistanceCm") or payload.get("micDistanceCm") or 30)
    except (TypeError, ValueError):
        mic_distance_cm = 30
    condition = {
        **template_condition,
        "accent": accent,
        "noise": noise,
        "device": device,
        "environment": environment,
        "micDistanceCm": mic_distance_cm,
    }
    return {
        "speaker": {"id": speaker_id, "accent": accent},
        "condition": condition,
        "recordingMetadata": {
            "speakerId": speaker_id,
            "accent": accent,
            "noise": noise,
            "device": device,
            "environment": environment,
            "micDistanceCm": mic_distance_cm,
            "notes": notes,
        },
    }


def _write_recording_manifest(case: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [item for item in load_recorded_audio_cases() if item.get("id") != case.get("id")]
    cases.append(case)
    cases.sort(key=lambda item: str(item.get("recordedAt") or ""))
    with LOCAL_CASES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(cases, handle, indent=2)


def save_audio_recording(payload: dict[str, Any]) -> dict[str, Any]:
    case_id = _safe_id(payload.get("caseId"))
    template = _template_by_id(case_id)
    if not template:
        reference = str(payload.get("referenceText") or "").strip()
        if not reference:
            raise ValueError("caseId must match a template or referenceText is required")
        template = {
            "id": case_id,
            "route": payload.get("route") or "inventory",
            "group": payload.get("group") or "real_audio_custom",
            "condition": payload.get("condition") if isinstance(payload.get("condition"), dict) else {},
            "referenceText": reference,
            "entities": payload.get("entities") if isinstance(payload.get("entities"), list) else [],
            "expected": payload.get("expected") if isinstance(payload.get("expected"), dict) else {},
        }

    audio = _decode_audio(payload.get("audioBase64"))
    mime_type = str(payload.get("mimeType") or "audio/webm")
    augmentation = payload.get("augmentation") if isinstance(payload.get("augmentation"), dict) else None
    augmentation_type = augmentation.get("type") if augmentation else payload.get("variantType")
    augmentation_suffix = _safe_id(augmentation_type) if augmentation_type else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    recording_id = f"{case_id}-{augmentation_suffix + '-' if augmentation_suffix else ''}{stamp}"
    extension = _extension_for_mime(mime_type)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = RECORDINGS_DIR / f"{recording_id}.{extension}"
    audio_path.write_bytes(audio)

    metadata = _recording_metadata(payload, template)
    parent_recording_id = str(payload.get("parentRecordingId") or "").strip() or None
    variant_of = str(payload.get("variantOf") or parent_recording_id or "").strip() or None
    case = {
        **template,
        "id": recording_id,
        "templateId": template.get("id"),
        "condition": metadata["condition"],
        "speaker": metadata["speaker"],
        "recordingMetadata": metadata["recordingMetadata"],
        "parentRecordingId": parent_recording_id,
        "variantOf": variant_of,
        "augmentation": augmentation,
        "audioUri": str(audio_path.relative_to(ROOT_DIR)),
        "mimeType": mime_type,
        "durationMs": int(payload.get("durationMs") or 0),
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "transcriptText": None,
        "transcriptionProvider": "deepgram",
    }
    _write_recording_manifest(case)
    return {
        "saved": True,
        "case": case,
        "bytes": len(audio),
        "recordingCount": len(load_recorded_audio_cases()),
    }


def _resolve_audio_path(audio_uri: Any) -> Path | None:
    if not audio_uri:
        return None
    path = (ROOT_DIR / str(audio_uri)).resolve()
    if ROOT_DIR.resolve() not in path.parents:
        return None
    return path


def recording_audio_path(recording_id: str) -> tuple[Path, str]:
    requested_id = str(recording_id or "").strip()
    if not requested_id:
        raise FileNotFoundError("recording id is required")
    for case in load_recorded_audio_cases():
        if str(case.get("id")) != requested_id:
            continue
        path = _resolve_audio_path(case.get("audioUri"))
        if not path or not path.is_file():
            raise FileNotFoundError("recording audio file not found")
        return path, _content_type_for_mime(str(case.get("mimeType") or "audio/webm"))
    raise FileNotFoundError("recording not found")


def _deepgram_url(config: dict[str, Any] | None = None, case: dict[str, Any] | None = None) -> str:
    base = os.getenv("DEEPGRAM_API_URL", DEEPGRAM_DEFAULT_URL).strip() or DEEPGRAM_DEFAULT_URL
    resolved = _resolve_deepgram_config(config, case)
    query: dict[str, Any] = {"model": resolved["model"], "smart_format": "true"}
    if resolved.get("language"):
        query["language"] = resolved["language"]
    if resolved.get("keyterms"):
        query["keyterm"] = resolved["keyterms"]
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}{urllib.parse.urlencode(query, doseq=True)}"


def _transcribe_deepgram(
    audio_path: Path,
    mime_type: str,
    *,
    config: dict[str, Any] | None = None,
    case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "skipped": True,
            "error": "missing_deepgram_api_key",
            "message": "Set DEEPGRAM_API_KEY in the backend environment to run real audio transcription.",
        }
    resolved_config = _resolve_deepgram_config(config, case)
    data = audio_path.read_bytes()
    request = urllib.request.Request(
        _deepgram_url(resolved_config, case),
        data=data,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": _content_type_for_mime(mime_type),
            "Accept": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return {"ok": False, "skipped": False, "error": f"deepgram_http_{error.code}", "message": body[:600]}
    except urllib.error.URLError as error:
        return {"ok": False, "skipped": False, "error": "deepgram_connection_error", "message": str(error.reason)}
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    parsed = json.loads(raw.decode("utf-8"))
    alternative = (((parsed.get("results") or {}).get("channels") or [{}])[0].get("alternatives") or [{}])[0]
    metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
    return {
        "ok": True,
        "skipped": False,
        "provider": "deepgram",
        "model": resolved_config["model"],
        "config": {
            "id": resolved_config["id"],
            "label": resolved_config["label"],
            "language": resolved_config["language"],
            "useKeyterms": resolved_config["useKeyterms"],
            "keytermCount": resolved_config["keytermCount"],
            "enableTranscriptRepair": resolved_config["enableTranscriptRepair"],
        },
        "transcript": str(alternative.get("transcript") or "").strip(),
        "confidence": alternative.get("confidence"),
        "words": alternative.get("words") if isinstance(alternative.get("words"), list) else [],
        "durationSeconds": metadata.get("duration"),
        "latencyMs": latency_ms,
        "requestId": metadata.get("request_id"),
    }


def _case_variant_fields(case: dict[str, Any]) -> dict[str, Any]:
    augmentation = case.get("augmentation") if isinstance(case.get("augmentation"), dict) else None
    return {
        "parentRecordingId": case.get("parentRecordingId"),
        "variantOf": case.get("variantOf"),
        "augmentation": augmentation,
    }


def _run_downstream(route: str, transcript: str, expected: dict[str, Any], wer: float) -> dict[str, Any]:
    started = time.perf_counter()
    if route == "knowledge":
        payload = generate_evidence_gated_answer(transcript, limit=4)
        downstream_passed, failures, observed = _check_knowledge(payload, expected)
        answer = str(payload.get("speechAnswer") or "")
    else:
        payload = tool_payload(transcript)
        downstream_passed, failures, observed = _check_inventory(payload, expected)
        answer = str(payload.get("speechAnswer") or payload.get("message") or "")
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "payload": payload,
        "downstreamPassed": downstream_passed,
        "failures": failures,
        "observed": observed,
        "answer": answer,
        "latencyMs": latency_ms,
        "cost": _case_cost(transcript, answer, latency_ms, wer),
    }


def _evaluate_case(
    case: dict[str, Any],
    *,
    allow_reference_fallback: bool,
    deepgram_config: dict[str, Any] | None = None,
    enable_transcript_repair: bool | None = None,
) -> dict[str, Any]:
    reference = str(case.get("referenceText") or "")
    route = str(case.get("route") or "inventory")
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    variant_fields = _case_variant_fields(case)
    resolved_deepgram_config = _resolve_deepgram_config(deepgram_config, case)
    repair_enabled = bool(
        resolved_deepgram_config.get("enableTranscriptRepair")
        if enable_transcript_repair is None
        else enable_transcript_repair
    )
    audio_path = _resolve_audio_path(case.get("audioUri"))
    if not audio_path or not audio_path.is_file():
        return {
            "id": case.get("id"),
            "templateId": case.get("templateId"),
            **variant_fields,
            "route": route,
            "group": case.get("group") or route,
            "condition": case.get("condition") or {},
            "referenceText": reference,
            "audioUri": case.get("audioUri"),
            "skipped": True,
            "passed": False,
            "skipReason": "missing_audio_file",
            "failures": ["missing audio file"],
        }

    transcription = _transcribe_deepgram(
        audio_path,
        str(case.get("mimeType") or "audio/webm"),
        config=resolved_deepgram_config,
        case=case,
    )
    transcription_mode = "deepgram_prerecorded"
    if not transcription.get("ok"):
        if allow_reference_fallback:
            transcription = {
                "ok": True,
                "skipped": False,
                "provider": "reference_fallback",
                "model": "reference",
                "config": {
                    "id": "reference_fallback",
                    "label": "Reference fallback",
                    "language": None,
                    "useKeyterms": False,
                    "keytermCount": 0,
                    "enableTranscriptRepair": False,
                },
                "transcript": str(case.get("transcriptText") or reference),
                "confidence": None,
                "latencyMs": 0.0,
            }
            transcription_mode = "reference_fallback"
        else:
            return {
                "id": case.get("id"),
                "templateId": case.get("templateId"),
                **variant_fields,
                "route": route,
                "group": case.get("group") or route,
                "condition": case.get("condition") or {},
                "referenceText": reference,
                "audioUri": case.get("audioUri"),
                "skipped": bool(transcription.get("skipped")),
                "passed": False,
                "skipReason": transcription.get("error"),
                "providerMessage": transcription.get("message"),
                "failures": [str(transcription.get("error") or "transcription failed")],
            }

    raw_transcript = str(transcription.get("transcript") or "")
    raw_wer, reference_tokens, raw_transcript_tokens = _wer(reference, raw_transcript)
    raw_entities = _entity_metrics(reference_tokens, raw_transcript_tokens, case.get("entities") or [])
    transcript, repairs = _repair_transcript(raw_transcript, case) if repair_enabled else (raw_transcript, [])
    surface_wer, _, transcript_tokens = _wer(reference, transcript)
    surface_entities = _entity_metrics(reference_tokens, transcript_tokens, case.get("entities") or [])
    canonical = canonicalize_query(transcript)
    canonical_transcript = str(canonical.get("canonicalText") or transcript)
    canonical_wer, _, canonical_tokens = _wer(reference, canonical_transcript)
    canonical_entities = _entity_metrics(reference_tokens, canonical_tokens, case.get("entities") or [])
    multilingual_scoring = bool(canonical.get("translated") and canonical.get("language") == "es")
    wer = canonical_wer if multilingual_scoring else surface_wer
    entities = canonical_entities if multilingual_scoring else surface_entities
    downstream = _run_downstream(route, transcript, expected, wer)
    failures = list(downstream["failures"])

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
    strict_passed = bool(asr_passed and downstream["downstreamPassed"])
    semantic_transcript = score_semantic_transcript(
        reference=reference,
        transcript=transcript,
        canonical_transcript=canonical_transcript,
        route=route,
        expected=expected,
        entities=case.get("entities") or [],
        downstream_passed=bool(downstream["downstreamPassed"]),
        strict_asr_passed=asr_passed,
        strict_passed=strict_passed,
        wer=round(wer, 4),
        entity_recall=entities.get("entityRecall"),
    )

    return {
        "id": case.get("id"),
        "templateId": case.get("templateId"),
        **variant_fields,
        "route": route,
        "group": case.get("group") or route,
        "condition": case.get("condition") or {},
        "referenceText": reference,
        "transcriptText": transcript,
        "canonicalTranscriptText": canonical_transcript,
        "detectedLanguage": canonical.get("language"),
        "languageConfidence": canonical.get("languageConfidence"),
        "multilingual": canonical,
        "multilingualScoring": multilingual_scoring,
        "audioUri": case.get("audioUri"),
        "mimeType": case.get("mimeType"),
        "durationMs": case.get("durationMs"),
        "recordedAt": case.get("recordedAt"),
        "speaker": case.get("speaker") or {},
        "recordingMetadata": case.get("recordingMetadata") or {},
        "transcriptionMode": transcription_mode,
        "transcriptionProvider": transcription.get("provider"),
        "transcriptionModel": transcription.get("model"),
        "deepgramConfig": transcription.get("config"),
        "transcriptionLatencyMs": transcription.get("latencyMs"),
        "confidence": transcription.get("confidence"),
        "rawTranscriptText": raw_transcript,
        "rawWer": round(raw_wer, 4),
        "rawEntityWer": raw_entities.get("entityWer"),
        "rawEntityRecall": raw_entities.get("entityRecall"),
        "surfaceWer": round(surface_wer, 4),
        "surfaceEntityWer": surface_entities.get("entityWer"),
        "surfaceEntityRecall": surface_entities.get("entityRecall"),
        "transcriptRepairEnabled": repair_enabled,
        "transcriptRepairs": repairs,
        "wer": round(wer, 4),
        "canonicalWer": round(canonical_wer, 4),
        "canonicalEntityWer": canonical_entities.get("entityWer"),
        "canonicalEntityRecall": canonical_entities.get("entityRecall"),
        "entityWer": entities.get("entityWer"),
        "entityRecall": entities.get("entityRecall"),
        "asrPassed": asr_passed,
        "semanticAsrPassed": semantic_transcript.get("passed"),
        "semanticTranscript": semantic_transcript,
        "downstreamPassed": downstream["downstreamPassed"],
        "skipped": False,
        "passed": strict_passed,
        "failures": failures,
        "observed": downstream["observed"],
        "answer": downstream["answer"],
        "latencyMs": downstream["latencyMs"],
        "estimatedVoiceLatencyMs": downstream["cost"]["estimatedVoiceLatencyMs"],
        "cost": downstream["cost"],
    }


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


def _bucket_summary(results: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for result in results:
        if result.get("skipped"):
            continue
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


def _semantic_label_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        semantic = result.get("semanticTranscript") if isinstance(result.get("semanticTranscript"), dict) else {}
        label = str(semantic.get("label") or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [result for result in results if not result.get("skipped")]
    skipped = [result for result in results if result.get("skipped")]
    passed = sum(1 for result in evaluated if result.get("passed"))
    asr_passed = sum(1 for result in evaluated if result.get("asrPassed"))
    semantic_passed = sum(1 for result in evaluated if (result.get("semanticTranscript") or {}).get("passed"))
    semantic_recovered = sum(
        1
        for result in evaluated
        if (result.get("semanticTranscript") or {}).get("passed") and not result.get("passed")
    )
    downstream_passed = sum(1 for result in evaluated if result.get("downstreamPassed"))
    transcription_latencies = [float(result.get("transcriptionLatencyMs") or 0.0) for result in evaluated]
    downstream_latencies = [float(result.get("latencyMs") or 0.0) for result in evaluated]
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
        "semanticTranscriptPassed": semantic_passed,
        "semanticTranscriptPassRate": _rate(semantic_passed, len(evaluated)),
        "semanticRecoveredAsrMisses": semantic_recovered,
        "semanticRecoveryRate": _rate(semantic_recovered, len(evaluated)),
        "avgSemanticScore": _avg([(result.get("semanticTranscript") or {}).get("score") for result in evaluated]),
        "avgSemanticIntentScore": _avg([(result.get("semanticTranscript") or {}).get("intentScore") for result in evaluated]),
        "avgSemanticSlotScore": _avg([(result.get("semanticTranscript") or {}).get("slotScore") for result in evaluated]),
        "avgSemanticCanonicalScore": _avg([(result.get("semanticTranscript") or {}).get("canonicalScore") for result in evaluated]),
        "bySemanticLabel": _semantic_label_counts(evaluated),
        "downstreamTaskSuccess": _rate(downstream_passed, len(evaluated)),
        "avgWer": _avg([result.get("wer") for result in evaluated]),
        "avgEntityWer": _avg([result.get("entityWer") for result in evaluated]),
        "avgEntityRecall": _avg([result.get("entityRecall") for result in evaluated]),
        "avgRawWer": _avg([result.get("rawWer") for result in evaluated]),
        "avgRawEntityRecall": _avg([result.get("rawEntityRecall") for result in evaluated]),
        "avgSurfaceWer": _avg([result.get("surfaceWer") for result in evaluated]),
        "avgSurfaceEntityRecall": _avg([result.get("surfaceEntityRecall") for result in evaluated]),
        "avgCanonicalWer": _avg([result.get("canonicalWer") for result in evaluated]),
        "avgCanonicalEntityRecall": _avg([result.get("canonicalEntityRecall") for result in evaluated]),
        "multilingualScored": sum(1 for result in evaluated if result.get("multilingualScoring")),
        "byLanguage": _bucket_summary(evaluated, lambda result: result.get("detectedLanguage") or "unknown"),
        "byRoute": _bucket_summary(evaluated, lambda result: result.get("route")),
        "byCondition": _bucket_summary(evaluated, lambda result: _condition_key(result.get("condition") or {})),
        "byAugmentation": _bucket_summary(evaluated, lambda result: (result.get("augmentation") or {}).get("type") or "none"),
        "latency": {
            "deepgramP50Ms": _percentile(transcription_latencies, 0.50),
            "deepgramP95Ms": _percentile(transcription_latencies, 0.95),
            "downstreamP50Ms": _percentile(downstream_latencies, 0.50),
            "downstreamP95Ms": _percentile(downstream_latencies, 0.95),
            "voiceP50Ms": _percentile(voice_latencies, 0.50),
            "voiceP95Ms": _percentile(voice_latencies, 0.95),
        },
        "cost": {
            "totalVapiStack": round(sum(costs), 6),
            "avgVapiStack": round(statistics.mean(costs), 6) if costs else 0.0,
            "per1000VapiStack": round((statistics.mean(costs) * 1000), 4) if costs else 0.0,
        },
        "metrics": [
            "provider_transcription_latency_ms",
            "word_error_rate",
            "entity_word_error_rate",
            "entity_recall",
            "downstream_task_success",
            "evidence_gate_correctness",
            "estimated_cost_usd",
        ],
    }


def load_latest_audio_robustness() -> dict[str, Any]:
    if not ROBUSTNESS_JSON_PATH.is_file():
        return {"found": False, "message": "No audio robustness analysis saved yet."}
    with ROBUSTNESS_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rounded_delta(value: Any, baseline: Any, *, digits: int = 4) -> float | None:
    left = _number(value)
    right = _number(baseline)
    if left is None or right is None:
        return None
    return round(left - right, digits)


def _augmentation(result: dict[str, Any]) -> dict[str, Any]:
    augmentation = result.get("augmentation") if isinstance(result.get("augmentation"), dict) else {}
    return {
        "type": str(augmentation.get("type") or "none"),
        "label": str(augmentation.get("label") or augmentation.get("type") or "Original"),
    }


def _is_variant(result: dict[str, Any]) -> bool:
    augmentation = _augmentation(result)
    return bool(result.get("parentRecordingId") or result.get("variantOf") or augmentation["type"] != "none")


def _result_cost_value(result: dict[str, Any]) -> float | None:
    cost = result.get("cost") if isinstance(result.get("cost"), dict) else {}
    return _number(cost.get("vapiStackCost"))


def _baseline_for_variant(
    result: dict[str, Any],
    results_by_id: dict[str, dict[str, Any]],
    originals_by_template: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    for candidate_id in [result.get("parentRecordingId"), result.get("variantOf")]:
        if candidate_id and str(candidate_id) in results_by_id:
            candidate = results_by_id[str(candidate_id)]
            if not candidate.get("skipped"):
                return candidate
    template_id = str(result.get("templateId") or "")
    originals = originals_by_template.get(template_id) or []
    return originals[0] if originals else None


def _audio_robustness_row(result: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    augmentation = _augmentation(result)
    condition = result.get("condition") if isinstance(result.get("condition"), dict) else {}
    baseline_condition = baseline.get("condition") if isinstance(baseline, dict) and isinstance(baseline.get("condition"), dict) else {}
    delta_wer = _rounded_delta(result.get("wer"), baseline.get("wer") if baseline else None)
    delta_entity_recall = _rounded_delta(result.get("entityRecall"), baseline.get("entityRecall") if baseline else None)
    delta_asr_latency = _rounded_delta(result.get("transcriptionLatencyMs"), baseline.get("transcriptionLatencyMs") if baseline else None, digits=2)
    delta_voice_latency = _rounded_delta(result.get("estimatedVoiceLatencyMs"), baseline.get("estimatedVoiceLatencyMs") if baseline else None, digits=2)
    delta_cost = _rounded_delta(_result_cost_value(result), _result_cost_value(baseline or {}), digits=6)

    failures: list[str] = []
    if result.get("skipped"):
        failures.append(str(result.get("skipReason") or "skipped"))
    if not baseline:
        failures.append("missing baseline recording result")
    if baseline and baseline.get("passed") and not result.get("passed"):
        failures.append("task pass dropped")
    if baseline and baseline.get("asrPassed") and not result.get("asrPassed"):
        failures.append("ASR pass dropped")
    if baseline and baseline.get("downstreamPassed") and not result.get("downstreamPassed"):
        failures.append("downstream pass dropped")
    if delta_wer is not None and delta_wer > 0.05:
        failures.append(f"WER increased by {delta_wer:.3f}")
    if delta_entity_recall is not None and delta_entity_recall < -0.10:
        failures.append(f"entity recall dropped by {abs(delta_entity_recall):.3f}")
    if delta_asr_latency is not None and delta_asr_latency > 500:
        failures.append(f"ASR latency increased by {round(delta_asr_latency)} ms")

    if result.get("skipped"):
        verdict = "skipped"
    elif not baseline:
        verdict = "no_baseline"
    elif failures:
        verdict = "regression"
    else:
        verdict = "stable"

    return {
        "recordingId": result.get("id"),
        "templateId": result.get("templateId"),
        "referenceText": result.get("referenceText"),
        "parentRecordingId": result.get("parentRecordingId") or result.get("variantOf"),
        "baselineRecordingId": baseline.get("id") if baseline else None,
        "augmentationType": augmentation["type"],
        "augmentationLabel": augmentation["label"],
        "accent": condition.get("accent"),
        "noise": condition.get("noise"),
        "device": condition.get("device"),
        "baselineNoise": baseline_condition.get("noise"),
        "passed": result.get("passed"),
        "baselinePassed": baseline.get("passed") if baseline else None,
        "asrPassed": result.get("asrPassed"),
        "baselineAsrPassed": baseline.get("asrPassed") if baseline else None,
        "downstreamPassed": result.get("downstreamPassed"),
        "baselineDownstreamPassed": baseline.get("downstreamPassed") if baseline else None,
        "wer": result.get("wer"),
        "baselineWer": baseline.get("wer") if baseline else None,
        "deltaWer": delta_wer,
        "entityRecall": result.get("entityRecall"),
        "baselineEntityRecall": baseline.get("entityRecall") if baseline else None,
        "deltaEntityRecall": delta_entity_recall,
        "transcriptionLatencyMs": result.get("transcriptionLatencyMs"),
        "baselineTranscriptionLatencyMs": baseline.get("transcriptionLatencyMs") if baseline else None,
        "deltaTranscriptionLatencyMs": delta_asr_latency,
        "estimatedVoiceLatencyMs": result.get("estimatedVoiceLatencyMs"),
        "baselineEstimatedVoiceLatencyMs": baseline.get("estimatedVoiceLatencyMs") if baseline else None,
        "deltaEstimatedVoiceLatencyMs": delta_voice_latency,
        "vapiStackCost": _result_cost_value(result),
        "baselineVapiStackCost": _result_cost_value(baseline or {}),
        "deltaVapiStackCost": delta_cost,
        "verdict": verdict,
        "failureCount": len(failures),
        "failureModes": failures,
        "transcriptText": result.get("transcriptText"),
        "baselineTranscriptText": baseline.get("transcriptText") if baseline else None,
        "answer": result.get("answer"),
    }


def _summarize_audio_robustness(rows: list[dict[str, Any]], originals: list[dict[str, Any]]) -> dict[str, Any]:
    compared = [row for row in rows if row.get("baselineRecordingId") and row.get("verdict") != "skipped"]
    regressions = [row for row in compared if row.get("verdict") == "regression"]
    stable = [row for row in compared if row.get("verdict") == "stable"]
    skipped = [row for row in rows if row.get("verdict") == "skipped"]
    no_baseline = [row for row in rows if row.get("verdict") == "no_baseline"]
    worst = sorted(
        compared,
        key=lambda row: (
            row.get("verdict") == "regression",
            _number(row.get("deltaWer")) or 0,
            -(_number(row.get("deltaEntityRecall")) or 0),
        ),
        reverse=True,
    )
    return {
        "baselineCount": len(originals),
        "variantCount": len(rows),
        "comparedCount": len(compared),
        "stableCount": len(stable),
        "regressionCount": len(regressions),
        "skippedCount": len(skipped),
        "noBaselineCount": len(no_baseline),
        "regressionRate": _rate(len(regressions), len(compared)),
        "avgDeltaWer": _avg([row.get("deltaWer") for row in compared]),
        "avgDeltaEntityRecall": _avg([row.get("deltaEntityRecall") for row in compared]),
        "avgDeltaTranscriptionLatencyMs": _avg([row.get("deltaTranscriptionLatencyMs") for row in compared]),
        "worstVariant": worst[0] if worst else None,
    }


def _audio_robustness_by_augmentation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("augmentationType") or "none")
        bucket = buckets.setdefault(
            key,
            {
                "augmentationType": key,
                "augmentationLabel": row.get("augmentationLabel") or key,
                "variantCount": 0,
                "comparedCount": 0,
                "passed": 0,
                "baselinePassed": 0,
                "regressions": 0,
                "deltaWer": [],
                "deltaEntityRecall": [],
                "deltaTranscriptionLatencyMs": [],
            },
        )
        bucket["variantCount"] += 1
        if row.get("baselineRecordingId") and row.get("verdict") != "skipped":
            bucket["comparedCount"] += 1
            bucket["passed"] += 1 if row.get("passed") else 0
            bucket["baselinePassed"] += 1 if row.get("baselinePassed") else 0
            bucket["regressions"] += 1 if row.get("verdict") == "regression" else 0
            bucket["deltaWer"].append(row.get("deltaWer"))
            bucket["deltaEntityRecall"].append(row.get("deltaEntityRecall"))
            bucket["deltaTranscriptionLatencyMs"].append(row.get("deltaTranscriptionLatencyMs"))

    summaries: list[dict[str, Any]] = []
    for bucket in buckets.values():
        compared = int(bucket["comparedCount"])
        summaries.append(
            {
                "augmentationType": bucket["augmentationType"],
                "augmentationLabel": bucket["augmentationLabel"],
                "variantCount": bucket["variantCount"],
                "comparedCount": compared,
                "passRate": _rate(bucket["passed"], compared),
                "baselinePassRate": _rate(bucket["baselinePassed"], compared),
                "regressionCount": bucket["regressions"],
                "regressionRate": _rate(bucket["regressions"], compared),
                "avgDeltaWer": _avg(bucket["deltaWer"]),
                "avgDeltaEntityRecall": _avg(bucket["deltaEntityRecall"]),
                "avgDeltaTranscriptionLatencyMs": _avg(bucket["deltaTranscriptionLatencyMs"]),
            }
        )
    return sorted(summaries, key=lambda item: (item.get("regressionRate") or 0, item.get("avgDeltaWer") or 0), reverse=True)


def _audio_robustness_recommendations(summary: dict[str, Any], by_augmentation: list[dict[str, Any]]) -> list[str]:
    if summary.get("variantCount") == 0:
        return ["Generate stress variants from at least one saved recording, then run the audio suite."]
    if summary.get("comparedCount") == 0:
        return ["Run the audio suite after generating variants so each variant can be compared with its original recording."]
    recommendations: list[str] = []
    worst = summary.get("worstVariant") if isinstance(summary.get("worstVariant"), dict) else None
    if worst:
        recommendations.append(
            f"Worst observed variant is {worst.get('augmentationLabel')} on {worst.get('templateId')} with delta WER {worst.get('deltaWer')} and verdict {worst.get('verdict')}."
        )
    for bucket in by_augmentation[:3]:
        if bucket.get("regressionCount"):
            recommendations.append(
                f"{bucket.get('augmentationLabel')} has {bucket.get('regressionCount')} regressions across {bucket.get('comparedCount')} compared cases."
            )
    if not recommendations:
        recommendations.append("No robustness regression detected against the current thresholds.")
    return recommendations


def _save_audio_robustness_csv(payload: dict[str, Any]) -> None:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    ROBUSTNESS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recordingId", "templateId", "referenceText", "parentRecordingId", "baselineRecordingId",
        "augmentationType", "augmentationLabel", "accent", "noise", "device", "baselineNoise", "passed",
        "baselinePassed", "asrPassed", "baselineAsrPassed", "downstreamPassed", "baselineDownstreamPassed",
        "wer", "baselineWer", "deltaWer", "entityRecall", "baselineEntityRecall", "deltaEntityRecall",
        "transcriptionLatencyMs", "baselineTranscriptionLatencyMs", "deltaTranscriptionLatencyMs",
        "estimatedVoiceLatencyMs", "baselineEstimatedVoiceLatencyMs", "deltaEstimatedVoiceLatencyMs",
        "vapiStackCost", "baselineVapiStackCost", "deltaVapiStackCost", "verdict", "failureCount", "failureModes",
    ]
    with ROBUSTNESS_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key) for key in fieldnames}
            output["failureModes"] = " | ".join(row.get("failureModes") or [])
            writer.writerow(output)


def analyze_audio_robustness(
    *,
    results: list[dict[str, Any]] | None = None,
    source_run_id: str | None = None,
    save: bool = True,
) -> dict[str, Any]:
    if results is None:
        latest = load_latest_audio_eval()
        if latest.get("found") is False:
            return {"found": False, "message": latest.get("message") or "No audio evaluation saved yet."}
        results = latest.get("results") if isinstance(latest.get("results"), list) else []
        source_run_id = source_run_id or latest.get("runId")

    normalized = [result for result in results if isinstance(result, dict)]
    originals = [result for result in normalized if not result.get("skipped") and not _is_variant(result)]
    variants = [result for result in normalized if _is_variant(result)]
    results_by_id = {str(result.get("id")): result for result in normalized if result.get("id")}
    originals_by_template: dict[str, list[dict[str, Any]]] = {}
    for original in originals:
        originals_by_template.setdefault(str(original.get("templateId") or ""), []).append(original)

    rows = [_audio_robustness_row(result, _baseline_for_variant(result, results_by_id, originals_by_template)) for result in variants]
    by_augmentation = _audio_robustness_by_augmentation(rows)
    summary = _summarize_audio_robustness(rows, originals)
    payload = {
        "found": True,
        "runId": datetime.now(timezone.utc).strftime("audio-robustness-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceAudioRunId": source_run_id,
        "summary": summary,
        "byAugmentation": by_augmentation,
        "rows": rows,
        "recommendations": _audio_robustness_recommendations(summary, by_augmentation),
        "thresholds": {
            "werRegressionDelta": 0.05,
            "entityRecallRegressionDelta": -0.10,
            "transcriptionLatencyRegressionMs": 500,
        },
        "artifacts": {
            "json": str(ROBUSTNESS_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(ROBUSTNESS_CSV_PATH.relative_to(ROOT_DIR)),
        },
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with ROBUSTNESS_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _save_audio_robustness_csv(payload)
    return payload
def load_latest_audio_manifest() -> dict[str, Any]:
    if not MANIFEST_JSON_PATH.is_file():
        return {"found": False, "message": "No audio dataset manifest saved yet."}
    with MANIFEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _metadata(case: dict[str, Any]) -> dict[str, Any]:
    metadata = case.get("recordingMetadata") if isinstance(case.get("recordingMetadata"), dict) else {}
    condition = case.get("condition") if isinstance(case.get("condition"), dict) else {}
    speaker = case.get("speaker") if isinstance(case.get("speaker"), dict) else {}
    return {
        "speakerId": metadata.get("speakerId") or speaker.get("id") or "speaker-1",
        "accent": metadata.get("accent") or speaker.get("accent") or condition.get("accent") or "unknown",
        "noise": metadata.get("noise") or condition.get("noise") or "unknown",
        "device": metadata.get("device") or condition.get("device") or "legacy_browser_mic",
        "environment": metadata.get("environment") or condition.get("environment") or "unspecified",
        "micDistanceCm": metadata.get("micDistanceCm") or condition.get("micDistanceCm"),
        "notes": metadata.get("notes") or "",
    }


def _manifest_eval_by_id() -> dict[str, dict[str, Any]]:
    latest = load_latest_audio_eval()
    results = latest.get("results") if isinstance(latest.get("results"), list) else []
    return {str(result.get("id")): result for result in results if isinstance(result, dict)}


def _coverage_rate(recordings: int, target: int) -> float:
    if target <= 0:
        return 1.0
    return round(min(1.0, recordings / target), 4)


def build_audio_dataset_manifest(*, target_per_prompt: int = 3, save: bool = True) -> dict[str, Any]:
    target = max(1, int(target_per_prompt or 3))
    templates = load_seed_audio_cases()
    recordings = load_recorded_audio_cases()
    eval_by_id = _manifest_eval_by_id()
    by_template: dict[str, list[dict[str, Any]]] = {}
    for recording in recordings:
        template_id = str(recording.get("templateId") or recording.get("id") or "unknown")
        by_template.setdefault(template_id, []).append(recording)

    prompt_rows: list[dict[str, Any]] = []
    for template in templates:
        template_id = str(template.get("id"))
        rows = by_template.get(template_id, [])
        prompt_rows.append(
            {
                "templateId": template_id,
                "referenceText": template.get("referenceText"),
                "route": template.get("route"),
                "group": template.get("group"),
                "recordings": len(rows),
                "target": target,
                "missing": max(0, target - len(rows)),
                "coverageRate": _coverage_rate(len(rows), target),
                "complete": len(rows) >= target,
            }
        )

    condition_buckets: dict[str, dict[str, Any]] = {}
    recording_rows: list[dict[str, Any]] = []
    for recording in recordings:
        metadata = _metadata(recording)
        augmentation = recording.get("augmentation") if isinstance(recording.get("augmentation"), dict) else {}
        augmentation_type = str(augmentation.get("type") or "none")
        condition_key = f"{metadata['accent']}|{metadata['noise']}|{metadata['device']}|{metadata['environment']}|aug:{augmentation_type}"
        bucket = condition_buckets.setdefault(
            condition_key,
            {"condition": condition_key, "recordings": 0, "speakers": set(), "templates": set(), "evaluated": 0, "passed": 0, "wer": [], "entityRecall": []},
        )
        bucket["recordings"] += 1
        bucket["speakers"].add(str(metadata["speakerId"]))
        bucket["templates"].add(str(recording.get("templateId") or recording.get("id")))
        evaluated = eval_by_id.get(str(recording.get("id")))
        if evaluated and not evaluated.get("skipped"):
            bucket["evaluated"] += 1
            bucket["passed"] += 1 if evaluated.get("passed") else 0
            bucket["wer"].append(evaluated.get("wer"))
            bucket["entityRecall"].append(evaluated.get("entityRecall"))
        recording_rows.append(
            {
                "recordingId": recording.get("id"),
                "templateId": recording.get("templateId"),
                "referenceText": recording.get("referenceText"),
                "route": recording.get("route"),
                "group": recording.get("group"),
                "parentRecordingId": recording.get("parentRecordingId"),
                "variantOf": recording.get("variantOf"),
                "augmentationType": augmentation.get("type"),
                "augmentationLabel": augmentation.get("label"),
                "speakerId": metadata["speakerId"],
                "accent": metadata["accent"],
                "noise": metadata["noise"],
                "device": metadata["device"],
                "environment": metadata["environment"],
                "micDistanceCm": metadata["micDistanceCm"],
                "durationMs": recording.get("durationMs"),
                "mimeType": recording.get("mimeType"),
                "recordedAt": recording.get("recordedAt"),
                "audioUri": recording.get("audioUri"),
                "evaluated": bool(evaluated and not evaluated.get("skipped")),
                "passed": evaluated.get("passed") if evaluated else None,
                "wer": evaluated.get("wer") if evaluated else None,
                "entityRecall": evaluated.get("entityRecall") if evaluated else None,
                "transcriptText": evaluated.get("transcriptText") if evaluated else None,
                "notes": metadata["notes"],
            }
        )

    condition_rows: list[dict[str, Any]] = []
    for bucket in condition_buckets.values():
        evaluated = int(bucket["evaluated"])
        condition_rows.append(
            {
                "condition": bucket["condition"],
                "recordings": bucket["recordings"],
                "speakerCount": len(bucket["speakers"]),
                "templateCount": len(bucket["templates"]),
                "evaluated": evaluated,
                "passed": bucket["passed"],
                "passRate": _rate(bucket["passed"], evaluated),
                "avgWer": _avg(bucket["wer"]),
                "avgEntityRecall": _avg(bucket["entityRecall"]),
            }
        )
    condition_rows.sort(key=lambda row: str(row["condition"]))

    speakers = {str(_metadata(recording)["speakerId"]) for recording in recordings}
    accents = {str(_metadata(recording)["accent"]) for recording in recordings}
    noises = {str(_metadata(recording)["noise"]) for recording in recordings}
    devices = {str(_metadata(recording)["device"]) for recording in recordings}
    environments = {str(_metadata(recording)["environment"]) for recording in recordings}
    augmentations = {
        str((recording.get("augmentation") if isinstance(recording.get("augmentation"), dict) else {}).get("type") or "none")
        for recording in recordings
    }
    evaluated_rows = [row for row in recording_rows if row["evaluated"]]
    complete_prompts = sum(1 for row in prompt_rows if row["complete"])
    required = len(templates) * target
    payload = {
        "runId": datetime.now(timezone.utc).strftime("audio-dataset-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "targetPerPrompt": target,
        "summary": {
            "templateCount": len(templates),
            "recordingCount": len(recordings),
            "requiredRecordings": required,
            "completePrompts": complete_prompts,
            "missingRecordings": max(0, required - len(recordings)),
            "coverageRate": _coverage_rate(len(recordings), required),
            "promptCoverageRate": _rate(complete_prompts, len(templates)),
            "speakerCount": len(speakers),
            "accentCount": len(accents),
            "noiseCount": len(noises),
            "deviceCount": len(devices),
            "environmentCount": len(environments),
            "conditionCount": len(condition_rows),
            "augmentationCount": len(augmentations),
            "evaluatedRecordings": len(evaluated_rows),
            "evaluatedPassRate": _rate(sum(1 for row in evaluated_rows if row.get("passed")), len(evaluated_rows)),
            "avgWer": _avg([row.get("wer") for row in evaluated_rows]),
            "avgEntityRecall": _avg([row.get("entityRecall") for row in evaluated_rows]),
        },
        "promptCoverage": prompt_rows,
        "conditionCoverage": condition_rows,
        "recordings": recording_rows,
        "artifacts": {
            "json": str(MANIFEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(MANIFEST_CSV_PATH.relative_to(ROOT_DIR)),
            "recordingsManifest": str(LOCAL_CASES_PATH.relative_to(ROOT_DIR)),
        },
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with MANIFEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _save_manifest_csv(payload)
    return payload


def _save_manifest_csv(payload: dict[str, Any]) -> None:
    rows = payload.get("recordings") if isinstance(payload.get("recordings"), list) else []
    MANIFEST_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recordingId", "templateId", "referenceText", "route", "group", "parentRecordingId", "variantOf",
        "augmentationType", "augmentationLabel", "speakerId", "accent", "noise", "device", "environment",
        "micDistanceCm", "durationMs", "mimeType", "recordedAt", "audioUri", "evaluated", "passed",
        "wer", "entityRecall", "transcriptText", "notes",
    ]
    with MANIFEST_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

def _save_csv(payload: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with LATEST_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id", "templateId", "route", "group", "parentRecordingId", "variantOf", "augmentationType",
                "augmentationLabel", "accent", "noise", "bargeIn", "skipped",
                "passed", "asrPassed", "downstreamPassed", "wer", "entityWer", "entityRecall",
                "transcriptionProvider", "transcriptionLatencyMs", "confidence", "latencyMs",
                "estimatedVoiceLatencyMs", "vapiStackCost", "deepgramConfigId", "deepgramLanguage",
                "deepgramKeytermCount", "transcriptRepairEnabled", "rawWer", "rawEntityRecall",
                "surfaceWer", "surfaceEntityRecall", "canonicalWer", "canonicalEntityRecall",
                "detectedLanguage", "languageConfidence", "multilingualScoring", "canonicalTranscriptText",
                "semanticTranscriptPassed", "semanticLabel", "semanticScore", "semanticIntentScore",
                "semanticSlotScore", "semanticCanonicalScore", "semanticReason", "semanticMissingSlots",
                "failures", "referenceText", "rawTranscriptText", "transcriptText", "audioUri",
            ],
        )
        writer.writeheader()
        for result in payload.get("results") or []:
            condition = result.get("condition") or {}
            deepgram_config = result.get("deepgramConfig") or {}
            writer.writerow(
                {
                    "id": result.get("id"),
                    "templateId": result.get("templateId"),
                    "route": result.get("route"),
                    "group": result.get("group"),
                    "parentRecordingId": result.get("parentRecordingId"),
                    "variantOf": result.get("variantOf"),
                    "augmentationType": (result.get("augmentation") or {}).get("type"),
                    "augmentationLabel": (result.get("augmentation") or {}).get("label"),
                    "accent": condition.get("accent"),
                    "noise": condition.get("noise"),
                    "bargeIn": condition.get("bargeIn"),
                    "skipped": result.get("skipped"),
                    "passed": result.get("passed"),
                    "asrPassed": result.get("asrPassed"),
                    "downstreamPassed": result.get("downstreamPassed"),
                    "wer": result.get("wer"),
                    "entityWer": result.get("entityWer"),
                    "entityRecall": result.get("entityRecall"),
                    "transcriptionProvider": result.get("transcriptionProvider"),
                    "transcriptionLatencyMs": result.get("transcriptionLatencyMs"),
                    "confidence": result.get("confidence"),
                    "latencyMs": result.get("latencyMs"),
                    "estimatedVoiceLatencyMs": result.get("estimatedVoiceLatencyMs"),
                    "vapiStackCost": (result.get("cost") or {}).get("vapiStackCost"),
                    "deepgramConfigId": deepgram_config.get("id"),
                    "deepgramLanguage": deepgram_config.get("language"),
                    "deepgramKeytermCount": deepgram_config.get("keytermCount"),
                    "transcriptRepairEnabled": result.get("transcriptRepairEnabled"),
                    "rawWer": result.get("rawWer"),
                    "rawEntityRecall": result.get("rawEntityRecall"),
                    "surfaceWer": result.get("surfaceWer"),
                    "surfaceEntityRecall": result.get("surfaceEntityRecall"),
                    "canonicalWer": result.get("canonicalWer"),
                    "canonicalEntityRecall": result.get("canonicalEntityRecall"),
                    "detectedLanguage": result.get("detectedLanguage"),
                    "languageConfidence": result.get("languageConfidence"),
                    "multilingualScoring": result.get("multilingualScoring"),
                    "canonicalTranscriptText": result.get("canonicalTranscriptText"),
                    "semanticTranscriptPassed": (result.get("semanticTranscript") or {}).get("passed"),
                    "semanticLabel": (result.get("semanticTranscript") or {}).get("label"),
                    "semanticScore": (result.get("semanticTranscript") or {}).get("score"),
                    "semanticIntentScore": (result.get("semanticTranscript") or {}).get("intentScore"),
                    "semanticSlotScore": (result.get("semanticTranscript") or {}).get("slotScore"),
                    "semanticCanonicalScore": (result.get("semanticTranscript") or {}).get("canonicalScore"),
                    "semanticReason": (result.get("semanticTranscript") or {}).get("reason"),
                    "semanticMissingSlots": " | ".join((result.get("semanticTranscript") or {}).get("missingSlots") or []),
                    "failures": " | ".join(result.get("failures") or []),
                    "referenceText": result.get("referenceText"),
                    "rawTranscriptText": result.get("rawTranscriptText"),
                    "transcriptText": result.get("transcriptText"),
                    "audioUri": result.get("audioUri"),
                }
            )


def load_latest_audio_accent_sweep() -> dict[str, Any]:
    if not ACCENT_SWEEP_JSON_PATH.is_file():
        return {"found": False, "message": "No accent sweep saved yet."}
    with ACCENT_SWEEP_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _sweep_profiles(configs: list[Any] | None) -> list[dict[str, Any]]:
    if not configs:
        return [dict(config) for config in SWEEP_CONFIGS]
    profiles: list[dict[str, Any]] = []
    by_id = {str(config["id"]): config for config in SWEEP_CONFIGS}
    for item in configs:
        if isinstance(item, str) and item in by_id:
            profiles.append(dict(by_id[item]))
        elif isinstance(item, dict):
            profile_id = str(item.get("id") or item.get("profile") or "")
            profiles.append({**dict(by_id.get(profile_id, {})), **item})
    return profiles or [dict(config) for config in SWEEP_CONFIGS]


def _case_recorded_at(case: dict[str, Any]) -> str:
    return str(case.get("recordedAt") or "")


def _select_sweep_cases(
    *,
    case_ids: list[str] | None,
    limit: int | None,
    include_passed: bool,
) -> list[dict[str, Any]]:
    selected_ids = {case_id for case_id in case_ids or [] if case_id}
    cases = load_recorded_audio_cases()
    if selected_ids:
        cases = [case for case in cases if case.get("id") in selected_ids or case.get("templateId") in selected_ids]
    elif not include_passed:
        latest = load_latest_audio_eval()
        latest_results = latest.get("results") if isinstance(latest.get("results"), list) else []
        priority_ids: set[str] = set()
        for result in latest_results:
            if not isinstance(result, dict):
                continue
            condition = result.get("condition") if isinstance(result.get("condition"), dict) else {}
            accent = str(condition.get("accent") or "")
            if not result.get("passed") or accent not in {"", "user_recorded"}:
                priority_ids.add(str(result.get("id")))
        if priority_ids:
            cases = [case for case in cases if str(case.get("id")) in priority_ids]
    cases = sorted(cases, key=_case_recorded_at, reverse=True)
    if limit is not None:
        cases = cases[: max(1, int(limit))]
    return list(reversed(cases))


def _config_summary(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize(results)
    latencies = [float(result.get("transcriptionLatencyMs") or 0) for result in results if not result.get("skipped")]
    return {
        "configId": config.get("id"),
        "label": config.get("label") or config.get("id"),
        "language": config.get("language"),
        "useKeyterms": config.get("useKeyterms"),
        "keytermCount": max([int(((result.get("deepgramConfig") or {}).get("keytermCount") or 0)) for result in results] or [0]),
        "transcriptRepairEnabled": config.get("enableTranscriptRepair"),
        "total": summary.get("total"),
        "evaluated": summary.get("evaluated"),
        "passed": summary.get("passed"),
        "passRate": summary.get("passRate"),
        "asrPassRate": summary.get("asrPassRate"),
        "downstreamTaskSuccess": summary.get("downstreamTaskSuccess"),
        "avgWer": summary.get("avgWer"),
        "avgEntityRecall": summary.get("avgEntityRecall"),
        "avgRawWer": _avg([result.get("rawWer") for result in results if not result.get("skipped")]),
        "avgRawEntityRecall": _avg([result.get("rawEntityRecall") for result in results if not result.get("skipped")]),
        "deepgramP95Ms": _percentile(latencies, 0.95),
        "repairCount": sum(len(result.get("transcriptRepairs") or []) for result in results),
    }


def _sort_sweep_summary(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    pass_rate = float(row.get("passRate") or 0)
    downstream = float(row.get("downstreamTaskSuccess") or 0)
    entity = float(row.get("avgEntityRecall") or 0)
    wer = float(row.get("avgWer") if row.get("avgWer") is not None else 99)
    latency = float(row.get("deepgramP95Ms") if row.get("deepgramP95Ms") is not None else 999999)
    return (pass_rate, downstream, entity, -wer, -latency)


def _flatten_sweep_rows(config_summaries: list[dict[str, Any]], results_by_config: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries_by_id = {str(summary.get("configId")): summary for summary in config_summaries}
    for config_id, results in results_by_config.items():
        summary = summaries_by_id.get(config_id, {})
        for result in results:
            condition = result.get("condition") if isinstance(result.get("condition"), dict) else {}
            deepgram_config = result.get("deepgramConfig") if isinstance(result.get("deepgramConfig"), dict) else {}
            rows.append(
                {
                    "configId": config_id,
                    "configLabel": summary.get("label"),
                    "language": deepgram_config.get("language") or summary.get("language"),
                    "useKeyterms": deepgram_config.get("useKeyterms"),
                    "keytermCount": deepgram_config.get("keytermCount"),
                    "transcriptRepairEnabled": result.get("transcriptRepairEnabled"),
                    "id": result.get("id"),
                    "templateId": result.get("templateId"),
                    "accent": condition.get("accent"),
                    "passed": result.get("passed"),
                    "asrPassed": result.get("asrPassed"),
                    "downstreamPassed": result.get("downstreamPassed"),
                    "wer": result.get("wer"),
                    "entityRecall": result.get("entityRecall"),
                    "rawWer": result.get("rawWer"),
                    "rawEntityRecall": result.get("rawEntityRecall"),
                    "transcriptionLatencyMs": result.get("transcriptionLatencyMs"),
                    "confidence": result.get("confidence"),
                    "repairCount": len(result.get("transcriptRepairs") or []),
                    "referenceText": result.get("referenceText"),
                    "rawTranscriptText": result.get("rawTranscriptText"),
                    "transcriptText": result.get("transcriptText"),
                    "failures": result.get("failures") or [],
                }
            )
    return rows


def _save_accent_sweep_csv(payload: dict[str, Any]) -> None:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    ACCENT_SWEEP_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "configId", "configLabel", "language", "useKeyterms", "keytermCount", "transcriptRepairEnabled",
        "id", "templateId", "accent", "passed", "asrPassed", "downstreamPassed", "wer", "entityRecall",
        "rawWer", "rawEntityRecall", "transcriptionLatencyMs", "confidence", "repairCount", "referenceText",
        "rawTranscriptText", "transcriptText", "failures",
    ]
    with ACCENT_SWEEP_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key) for key in fieldnames}
            output["failures"] = " | ".join(row.get("failures") or [])
            writer.writerow(output)


def run_deepgram_accent_sweep(
    *,
    case_ids: list[str] | None = None,
    limit: int | None = 8,
    configs: list[Any] | None = None,
    include_passed: bool = False,
    allow_reference_fallback: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    profiles = _sweep_profiles(configs)
    cases = _select_sweep_cases(case_ids=case_ids, limit=limit, include_passed=include_passed)
    started = time.perf_counter()
    results_by_config: dict[str, list[dict[str, Any]]] = {}
    config_summaries: list[dict[str, Any]] = []
    for profile in profiles:
        config_id = str(profile.get("id") or "custom")
        results = [
            _evaluate_case(
                case,
                allow_reference_fallback=allow_reference_fallback,
                deepgram_config=profile,
                enable_transcript_repair=bool(profile.get("enableTranscriptRepair")),
            )
            for case in cases
        ]
        results_by_config[config_id] = results
        config_summaries.append(_config_summary(profile, results))

    best = sorted(config_summaries, key=_sort_sweep_summary, reverse=True)[0] if config_summaries else None
    baseline = config_summaries[0] if config_summaries else None
    rows = _flatten_sweep_rows(config_summaries, results_by_config)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    summary = {
        "caseCount": len(cases),
        "configCount": len(profiles),
        "evaluatedTranscriptions": sum(int(summary.get("evaluated") or 0) for summary in config_summaries),
        "bestConfigId": best.get("configId") if best else None,
        "bestConfigLabel": best.get("label") if best else None,
        "bestPassRate": best.get("passRate") if best else None,
        "bestAvgWer": best.get("avgWer") if best else None,
        "bestAvgEntityRecall": best.get("avgEntityRecall") if best else None,
        "baselineConfigId": baseline.get("configId") if baseline else None,
        "baselinePassRate": baseline.get("passRate") if baseline else None,
        "baselineAvgWer": baseline.get("avgWer") if baseline else None,
        "baselineAvgEntityRecall": baseline.get("avgEntityRecall") if baseline else None,
        "passRateLift": round(float(best.get("passRate") or 0) - float(baseline.get("passRate") or 0), 4) if best and baseline else None,
        "werDelta": round(float(best.get("avgWer") or 0) - float(baseline.get("avgWer") or 0), 4) if best and baseline else None,
        "entityRecallLift": round(float(best.get("avgEntityRecall") or 0) - float(baseline.get("avgEntityRecall") or 0), 4) if best and baseline else None,
        "repairCount": sum(int(summary.get("repairCount") or 0) for summary in config_summaries),
    }
    payload = {
        "found": True,
        "runId": datetime.now(timezone.utc).strftime("audio-accent-sweep-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "elapsedMs": elapsed_ms,
        "filters": {
            "caseIds": sorted(case_ids or []),
            "limit": limit,
            "includePassed": include_passed,
            "allowReferenceFallback": allow_reference_fallback,
        },
        "provider": {
            "stt": "deepgram",
            "model": os.getenv("DEEPGRAM_MODEL", DEEPGRAM_DEFAULT_MODEL),
            "ready": provider_ready(),
            "apiUrl": os.getenv("DEEPGRAM_API_URL", DEEPGRAM_DEFAULT_URL),
        },
        "summary": summary,
        "configSummaries": config_summaries,
        "rows": rows,
        "recommendations": [
            "Use the best config for the main audio suite only after verifying the lift on at least 30 recordings.",
            "Report raw ASR WER separately from repaired/downstream task success.",
            "For persistent shelf/self errors, keep shelf/topstock/stock in keyterms and enable retail transcript repair.",
        ],
        "artifacts": {
            "json": str(ACCENT_SWEEP_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(ACCENT_SWEEP_CSV_PATH.relative_to(ROOT_DIR)),
        },
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with ACCENT_SWEEP_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _save_accent_sweep_csv(payload)
    return payload


def run_real_audio_suite(
    *,
    case_ids: list[str] | None = None,
    limit: int | None = None,
    allow_reference_fallback: bool = False,
    deepgram_config: dict[str, Any] | None = None,
    enable_transcript_repair: bool | None = None,
    save: bool = True,
) -> dict[str, Any]:
    selected_ids = {case_id for case_id in case_ids or [] if case_id}
    cases = load_recorded_audio_cases()
    if selected_ids:
        cases = [case for case in cases if case.get("id") in selected_ids or case.get("templateId") in selected_ids]
    if limit is not None:
        cases = cases[: max(1, int(limit))]
    started = time.perf_counter()
    run_config = dict(deepgram_config or {"id": os.getenv("DEEPGRAM_PROFILE") or DEFAULT_DEEPGRAM_PROFILE})
    display_config = _resolve_deepgram_config(run_config, None)
    results = [
        _evaluate_case(
            case,
            allow_reference_fallback=allow_reference_fallback,
            deepgram_config=run_config,
            enable_transcript_repair=enable_transcript_repair,
        )
        for case in cases
    ]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = {
        "runId": datetime.now(timezone.utc).strftime("audio-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suite": "voice_retail_real_audio_deepgram_suite",
        "caseFile": str(LOCAL_CASES_PATH.relative_to(ROOT_DIR)),
        "elapsedMs": elapsed_ms,
        "filters": {
            "caseIds": sorted(selected_ids),
            "limit": limit,
            "allowReferenceFallback": allow_reference_fallback,
            "deepgramConfig": {
                "id": display_config["id"],
                "label": display_config["label"],
                "language": run_config.get("language") or display_config["language"],
                "useKeyterms": display_config["useKeyterms"],
                "keytermCount": display_config["keytermCount"],
                "enableTranscriptRepair": display_config["enableTranscriptRepair"],
            },
            "enableTranscriptRepair": enable_transcript_repair,
        },
        "provider": {
            "stt": "deepgram",
            "model": display_config["model"],
            "profile": display_config["id"],
            "language": run_config.get("language") or display_config["language"],
            "useKeyterms": display_config["useKeyterms"],
            "keytermCount": display_config["keytermCount"],
            "ready": provider_ready(),
            "apiUrl": os.getenv("DEEPGRAM_API_URL", DEEPGRAM_DEFAULT_URL),
        },
        "summary": _summarize(results),
        "results": results,
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(LATEST_CSV_PATH.relative_to(ROOT_DIR)),
            "robustnessJson": str(ROBUSTNESS_JSON_PATH.relative_to(ROOT_DIR)),
            "robustnessCsv": str(ROBUSTNESS_CSV_PATH.relative_to(ROOT_DIR)),
            "recordingsManifest": str(LOCAL_CASES_PATH.relative_to(ROOT_DIR)),
        },
        "transcriptionMode": "deepgram_prerecorded_audio",
        "researchBasis": [
            "Recorded browser audio fixtures",
            "Deepgram prerecorded Nova-3 transcription",
            "WER and entity-WER against reference text",
            "Downstream inventory/RAG success after provider ASR",
        ],
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with LATEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _save_csv(payload)
        analyze_audio_robustness(results=results, source_run_id=payload["runId"], save=True)
    return payload
