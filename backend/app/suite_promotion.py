from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .draft_validation import PROMOTION_JSON_PATH, run_draft_validation


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "paper"
BACKUP_DIR = ARTIFACT_DIR / "suite_backups"
LATEST_JSON_PATH = ARTIFACT_DIR / "suite_promotion_latest.json"
LATEST_CSV_PATH = ARTIFACT_DIR / "suite_promotion_latest.csv"

BENCHMARK_CASES_PATH = ROOT_DIR / "benchmarks" / "eval_cases.json"
SPEECH_CASES_PATH = ROOT_DIR / "benchmarks" / "speech_cases.json"
PROMOTED_AUDIO_QUEUE_PATH = ROOT_DIR / "artifacts" / "audio_eval" / "promoted_recording_queue.json"


def load_latest_suite_promotion() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No suite promotion run saved yet."}
    with LATEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.is_file():
        return fallback
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _relative(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(ROOT_DIR))


def _case_id(case: dict[str, Any]) -> str:
    return str(case.get("id") or "")


def _is_factory_case(case: dict[str, Any]) -> bool:
    factory = case.get("factory") if isinstance(case.get("factory"), dict) else {}
    return factory.get("source") == "case_factory" or _case_id(case).startswith("factory_")


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _promoted_case(case: dict[str, Any], *, run_id: str, created_at: str, target: str) -> dict[str, Any]:
    promoted = _json_copy(case)
    factory = promoted.get("factory") if isinstance(promoted.get("factory"), dict) else {}
    promoted["factory"] = {
        **factory,
        "source": factory.get("source") or "case_factory",
        "promotion": "promoted",
        "promotedAt": created_at,
        "promotionRunId": run_id,
        "promotionTarget": target,
    }
    return promoted


def _load_manifest(refresh_validation: bool) -> dict[str, Any]:
    if refresh_validation or not PROMOTION_JSON_PATH.is_file():
        validation = run_draft_validation(refresh_factory=refresh_validation, include_payloads=False, save=True)
        return validation.get("promotionManifest") or {}
    return _load_json(PROMOTION_JSON_PATH, {})


def _backup_path(path: Path, run_id: str) -> Path | None:
    if not path.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"{path.stem}-{run_id}{path.suffix}"
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def _plan_cases(
    *,
    target: str,
    target_path: Path,
    existing_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
    run_id: str,
    created_at: str,
    dry_run: bool,
    replace_factory_cases: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    retained = [case for case in existing_cases if not (replace_factory_cases and _is_factory_case(case))]
    replaced = len(existing_cases) - len(retained)
    seen_ids = {_case_id(case) for case in retained if _case_id(case)}
    candidate_seen: set[str] = set()
    added: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for case in candidate_cases:
        case_id = _case_id(case)
        if not case_id:
            skipped.append({"id": "", "target": target, "reason": "missing_id"})
            continue
        if case_id in candidate_seen:
            skipped.append({"id": case_id, "target": target, "reason": "duplicate_candidate_id"})
            continue
        candidate_seen.add(case_id)
        if case_id in seen_ids:
            skipped.append({"id": case_id, "target": target, "reason": "duplicate_existing_id"})
            continue
        promoted = _promoted_case(case, run_id=run_id, created_at=created_at, target=str(target_path.relative_to(ROOT_DIR)))
        added.append(promoted)
        seen_ids.add(case_id)

    final_cases = [*retained, *added]
    row = {
        "target": target,
        "targetPath": str(target_path.relative_to(ROOT_DIR)),
        "existingCount": len(existing_cases),
        "replacedFactoryCount": replaced,
        "candidateCount": len(candidate_cases),
        "addedCount": len(added),
        "skippedCount": len(skipped),
        "finalCount": len(final_cases),
        "dryRun": dry_run,
        "wrote": False,
        "backupPath": None,
    }
    return final_cases, row, skipped


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "targetPath",
        "existingCount",
        "replacedFactoryCount",
        "candidateCount",
        "addedCount",
        "skippedCount",
        "finalCount",
        "dryRun",
        "wrote",
        "backupPath",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_suite_promotion(
    *,
    dry_run: bool = True,
    replace_factory_cases: bool = True,
    include_benchmark: bool = True,
    include_speech: bool = True,
    include_audio_queue: bool = True,
    refresh_validation: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("suite-promo-%Y%m%d%H%M%S")
    manifest = _load_manifest(refresh_validation)
    blocked = manifest.get("blocked") if isinstance(manifest.get("blocked"), list) else []

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    written: list[dict[str, str | None]] = []

    benchmark_added: list[dict[str, Any]] = []
    speech_added: list[dict[str, Any]] = []
    audio_queue: list[dict[str, Any]] = []

    if include_benchmark:
        existing = _load_json(BENCHMARK_CASES_PATH, [])
        candidates = [case for case in manifest.get("readyBenchmarkCases") or [] if isinstance(case, dict)]
        final_cases, row, target_skipped = _plan_cases(
            target="benchmark_cases",
            target_path=BENCHMARK_CASES_PATH,
            existing_cases=existing if isinstance(existing, list) else [],
            candidate_cases=candidates,
            run_id=run_id,
            created_at=created_at,
            dry_run=dry_run,
            replace_factory_cases=replace_factory_cases,
        )
        if not dry_run:
            backup = _backup_path(BENCHMARK_CASES_PATH, run_id)
            _write_json(BENCHMARK_CASES_PATH, final_cases)
            row["wrote"] = True
            row["backupPath"] = _relative(backup)
            written.append({"target": "benchmark_cases", "path": row["targetPath"], "backupPath": row["backupPath"]})
        rows.append(row)
        skipped.extend(target_skipped)
        benchmark_added = final_cases[-row["addedCount"] :] if row["addedCount"] else []

    if include_speech:
        existing = _load_json(SPEECH_CASES_PATH, [])
        candidates = [case for case in manifest.get("readySpeechCases") or [] if isinstance(case, dict)]
        final_cases, row, target_skipped = _plan_cases(
            target="speech_cases",
            target_path=SPEECH_CASES_PATH,
            existing_cases=existing if isinstance(existing, list) else [],
            candidate_cases=candidates,
            run_id=run_id,
            created_at=created_at,
            dry_run=dry_run,
            replace_factory_cases=replace_factory_cases,
        )
        if not dry_run:
            backup = _backup_path(SPEECH_CASES_PATH, run_id)
            _write_json(SPEECH_CASES_PATH, final_cases)
            row["wrote"] = True
            row["backupPath"] = _relative(backup)
            written.append({"target": "speech_cases", "path": row["targetPath"], "backupPath": row["backupPath"]})
        rows.append(row)
        skipped.extend(target_skipped)
        speech_added = final_cases[-row["addedCount"] :] if row["addedCount"] else []

    if include_audio_queue:
        prompts = [case for case in manifest.get("readyAudioRecordingPrompts") or [] if isinstance(case, dict)]
        audio_queue = [
            {
                **_json_copy(prompt),
                "promotion": {
                    "status": "recording_queue",
                    "promotedAt": created_at,
                    "promotionRunId": run_id,
                    "target": str(PROMOTED_AUDIO_QUEUE_PATH.relative_to(ROOT_DIR)),
                },
            }
            for prompt in prompts
        ]
        row = {
            "target": "audio_recording_queue",
            "targetPath": str(PROMOTED_AUDIO_QUEUE_PATH.relative_to(ROOT_DIR)),
            "existingCount": len(_load_json(PROMOTED_AUDIO_QUEUE_PATH, [])),
            "replacedFactoryCount": 0,
            "candidateCount": len(prompts),
            "addedCount": len(audio_queue),
            "skippedCount": 0,
            "finalCount": len(audio_queue),
            "dryRun": dry_run,
            "wrote": False,
            "backupPath": None,
        }
        if not dry_run:
            backup = _backup_path(PROMOTED_AUDIO_QUEUE_PATH, run_id)
            _write_json(PROMOTED_AUDIO_QUEUE_PATH, audio_queue)
            row["wrote"] = True
            row["backupPath"] = _relative(backup)
            written.append({"target": "audio_recording_queue", "path": row["targetPath"], "backupPath": row["backupPath"]})
        rows.append(row)

    summary = {
        "dryRun": dry_run,
        "replaceFactoryCases": replace_factory_cases,
        "blockedDrafts": len(blocked),
        "targets": len(rows),
        "totalCandidates": sum(int(row.get("candidateCount") or 0) for row in rows),
        "totalAdded": sum(int(row.get("addedCount") or 0) for row in rows),
        "totalSkipped": len(skipped),
        "benchmarkAdded": sum(int(row.get("addedCount") or 0) for row in rows if row.get("target") == "benchmark_cases"),
        "speechAdded": sum(int(row.get("addedCount") or 0) for row in rows if row.get("target") == "speech_cases"),
        "audioQueued": sum(int(row.get("addedCount") or 0) for row in rows if row.get("target") == "audio_recording_queue"),
        "wroteFiles": sum(1 for row in rows if row.get("wrote")),
        "readyToWrite": dry_run and not blocked and not skipped,
    }
    payload = {
        "found": True,
        "runId": run_id,
        "createdAt": created_at,
        "suite": "validated_suite_promotion",
        "summary": summary,
        "rows": rows,
        "skipped": skipped,
        "written": written,
        "blockedDrafts": blocked,
        "preview": {
            "benchmarkCases": benchmark_added[:12],
            "speechCases": speech_added[:12],
            "audioRecordingPrompts": audio_queue[:12],
        },
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(LATEST_CSV_PATH.relative_to(ROOT_DIR)),
            "promotedAudioQueue": str(PROMOTED_AUDIO_QUEUE_PATH.relative_to(ROOT_DIR)),
        },
    }
    if save:
        _write_json(LATEST_JSON_PATH, payload)
        _write_csv(rows, LATEST_CSV_PATH)
    return payload
