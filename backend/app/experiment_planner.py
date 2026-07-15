from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .audio_eval import build_audio_dataset_manifest, load_latest_audio_manifest, load_latest_audio_robustness
from .benchmark_suite import load_benchmark_cases
from .claim_readiness import generate_claim_readiness_pack, load_latest_claim_readiness
from .speech_eval import load_speech_cases
from .statistics_pack import generate_statistics_pack, load_latest_statistics_pack


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "paper"
LATEST_JSON_PATH = ARTIFACT_DIR / "experiment_plan_latest.json"
LATEST_CSV_PATH = ARTIFACT_DIR / "experiment_plan_latest.csv"


BENCHMARK_BLUEPRINTS = [
    {"group": "inventory_exact", "weight": 5, "purpose": "Exact item, aisle, bay, and stock checks"},
    {"group": "category_inventory", "weight": 5, "purpose": "Open category questions such as dog food or paper goods"},
    {"group": "recommendation", "weight": 4, "purpose": "Best-option questions grounded in reviews and stock"},
    {"group": "policy_grounding", "weight": 6, "purpose": "RAG policy/SOP answers with citations"},
    {"group": "asr_noisy_inventory", "weight": 4, "purpose": "Misspellings and ASR-like item substitutions"},
    {"group": "adversarial_policy", "weight": 3, "purpose": "Prompt-injection and unsafe-policy requests"},
    {"group": "multi_tool_grounding", "weight": 3, "purpose": "Questions requiring both inventory and policy evidence"},
]

SPEECH_STRATA = [
    {"condition": "reference_us|clean|barge:false", "weight": 4, "purpose": "Clean reference speech"},
    {"condition": "indian_english_proxy|clean|barge:false", "weight": 4, "purpose": "Indian English pronunciation proxies"},
    {"condition": "spanish_l1_proxy|checkout_beeps_proxy|barge:false", "weight": 4, "purpose": "L1 Spanish accent and checkout noise"},
    {"condition": "fast_speech_proxy|store_ambient_proxy|barge:false", "weight": 4, "purpose": "Fast speech with ambient store noise"},
    {"condition": "low_volume_proxy|room|barge:false", "weight": 3, "purpose": "Low volume and clipped ASR tokens"},
    {"condition": "reference_us|clean|barge:true", "weight": 2, "purpose": "Barge-in follow-up turns"},
]

AUDIO_STRATA = [
    "reference_us|room|browser_mic",
    "indian_english|room|browser_mic",
    "spanish_l1|room|browser_mic",
    "reference_us|store_noise|browser_mic",
    "fast_speech|room|browser_mic",
]

AUGMENTATION_MATRIX = [
    {"augmentationType": "store_noise", "label": "Synthetic store noise"},
    {"augmentationType": "low_volume", "label": "Low volume"},
    {"augmentationType": "fast_speech", "label": "Fast speech"},
    {"augmentationType": "slow_speech", "label": "Slow speech"},
    {"augmentationType": "clipped_audio", "label": "Mild clipping"},
]


def load_latest_experiment_plan() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No experiment plan saved yet."}
    with LATEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _safe_count(loader: Callable[[], list[dict[str, Any]]]) -> int:
    try:
        rows = loader()
    except Exception:
        return 0
    return len(rows) if isinstance(rows, list) else 0


def _summary_value(payload: dict[str, Any], key: str, default: Any = 0) -> Any:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return summary.get(key, default)


def _claims(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("claims") if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _matching_claims(claims: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    return [claim for claim in claims if claim.get("status") != "publishable" and predicate(claim)]


def _max_additional(claims: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> int:
    values = [
        int(_number(claim.get("additionalSamples")) or 0)
        for claim in _matching_claims(claims, predicate)
    ]
    return max(values) if values else 0


def _claim_ids(claims: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[str]:
    return [str(claim.get("id")) for claim in _matching_claims(claims, predicate) if claim.get("id")]


def _allocate(total: int, blueprints: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    if total <= 0 or not blueprints:
        return []
    weight_sum = sum(int(row.get("weight") or 1) for row in blueprints)
    allocations: list[dict[str, Any]] = []
    assigned = 0
    for row in blueprints:
        raw = total * (int(row.get("weight") or 1) / weight_sum)
        count = int(math.floor(raw))
        allocations.append({**row, key: count, "_fraction": raw - count})
        assigned += count
    remaining = total - assigned
    for row in sorted(allocations, key=lambda item: item["_fraction"], reverse=True)[:remaining]:
        row[key] += 1
    for row in allocations:
        row.pop("_fraction", None)
    return allocations


def _prompt_rows(audio_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = audio_manifest.get("promptCoverage") if isinstance(audio_manifest, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _recording_queue(audio_manifest: dict[str, Any], target_additions: int, source_claim_ids: list[str]) -> list[dict[str, Any]]:
    prompts = _prompt_rows(audio_manifest)
    if target_additions <= 0 or not prompts:
        return []

    queue: list[dict[str, Any]] = []
    remaining = target_additions
    for prompt in sorted(prompts, key=lambda item: int(_number(item.get("missing")) or 0), reverse=True):
        missing = int(_number(prompt.get("missing")) or 0)
        if missing <= 0 or remaining <= 0:
            continue
        add = min(missing, remaining)
        queue.append(_recording_item(prompt, add, "Complete manifest target", source_claim_ids))
        remaining -= add

    extras: dict[str, dict[str, Any]] = {}
    ordered = sorted(prompts, key=lambda item: (int(_number(item.get("recordings")) or 0), str(item.get("templateId"))))
    index = 0
    while remaining > 0 and ordered:
        prompt = ordered[index % len(ordered)]
        key = str(prompt.get("templateId") or f"prompt-{index}")
        item = extras.setdefault(key, _recording_item(prompt, 0, "Expand replication for confidence interval", source_claim_ids))
        item["additionalRecordings"] += 1
        item["targetRecordings"] += 1
        remaining -= 1
        index += 1

    return [*queue, *extras.values()]


def _recording_item(prompt: dict[str, Any], add: int, reason: str, source_claim_ids: list[str]) -> dict[str, Any]:
    current = int(_number(prompt.get("recordings")) or 0)
    strata = [AUDIO_STRATA[(current + offset) % len(AUDIO_STRATA)] for offset in range(max(add, 1))]
    return {
        "id": f"record-{prompt.get('templateId') or 'prompt'}-{reason.lower().replace(' ', '-')}",
        "templateId": prompt.get("templateId"),
        "referenceText": prompt.get("referenceText"),
        "route": prompt.get("route"),
        "group": prompt.get("group"),
        "currentRecordings": current,
        "additionalRecordings": add,
        "targetRecordings": current + add,
        "reason": reason,
        "recommendedStrata": strata[:add],
        "sourceClaimIds": source_claim_ids,
    }


def _stress_matrix(target_pairs: int, source_claim_ids: list[str]) -> list[dict[str, Any]]:
    if target_pairs <= 0:
        return []
    allocations = _allocate(target_pairs, [{**row, "weight": 1} for row in AUGMENTATION_MATRIX], "targetPairs")
    return [
        {
            "id": f"stress-{row['augmentationType']}",
            "augmentationType": row["augmentationType"],
            "augmentationLabel": row["label"],
            "targetPairs": row["targetPairs"],
            "sourceClaimIds": source_claim_ids,
        }
        for row in allocations
    ]


def _work_item(
    *,
    item_id: str,
    lane: str,
    priority: int,
    action: str,
    add_count: int,
    reason: str,
    source_claim_ids: list[str],
    command: str | None = None,
    paper_use: str,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "lane": lane,
        "priority": priority,
        "action": action,
        "addCount": max(0, int(add_count)),
        "reason": reason,
        "sourceClaimIds": source_claim_ids,
        "command": command,
        "paperUse": paper_use,
    }


def _phase_rows(work_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phases = [
        {
            "phase": 1,
            "name": "Expand no-provider suites",
            "lanes": ["benchmark_text", "speech_proxy"],
            "purpose": "Increase task and transcript proxy sample size without provider spend.",
        },
        {
            "phase": 2,
            "name": "Collect real audio",
            "lanes": ["real_audio"],
            "purpose": "Record and evaluate browser-mic audio across prompts and speaker/noise strata.",
        },
        {
            "phase": 3,
            "name": "Stress acoustic robustness",
            "lanes": ["audio_robustness"],
            "purpose": "Generate paired variants and measure degradation against originals.",
        },
        {
            "phase": 4,
            "name": "Recompute paper evidence",
            "lanes": ["analysis", "system_work"],
            "purpose": "Rerun statistics, claim readiness, and paper export.",
        },
    ]
    for phase in phases:
        phase["workItemCount"] = sum(1 for item in work_items if item.get("lane") in phase["lanes"])
        phase["plannedSamples"] = sum(int(item.get("addCount") or 0) for item in work_items if item.get("lane") in phase["lanes"])
    return phases


def _write_csv(work_items: list[dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "lane", "priority", "action", "addCount", "reason", "sourceClaimIds", "command", "paperUse"]
    with LATEST_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in work_items:
            writer.writerow({**item, "sourceClaimIds": ";".join(item.get("sourceClaimIds") or [])})


def generate_experiment_plan(*, refresh_claims: bool = False, save: bool = True) -> dict[str, Any]:
    if refresh_claims:
        statistics_pack = generate_statistics_pack(save=True)
        claim_pack = generate_claim_readiness_pack(regenerate_statistics=False, save=True)
    else:
        statistics_pack = load_latest_statistics_pack()
        if statistics_pack.get("found") is False:
            statistics_pack = generate_statistics_pack(save=True)
        claim_pack = load_latest_claim_readiness()
        if claim_pack.get("found") is False:
            claim_pack = generate_claim_readiness_pack(save=True)

    audio_manifest = load_latest_audio_manifest()
    if audio_manifest.get("found") is False:
        audio_manifest = build_audio_dataset_manifest(save=True)
    audio_robustness = load_latest_audio_robustness()

    claims = _claims(claim_pack)
    benchmark_needed = _max_additional(claims, lambda claim: str(claim.get("metricId", "")).startswith("benchmark."))
    speech_needed = _max_additional(claims, lambda claim: str(claim.get("metricId", "")).startswith("speech_proxy."))
    real_audio_needed = _max_additional(
        claims,
        lambda claim: str(claim.get("metricId", "")).startswith("real_audio.") or str(claim.get("metricId", "")).startswith("audio_manifest.") or str(claim.get("metricId", "")).startswith("statistics.") or str(claim.get("id", "")).startswith("deepgram_"),
    )
    stress_needed = _max_additional(claims, lambda claim: str(claim.get("metricId", "")).startswith("audio_robustness."))
    combined_needed = _max_additional(claims, lambda claim: str(claim.get("metricId", "")).startswith("combined."))
    combined_covered = benchmark_needed + speech_needed + real_audio_needed
    combined_extra = max(0, combined_needed - combined_covered)

    system_work_claim_ids = _claim_ids(claims, lambda claim: claim.get("status") == "needs_system_work")
    benchmark_claim_ids = _claim_ids(claims, lambda claim: str(claim.get("metricId", "")).startswith("benchmark."))
    speech_claim_ids = _claim_ids(claims, lambda claim: str(claim.get("metricId", "")).startswith("speech_proxy."))
    audio_claim_ids = _claim_ids(
        claims,
        lambda claim: str(claim.get("metricId", "")).startswith("real_audio.") or str(claim.get("metricId", "")).startswith("audio_manifest.") or str(claim.get("id", "")).startswith("deepgram_"),
    )
    stress_claim_ids = _claim_ids(claims, lambda claim: str(claim.get("metricId", "")).startswith("audio_robustness."))
    combined_claim_ids = _claim_ids(claims, lambda claim: str(claim.get("metricId", "")).startswith("combined."))

    work_items: list[dict[str, Any]] = []
    if system_work_claim_ids:
        work_items.append(
            _work_item(
                item_id="revise-cost-claim-or-optimize-stack",
                lane="system_work",
                priority=100,
                action="Revise the Gemini Live cost claim or change the stack assumptions before citing it.",
                add_count=0,
                reason="Current Gemini comparison interval is below the not-more-expensive threshold.",
                source_claim_ids=system_work_claim_ids,
                paper_use="Prevents overclaiming against the cheapest baseline.",
            )
        )
    if benchmark_needed:
        work_items.append(
            _work_item(
                item_id="expand-task-benchmark-suite",
                lane="benchmark_text",
                priority=80,
                action=f"Add {benchmark_needed} curated text benchmark cases across inventory, RAG, recommendation, and adversarial groups.",
                add_count=benchmark_needed,
                reason="Task-success and voice-latency claims need larger sample support.",
                source_claim_ids=benchmark_claim_ids,
                command="Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/suite/run -Method Post -ContentType 'application/json' -Body (@{ groups=@(); limit=100; includePayloads=$false; save=$true } | ConvertTo-Json)",
                paper_use="Raises task benchmark N for confidence-bound claims.",
            )
        )
    if speech_needed:
        work_items.append(
            _work_item(
                item_id="expand-speech-proxy-suite",
                lane="speech_proxy",
                priority=75,
                action=f"Add {speech_needed} transcript-proxy speech cases across accent, noise, speed, and barge-in strata.",
                add_count=speech_needed,
                reason="Speech success and WER claims need a broader proxy suite.",
                source_claim_ids=speech_claim_ids,
                command="Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/speech/run -Method Post -ContentType 'application/json' -Body (@{ groups=@(); conditions=@(); limit=100; save=$true } | ConvertTo-Json)",
                paper_use="Improves ASR-noise robustness evidence without provider spend.",
            )
        )
    if real_audio_needed:
        work_items.append(
            _work_item(
                item_id="collect-real-audio-recordings",
                lane="real_audio",
                priority=90,
                action=f"Record and evaluate {real_audio_needed} additional real-audio fixtures.",
                add_count=real_audio_needed,
                reason="Real-audio success, WER, Deepgram latency, and dataset coverage claims need more evaluated recordings.",
                source_claim_ids=audio_claim_ids,
                command="Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/audio/run -Method Post -ContentType 'application/json' -Body (@{ caseIds=@(); limit=100; allowReferenceFallback=$false; save=$true } | ConvertTo-Json)",
                paper_use="Turns the demo from transcript proxies into recorded provider-ASR evidence.",
            )
        )
    if stress_needed:
        work_items.append(
            _work_item(
                item_id="evaluate-audio-stress-variants",
                lane="audio_robustness",
                priority=95,
                action=f"Generate and evaluate {stress_needed} paired acoustic stress variants.",
                add_count=stress_needed,
                reason="Audio robustness regression claims have no paired variant evidence yet.",
                source_claim_ids=stress_claim_ids,
                command="Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/audio/robustness -Method Post -ContentType 'application/json' -Body (@{ save=$true } | ConvertTo-Json)",
                paper_use="Measures degradation under controlled acoustic perturbations.",
            )
        )
    if combined_extra:
        work_items.append(
            _work_item(
                item_id="add-combined-cost-validation-turns",
                lane="analysis",
                priority=40,
                action=f"Add {combined_extra} extra turns if other expansions do not cover combined cost confidence.",
                add_count=combined_extra,
                reason="Combined cost baseline claims require more total turns.",
                source_claim_ids=combined_claim_ids,
                paper_use="Improves cost CI stability.",
            )
        )
    work_items.append(
        _work_item(
            item_id="rerun-paper-evidence-pack",
            lane="analysis",
            priority=30,
            action="Rerun statistics, claim readiness, and paper report after collecting the planned samples.",
            add_count=0,
            reason="Evidence artifacts need to match the latest experiment runs.",
            source_claim_ids=[claim.get("id") for claim in claims if claim.get("status") != "publishable" and claim.get("id")],
            command="Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/report/run -Method Post -ContentType 'application/json' -Body (@{ rerunSuites=$false; save=$true } | ConvertTo-Json)",
            paper_use="Refreshes all paper tables and claim gates.",
        )
    )

    recording_queue = _recording_queue(audio_manifest, real_audio_needed, audio_claim_ids)
    benchmark_blueprints = _allocate(benchmark_needed, BENCHMARK_BLUEPRINTS, "targetCases")
    speech_blueprints = _allocate(speech_needed, SPEECH_STRATA, "targetCases")
    stress_matrix = _stress_matrix(stress_needed, stress_claim_ids)

    planned_samples = benchmark_needed + speech_needed + real_audio_needed + stress_needed + combined_extra
    raw_recommendation = int(_summary_value(claim_pack, "additionalSamplesRecommended", planned_samples) or 0)
    summary = {
        "plannedSamples": planned_samples,
        "rawClaimRecommendedSamples": raw_recommendation,
        "deduplicationSavings": max(0, raw_recommendation - planned_samples),
        "benchmarkCasesToAdd": benchmark_needed,
        "speechProxyCasesToAdd": speech_needed,
        "realAudioRecordingsToAdd": real_audio_needed,
        "stressPairsToEvaluate": stress_needed,
        "providerEvalCallsNeeded": real_audio_needed + stress_needed,
        "systemWorkItems": len(system_work_claim_ids),
        "workItemCount": len(work_items),
        "currentBenchmarkCases": _safe_count(load_benchmark_cases),
        "currentSpeechCases": _safe_count(load_speech_cases),
        "currentAudioRecordings": _summary_value(audio_manifest, "recordingCount", 0),
        "currentAudioEvaluated": _summary_value(audio_manifest, "evaluatedRecordings", 0),
        "currentRobustnessCompared": _summary_value(audio_robustness, "comparedCount", 0),
        "claimReadinessScore": _summary_value(claim_pack, "claimReadinessScore", 0),
    }

    payload = {
        "found": True,
        "runId": datetime.now(timezone.utc).strftime("plan-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suite": "paper_experiment_planner",
        "summary": summary,
        "phases": _phase_rows(work_items),
        "workItems": sorted(work_items, key=lambda item: (-int(item.get("priority") or 0), str(item.get("id")))),
        "recordingQueue": recording_queue,
        "benchmarkBlueprints": benchmark_blueprints,
        "speechBlueprints": speech_blueprints,
        "stressMatrix": stress_matrix,
        "methodNotes": [
            "Counts are deduplicated across claims so one new sample can satisfy multiple confidence-bound gaps.",
            "Real-audio recommendations fill missing manifest prompts first, then spread extra repetitions across prompts and speaker/noise strata.",
            "Stress-pair recommendations assume generated variants are evaluated through the same Deepgram and downstream scoring path as originals.",
        ],
        "inputs": {
            "claimRunId": claim_pack.get("runId"),
            "statisticsRunId": statistics_pack.get("runId"),
            "audioManifestRunId": audio_manifest.get("runId"),
            "audioRobustnessRunId": audio_robustness.get("runId"),
        },
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(LATEST_CSV_PATH.relative_to(ROOT_DIR)),
        },
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with LATEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _write_csv(payload["workItems"])
    return payload
