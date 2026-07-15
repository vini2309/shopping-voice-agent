from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio_eval import build_audio_dataset_manifest, load_latest_audio_manifest
from .statistics_pack import generate_statistics_pack, load_latest_statistics_pack


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "paper"
LATEST_JSON_PATH = ARTIFACT_DIR / "claims_latest.json"
LATEST_CSV_PATH = ARTIFACT_DIR / "claims_latest.csv"


CLAIMS: list[dict[str, Any]] = [
    {
        "id": "task_success_90",
        "section": "Task Success",
        "claim": "End-to-end task success is at least 90%.",
        "metricId": "benchmark.pass_rate",
        "comparator": "lower_bound_at_least",
        "target": 0.90,
        "minN": 50,
        "paperUse": "Primary task-success claim over curated shopping tasks",
    },
    {
        "id": "speech_success_90",
        "section": "Speech Robustness",
        "claim": "Speech proxy task success is at least 90%.",
        "metricId": "speech_proxy.pass_rate",
        "comparator": "lower_bound_at_least",
        "target": 0.90,
        "minN": 50,
        "paperUse": "Accent/noise transcript robustness claim",
    },
    {
        "id": "speech_wer_20",
        "section": "Speech Robustness",
        "claim": "Speech proxy mean WER is at most 20%.",
        "metricId": "speech_proxy.wer_mean",
        "comparator": "upper_bound_at_most",
        "target": 0.20,
        "minN": 50,
        "maxCiWidth": 0.10,
        "paperUse": "ASR-noise tolerance claim on transcript proxies",
    },
    {
        "id": "real_audio_success_90",
        "section": "Real Audio",
        "claim": "Accepted recorded real-audio task success is at least 90%.",
        "metricId": "accepted_audio.pass_rate",
        "comparator": "lower_bound_at_least",
        "target": 0.90,
        "minN": 30,
        "paperUse": "Provider-ASR task success over accepted recorded browser audio",
    },
    {
        "id": "real_audio_wer_15",
        "section": "Real Audio",
        "claim": "Accepted recorded real-audio mean WER is at most 15%.",
        "metricId": "accepted_audio.wer_mean",
        "comparator": "upper_bound_at_most",
        "target": 0.15,
        "minN": 30,
        "maxCiWidth": 0.06,
        "paperUse": "Provider-ASR quality claim on accepted actual recordings",
    },
    {
        "id": "semantic_transcript_success_90",
        "section": "Real Audio",
        "claim": "Real-audio semantic transcript preservation is at least 90%.",
        "metricId": "real_audio.semantic_transcript_pass_rate",
        "comparator": "lower_bound_at_least",
        "target": 0.90,
        "minN": 30,
        "paperUse": "Task-aware ASR preservation claim using intent, slots, canonical query, and downstream recovery",
    },
    {
        "id": "deepgram_p95_1000",
        "section": "Latency",
        "claim": "Accepted-audio Deepgram p95 transcription latency is below 1000 ms.",
        "metricId": "accepted_audio.deepgram_p95_ms",
        "comparator": "upper_bound_at_most",
        "target": 1000.0,
        "minN": 30,
        "maxCiWidth": 250.0,
        "paperUse": "Provider ASR tail-latency claim",
    },
    {
        "id": "voice_p95_2500",
        "section": "Latency",
        "claim": "Task benchmark p95 voice response latency is below 2500 ms.",
        "metricId": "benchmark.voice_p95_ms",
        "comparator": "upper_bound_at_most",
        "target": 2500.0,
        "minN": 50,
        "maxCiWidth": 500.0,
        "paperUse": "Stage-demo latency claim",
    },
    {
        "id": "cost_20_per_1k",
        "section": "Cost",
        "claim": "Composed stack cost is at most $20 per 1k turns.",
        "metricId": "combined.cost_per_1k_turns",
        "comparator": "upper_bound_at_most",
        "target": 20.0,
        "minN": 50,
        "maxCiWidth": 5.0,
        "paperUse": "Main cost-efficiency claim",
    },
    {
        "id": "openai_realtime_savings_50",
        "section": "Cost",
        "claim": "Composed stack saves at least 50% versus GPT Realtime.",
        "metricId": "combined.savings_vs_openai_realtime",
        "comparator": "lower_bound_at_least",
        "target": 0.50,
        "minN": 50,
        "maxCiWidth": 0.10,
        "paperUse": "Baseline cost comparison claim",
    },
    {
        "id": "gemini_live_not_more_expensive",
        "section": "Cost",
        "claim": "Composed stack is not more expensive than Gemini Live.",
        "metricId": "combined.savings_vs_gemini_live",
        "comparator": "lower_bound_at_least",
        "target": 0.0,
        "minN": 50,
        "maxCiWidth": 0.15,
        "paperUse": "Baseline cost comparison claim",
    },
    {
        "id": "audio_robustness_regression_10",
        "section": "Robustness",
        "claim": "Generated acoustic variants regress in at most 10% of paired trials.",
        "metricId": "audio_robustness.regression_rate",
        "comparator": "upper_bound_at_most",
        "target": 0.10,
        "minN": 30,
        "paperUse": "Acoustic stress robustness claim",
    },
]


def load_latest_claim_readiness() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No claim readiness pack saved yet."}
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


def _format_value(value: Any, unit: str) -> str:
    number = _number(value)
    if number is None:
        return "-"
    if unit == "rate":
        return f"{number * 100:.1f}%"
    if unit == "usd_per_1k_turns":
        return f"${number:.2f}"
    if unit == "ms":
        return f"{number:.1f} ms"
    if unit in {"wer", "delta_wer"}:
        return f"{number:.4f}"
    return f"{number:.4f}"


def _format_ci(metric: dict[str, Any]) -> str:
    lower = metric.get("ciLower")
    upper = metric.get("ciUpper")
    unit = str(metric.get("unit") or "")
    if lower is None or upper is None:
        return "-"
    return f"{_format_value(lower, unit)} to {_format_value(upper, unit)}"


def _metric_map(statistics_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = statistics_pack.get("metrics") if isinstance(statistics_pack, dict) else []
    return {str(row.get("id")): row for row in rows if isinstance(row, dict) and row.get("id")}


def _wilson_lower(successes: int, total: int, confidence: float) -> float:
    if total <= 0:
        return 0.0
    alpha = max(0.0001, min(0.9999, 1.0 - confidence))
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = successes / total
    denominator = 1.0 + (z * z / total)
    center = (phat + (z * z / (2.0 * total))) / denominator
    half_width = z * math.sqrt((phat * (1.0 - phat) / total) + (z * z / (4.0 * total * total))) / denominator
    return max(0.0, center - half_width)


def _wilson_upper(successes: int, total: int, confidence: float) -> float:
    if total <= 0:
        return 1.0
    alpha = max(0.0001, min(0.9999, 1.0 - confidence))
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = successes / total
    denominator = 1.0 + (z * z / total)
    center = (phat + (z * z / (2.0 * total))) / denominator
    half_width = z * math.sqrt((phat * (1.0 - phat) / total) + (z * z / (4.0 * total * total))) / denominator
    return min(1.0, center + half_width)


def _required_rate_samples(metric: dict[str, Any], claim: dict[str, Any], confidence: float) -> int | None:
    total = int(_number(metric.get("n")) or 0)
    if total <= 0:
        return int(claim.get("minN") or 0)
    raw_successes = int(_number(metric.get("successes")) or round((_number(metric.get("observed")) or 0.0) * total))
    successes = min(total, max(0, raw_successes))
    target = float(claim["target"])
    min_n = int(claim.get("minN") or 0)
    comparator = str(claim.get("comparator"))
    for extra in range(0, 2001):
        candidate_total = total + extra
        if candidate_total < min_n:
            continue
        candidate_successes = min(candidate_total, successes + extra)
        if comparator == "lower_bound_at_least" and _wilson_lower(candidate_successes, candidate_total, confidence) >= target:
            return extra
        if comparator == "upper_bound_at_most" and _wilson_upper(successes, candidate_total, confidence) <= target:
            return extra
    return None


def _estimated_continuous_samples(metric: dict[str, Any], claim: dict[str, Any]) -> int:
    n = int(_number(metric.get("n")) or 0)
    min_n = int(claim.get("minN") or 0)
    width = _number(metric.get("ciWidth"))
    max_width = _number(claim.get("maxCiWidth"))
    required_n = min_n
    if n > 1 and width and max_width and width > max_width:
        required_n = max(required_n, math.ceil(n * (width / max_width) ** 2))
    return max(0, required_n - n)


def _passes_metric(metric: dict[str, Any], claim: dict[str, Any]) -> tuple[bool, bool]:
    comparator = str(claim.get("comparator"))
    target = float(claim["target"])
    observed = _number(metric.get("observed"))
    lower = _number(metric.get("ciLower"))
    upper = _number(metric.get("ciUpper"))
    if comparator == "lower_bound_at_least":
        return (observed is not None and observed >= target, lower is not None and lower >= target)
    if comparator == "upper_bound_at_most":
        return (observed is not None and observed <= target, upper is not None and upper <= target)
    return (False, False)


def _claim_from_metric(claim: dict[str, Any], metric: dict[str, Any] | None, confidence: float) -> dict[str, Any]:
    if not metric or metric.get("status") != "ok":
        min_n = int(claim.get("minN") or 0)
        return {
            **claim,
            "status": "missing_evidence",
            "severity": 3,
            "observed": None,
            "ciLower": None,
            "ciUpper": None,
            "n": 0,
            "unit": None,
            "displayObserved": "-",
            "displayCi": "-",
            "additionalSamples": min_n,
            "nextAction": f"Collect and evaluate at least {min_n} samples for this claim.",
        }

    n = int(_number(metric.get("n")) or 0)
    min_n = int(claim.get("minN") or 0)
    observed_good, bound_good = _passes_metric(metric, claim)
    width = _number(metric.get("ciWidth"))
    max_width = _number(claim.get("maxCiWidth"))
    width_good = max_width is None or width is None or width <= max_width

    if not observed_good:
        status = "needs_system_work"
        severity = 2
        additional = _required_rate_samples(metric, claim, confidence) if metric.get("unit") == "rate" else _estimated_continuous_samples(metric, claim)
        next_action = "Improve the pipeline or revise the claim threshold, then rerun the suite."
    elif n < min_n or not bound_good or not width_good:
        status = "needs_more_data"
        severity = 1
        additional = _required_rate_samples(metric, claim, confidence) if metric.get("unit") == "rate" else _estimated_continuous_samples(metric, claim)
        if additional is None:
            next_action = "Collect more samples and rerun; this interval did not converge within the planner cap."
        elif additional > 0:
            next_action = f"Add about {additional} more passing/evaluable samples, then rerun statistics."
        else:
            next_action = "Rerun statistics after broadening the suite to reduce interval uncertainty."
    else:
        status = "publishable"
        severity = 0
        additional = 0
        next_action = "Ready to cite with the current evidence interval."

    return {
        **claim,
        "status": status,
        "severity": severity,
        "observed": metric.get("observed"),
        "ciLower": metric.get("ciLower"),
        "ciUpper": metric.get("ciUpper"),
        "ciWidth": metric.get("ciWidth"),
        "n": n,
        "unit": metric.get("unit"),
        "method": metric.get("method"),
        "displayObserved": metric.get("displayObserved") or _format_value(metric.get("observed"), str(metric.get("unit") or "")),
        "displayCi": metric.get("displayCi") or _format_ci(metric),
        "additionalSamples": additional,
        "nextAction": next_action,
    }


def _manifest_claim(audio_manifest: dict[str, Any]) -> dict[str, Any]:
    summary = audio_manifest.get("summary") if isinstance(audio_manifest.get("summary"), dict) else {}
    coverage = _number(summary.get("coverageRate")) or 0.0
    recordings = int(_number(summary.get("recordingCount")) or 0)
    missing = int(_number(summary.get("missingRecordings")) or 0)
    target = 0.80
    status = "publishable" if coverage >= target and missing == 0 else "needs_more_data"
    additional = 0 if status == "publishable" else max(missing, 1)
    return {
        "id": "audio_dataset_coverage_80",
        "section": "Real Audio",
        "claim": "Real-audio dataset coverage is at least 80% of the target manifest.",
        "metricId": "audio_manifest.coverage_rate",
        "comparator": "observed_at_least",
        "target": target,
        "status": status,
        "severity": 0 if status == "publishable" else 1,
        "observed": round(coverage, 4),
        "ciLower": None,
        "ciUpper": None,
        "n": recordings,
        "unit": "rate",
        "displayObserved": _format_value(coverage, "rate"),
        "displayCi": "-",
        "additionalSamples": additional,
        "paperUse": "Recorded-audio dataset sufficiency claim",
        "nextAction": "Coverage target met." if status == "publishable" else f"Record about {additional} more prompt fixtures to meet the manifest target.",
    }


def _statistics_coverage_claim(statistics_pack: dict[str, Any]) -> dict[str, Any]:
    summary = statistics_pack.get("summary") if isinstance(statistics_pack.get("summary"), dict) else {}
    coverage = _number(summary.get("coverageRate")) or 0.0
    missing = int(_number(summary.get("missingMetricCount")) or 0)
    target = 0.90
    status = "publishable" if coverage >= target and missing == 0 else "needs_more_data"
    return {
        "id": "evidence_metric_coverage_90",
        "section": "Evidence Completeness",
        "claim": "At least 90% of planned evidence metrics are populated.",
        "metricId": "statistics.coverage_rate",
        "comparator": "observed_at_least",
        "target": target,
        "status": status,
        "severity": 0 if status == "publishable" else 1,
        "observed": round(coverage, 4),
        "ciLower": None,
        "ciUpper": None,
        "n": int(_number(summary.get("metricCount")) or 0),
        "unit": "rate",
        "displayObserved": _format_value(coverage, "rate"),
        "displayCi": "-",
        "additionalSamples": missing,
        "paperUse": "Evidence coverage sanity check",
        "nextAction": "Evidence matrix is populated." if status == "publishable" else "Populate missing robustness/audio metrics before submission.",
    }


def _summary(claims: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "publishable": 0,
        "needsMoreData": 0,
        "needsSystemWork": 0,
        "missingEvidence": 0,
    }
    for claim in claims:
        status = claim.get("status")
        if status == "publishable":
            counts["publishable"] += 1
        elif status == "needs_more_data":
            counts["needsMoreData"] += 1
        elif status == "needs_system_work":
            counts["needsSystemWork"] += 1
        elif status == "missing_evidence":
            counts["missingEvidence"] += 1
    total = len(claims)
    weighted = (
        counts["publishable"] * 1.0
        + counts["needsMoreData"] * 0.55
        + counts["needsSystemWork"] * 0.20
    )
    blockers = [claim for claim in claims if claim.get("status") in {"needs_system_work", "missing_evidence"}]
    data_claims = [claim for claim in claims if claim.get("status") == "needs_more_data"]
    top_action = None
    if blockers:
        top_action = blockers[0].get("nextAction")
    elif data_claims:
        top_action = data_claims[0].get("nextAction")
    return {
        "totalClaims": total,
        **counts,
        "paperReady": counts["publishable"] == total and total > 0,
        "claimReadinessScore": round((weighted / total) * 100, 1) if total else 0.0,
        "additionalSamplesRecommended": sum(int(_number(claim.get("additionalSamples")) or 0) for claim in claims if claim.get("status") != "publishable"),
        "topAction": top_action or "All tracked claims are ready to cite.",
    }


def _action_plan(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [claim for claim in claims if claim.get("status") != "publishable"]
    rows.sort(key=lambda claim: (-int(claim.get("severity") or 0), str(claim.get("section")), str(claim.get("id"))))
    return [
        {
            "claimId": claim.get("id"),
            "section": claim.get("section"),
            "status": claim.get("status"),
            "additionalSamples": claim.get("additionalSamples"),
            "nextAction": claim.get("nextAction"),
        }
        for claim in rows[:12]
    ]


def _write_csv(claims: list[dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "section",
        "claim",
        "status",
        "metricId",
        "target",
        "observed",
        "ciLower",
        "ciUpper",
        "ciWidth",
        "n",
        "unit",
        "method",
        "additionalSamples",
        "nextAction",
        "paperUse",
    ]
    with LATEST_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(claims)


def generate_claim_readiness_pack(*, regenerate_statistics: bool = False, save: bool = True) -> dict[str, Any]:
    statistics_pack = generate_statistics_pack(save=True) if regenerate_statistics else load_latest_statistics_pack()
    if statistics_pack.get("found") is False:
        statistics_pack = generate_statistics_pack(save=True)
    audio_manifest = load_latest_audio_manifest()
    if audio_manifest.get("found") is False:
        audio_manifest = build_audio_dataset_manifest(save=True)

    confidence = float(((statistics_pack.get("summary") or {}).get("confidence")) or 0.95)
    metrics = _metric_map(statistics_pack)
    claims = [_claim_from_metric(claim, metrics.get(str(claim["metricId"])), confidence) for claim in CLAIMS]
    claims.append(_manifest_claim(audio_manifest))
    claims.append(_statistics_coverage_claim(statistics_pack))
    claims.sort(key=lambda item: (-int(item.get("severity") or 0), str(item.get("section")), str(item.get("id"))))

    summary = _summary(claims)
    payload = {
        "found": True,
        "runId": datetime.now(timezone.utc).strftime("claims-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suite": "paper_claim_readiness_gate",
        "summary": summary,
        "claims": claims,
        "actionPlan": _action_plan(claims),
        "inputs": {
            "statisticsRunId": statistics_pack.get("runId"),
            "audioManifestRunId": audio_manifest.get("runId"),
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
        _write_csv(claims)
    return payload
