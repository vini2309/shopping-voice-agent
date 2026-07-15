from __future__ import annotations

import csv
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .audio_accepted import load_latest_audio_accepted_set
from .audio_eval import load_latest_audio_eval, load_latest_audio_robustness
from .benchmark_suite import load_latest_benchmark
from .speech_eval import load_latest_speech_eval


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "paper"
LATEST_JSON_PATH = ARTIFACT_DIR / "statistics_latest.json"
LATEST_CSV_PATH = ARTIFACT_DIR / "statistics_latest.csv"
DEFAULT_SEED = 20260706


StatisticFn = Callable[[list[float]], float | None]


def load_latest_statistics_pack() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No statistics pack saved yet."}
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


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _p95(values: list[float]) -> float | None:
    return _percentile(values, 0.95)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    clipped = min(max(probability, 0.0), 1.0)
    return _percentile(sorted(values), clipped)


def _round(value: Any, digits: int = 6) -> float | None:
    number = _number(value)
    if number is None:
        return None
    return round(number, digits)


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
    if unit in {"wer", "delta_wer", "ratio"}:
        return f"{number:.4f}"
    return f"{number:.4f}"


def _format_ci(lower: Any, upper: Any, unit: str) -> str:
    if lower is None or upper is None:
        return "-"
    return f"{_format_value(lower, unit)} to {_format_value(upper, unit)}"


def _results(payload: dict[str, Any], *, include_skipped: bool = False) -> list[dict[str, Any]]:
    rows = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and (include_skipped or not row.get("skipped"))]


def _numeric_values(rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], Any]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _number(selector(row))
        if value is not None:
            values.append(value)
    return values


def _cost_value(row: dict[str, Any], key: str) -> float | None:
    cost = row.get("cost") if isinstance(row.get("cost"), dict) else {}
    return _number(cost.get(key))


def _proportion_metric(
    *,
    metric_id: str,
    label: str,
    source: str,
    values: list[float],
    confidence: float,
    paper_use: str,
    direction: str = "higher_is_better",
) -> dict[str, Any]:
    n = len(values)
    if not n:
        return _missing_metric(metric_id, label, source, "rate", "wilson_score", paper_use, direction)
    successes = sum(1 for value in values if value >= 0.5)
    phat = successes / n
    alpha = max(0.0001, min(0.9999, 1.0 - confidence))
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    denominator = 1.0 + (z * z / n)
    center = (phat + (z * z / (2.0 * n))) / denominator
    half_width = z * math.sqrt((phat * (1.0 - phat) / n) + (z * z / (4.0 * n * n))) / denominator
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return _metric_payload(
        metric_id=metric_id,
        label=label,
        source=source,
        values=values,
        observed=phat,
        lower=lower,
        upper=upper,
        confidence=confidence,
        iterations=0,
        unit="rate",
        method="wilson_score",
        paper_use=paper_use,
        direction=direction,
        successes=successes,
    )


def _bootstrap_metric(
    *,
    metric_id: str,
    label: str,
    source: str,
    values: list[float],
    statistic_fn: StatisticFn,
    statistic_name: str,
    unit: str,
    confidence: float,
    iterations: int,
    rng: random.Random,
    paper_use: str,
    direction: str,
) -> dict[str, Any]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return _missing_metric(metric_id, label, source, unit, f"bootstrap_{statistic_name}", paper_use, direction)
    observed = statistic_fn(clean)
    if observed is None:
        return _missing_metric(metric_id, label, source, unit, f"bootstrap_{statistic_name}", paper_use, direction)
    if len(clean) == 1 or iterations <= 0:
        lower = upper = observed
    else:
        samples: list[float] = []
        n = len(clean)
        for _ in range(iterations):
            sample = [clean[rng.randrange(n)] for _ in range(n)]
            value = statistic_fn(sample)
            if value is not None and math.isfinite(value):
                samples.append(value)
        alpha = max(0.0001, min(0.9999, 1.0 - confidence))
        lower = _quantile(samples, alpha / 2.0)
        upper = _quantile(samples, 1.0 - alpha / 2.0)
    return _metric_payload(
        metric_id=metric_id,
        label=label,
        source=source,
        values=clean,
        observed=observed,
        lower=lower,
        upper=upper,
        confidence=confidence,
        iterations=iterations,
        unit=unit,
        method=f"bootstrap_{statistic_name}",
        paper_use=paper_use,
        direction=direction,
    )


def _missing_metric(metric_id: str, label: str, source: str, unit: str, method: str, paper_use: str, direction: str) -> dict[str, Any]:
    return {
        "id": metric_id,
        "label": label,
        "source": source,
        "n": 0,
        "observed": None,
        "ciLower": None,
        "ciUpper": None,
        "ciWidth": None,
        "confidence": None,
        "iterations": None,
        "unit": unit,
        "method": method,
        "direction": direction,
        "status": "missing_data",
        "displayObserved": "-",
        "displayCi": "-",
        "paperUse": paper_use,
    }


def _metric_payload(
    *,
    metric_id: str,
    label: str,
    source: str,
    values: list[float],
    observed: float,
    lower: float | None,
    upper: float | None,
    confidence: float,
    iterations: int,
    unit: str,
    method: str,
    paper_use: str,
    direction: str,
    successes: int | None = None,
) -> dict[str, Any]:
    lower_r = _round(lower)
    upper_r = _round(upper)
    observed_r = _round(observed)
    width = None if lower_r is None or upper_r is None else round(upper_r - lower_r, 6)
    payload = {
        "id": metric_id,
        "label": label,
        "source": source,
        "n": len(values),
        "observed": observed_r,
        "ciLower": lower_r,
        "ciUpper": upper_r,
        "ciWidth": width,
        "confidence": round(confidence, 4),
        "iterations": iterations,
        "unit": unit,
        "method": method,
        "direction": direction,
        "status": "ok",
        "displayObserved": _format_value(observed_r, unit),
        "displayCi": _format_ci(lower_r, upper_r, unit),
        "paperUse": paper_use,
    }
    if successes is not None:
        payload["successes"] = successes
    return payload


def _artifact_payload(loader: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        payload = loader()
    except Exception as exc:  # pragma: no cover - defensive artifact loader
        return {"found": False, "message": str(exc), "results": []}
    return payload if isinstance(payload, dict) else {"found": False, "results": []}


def _add_result_metrics(
    metrics: list[dict[str, Any]],
    *,
    prefix: str,
    source: str,
    rows: list[dict[str, Any]],
    confidence: float,
    iterations: int,
    rng: random.Random,
    include_wer: bool,
    include_entity: bool,
    include_deepgram: bool = False,
) -> None:
    metrics.append(
        _proportion_metric(
            metric_id=f"{prefix}.pass_rate",
            label=f"{source} pass rate",
            source=source,
            values=_numeric_values(rows, lambda row: row.get("passed")),
            confidence=confidence,
            paper_use="End-to-end task success with uncertainty bounds",
        )
    )
    metrics.append(
        _bootstrap_metric(
            metric_id=f"{prefix}.voice_p95_ms",
            label=f"{source} voice p95 latency",
            source=source,
            values=_numeric_values(rows, lambda row: row.get("estimatedVoiceLatencyMs")),
            statistic_fn=_p95,
            statistic_name="p95",
            unit="ms",
            confidence=confidence,
            iterations=iterations,
            rng=rng,
            paper_use="Tail latency estimate for spoken response start",
            direction="lower_is_better",
        )
    )
    metrics.append(
        _bootstrap_metric(
            metric_id=f"{prefix}.cost_per_1k_turns",
            label=f"{source} cost per 1k turns",
            source=source,
            values=_numeric_values(rows, lambda row: (_cost_value(row, "vapiStackCost") or 0.0) * 1000.0 if _cost_value(row, "vapiStackCost") is not None else None),
            statistic_fn=_mean,
            statistic_name="mean",
            unit="usd_per_1k_turns",
            confidence=confidence,
            iterations=iterations,
            rng=rng,
            paper_use="Composed voice-stack cost estimate per 1k turns",
            direction="lower_is_better",
        )
    )
    if include_wer:
        metrics.append(
            _bootstrap_metric(
                metric_id=f"{prefix}.wer_mean",
                label=f"{source} mean WER",
                source=source,
                values=_numeric_values(rows, lambda row: row.get("wer")),
                statistic_fn=_mean,
                statistic_name="mean",
                unit="wer",
                confidence=confidence,
                iterations=iterations,
                rng=rng,
                paper_use="ASR robustness mean word error rate",
                direction="lower_is_better",
            )
        )
    if include_entity:
        metrics.append(
            _bootstrap_metric(
                metric_id=f"{prefix}.entity_recall_mean",
                label=f"{source} mean entity recall",
                source=source,
                values=_numeric_values(rows, lambda row: row.get("entityRecall")),
                statistic_fn=_mean,
                statistic_name="mean",
                unit="rate",
                confidence=confidence,
                iterations=iterations,
                rng=rng,
                paper_use="Entity preservation under noisy speech or transcription",
                direction="higher_is_better",
            )
        )
    if include_deepgram:
        metrics.append(
            _bootstrap_metric(
                metric_id=f"{prefix}.deepgram_p95_ms",
                label=f"{source} Deepgram p95 latency",
                source=source,
                values=_numeric_values(rows, lambda row: row.get("transcriptionLatencyMs")),
                statistic_fn=_p95,
                statistic_name="p95",
                unit="ms",
                confidence=confidence,
                iterations=iterations,
                rng=rng,
                paper_use="Provider ASR tail latency over recorded audio fixtures",
                direction="lower_is_better",
            )
        )


def _cost_savings_values(rows: list[dict[str, Any]], baseline_key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        ours = _cost_value(row, "vapiStackCost")
        baseline = _cost_value(row, baseline_key)
        if ours is not None and baseline and baseline > 0:
            values.append((baseline - ours) / baseline)
    return values


def _summary(metrics: list[dict[str, Any]], run_ids: dict[str, Any], confidence: float, iterations: int, seed: int) -> dict[str, Any]:
    populated = [metric for metric in metrics if metric.get("status") == "ok"]
    missing = len(metrics) - len(populated)
    widest = None
    for metric in populated:
        width = _number(metric.get("ciWidth"))
        if width is None:
            continue
        if widest is None or width > (_number(widest.get("ciWidth")) or -1):
            widest = metric
    return {
        "metricCount": len(metrics),
        "populatedMetricCount": len(populated),
        "missingMetricCount": missing,
        "coverageRate": round(len(populated) / len(metrics), 4) if metrics else 0.0,
        "confidence": round(confidence, 4),
        "iterations": iterations,
        "seed": seed,
        "runIds": run_ids,
        "widestCiMetric": {
            "id": widest.get("id"),
            "label": widest.get("label"),
            "ciWidth": widest.get("ciWidth"),
            "displayCi": widest.get("displayCi"),
        }
        if widest
        else None,
    }


def _interpretation(summary: dict[str, Any]) -> list[str]:
    notes = [
        "Use Wilson intervals for pass-rate claims and bootstrap intervals for continuous WER, latency, and cost claims.",
        "Intervals are computed from the latest saved benchmark artifacts, so rerun suites before final paper export.",
    ]
    if (summary.get("missingMetricCount") or 0) > 0:
        notes.append("Some intervals are missing because the corresponding saved suite has too few evaluated rows.")
    return notes


def _write_csv(metrics: list[dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "label",
        "source",
        "n",
        "successes",
        "observed",
        "ciLower",
        "ciUpper",
        "ciWidth",
        "confidence",
        "iterations",
        "unit",
        "method",
        "direction",
        "status",
        "displayObserved",
        "displayCi",
        "paperUse",
    ]
    with LATEST_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)


def generate_statistics_pack(
    *,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
    save: bool = True,
) -> dict[str, Any]:
    safe_iterations = max(100, min(int(iterations or 1000), 10000))
    safe_confidence = max(0.80, min(float(confidence or 0.95), 0.99))
    rng = random.Random(seed)

    benchmark = _artifact_payload(load_latest_benchmark)
    speech = _artifact_payload(load_latest_speech_eval)
    audio = _artifact_payload(load_latest_audio_eval)
    audio_accepted = _artifact_payload(load_latest_audio_accepted_set)
    robustness = _artifact_payload(load_latest_audio_robustness)

    benchmark_rows = _results(benchmark)
    speech_rows = _results(speech)
    audio_rows = _results(audio)
    accepted_audio_rows = [
        row
        for row in audio_accepted.get("acceptedResults") or []
        if isinstance(row, dict) and not row.get("skipped")
    ]
    robustness_rows = [row for row in robustness.get("rows") or [] if isinstance(row, dict) and row.get("baselineRecordingId") and row.get("verdict") != "skipped"]
    combined_audio_rows = accepted_audio_rows or audio_rows
    combined_rows = [*benchmark_rows, *speech_rows, *combined_audio_rows]

    metrics: list[dict[str, Any]] = []
    _add_result_metrics(
        metrics,
        prefix="benchmark",
        source="Task benchmark",
        rows=benchmark_rows,
        confidence=safe_confidence,
        iterations=safe_iterations,
        rng=rng,
        include_wer=False,
        include_entity=False,
    )
    _add_result_metrics(
        metrics,
        prefix="speech_proxy",
        source="Speech proxy suite",
        rows=speech_rows,
        confidence=safe_confidence,
        iterations=safe_iterations,
        rng=rng,
        include_wer=True,
        include_entity=True,
    )
    _add_result_metrics(
        metrics,
        prefix="real_audio",
        source="Real audio suite",
        rows=audio_rows,
        confidence=safe_confidence,
        iterations=safe_iterations,
        rng=rng,
        include_wer=True,
        include_entity=True,
        include_deepgram=True,
    )
    metrics.append(
        _proportion_metric(
            metric_id="real_audio.semantic_transcript_pass_rate",
            label="Real audio semantic transcript pass rate",
            source="Real audio suite",
            values=_numeric_values(rows=audio_rows, selector=lambda row: (row.get("semanticTranscript") or {}).get("passed")),
            confidence=safe_confidence,
            paper_use="Intent/slot-preserving transcript success after canonicalization",
        )
    )
    metrics.append(
        _bootstrap_metric(
            metric_id="real_audio.semantic_score_mean",
            label="Real audio mean semantic transcript score",
            source="Real audio suite",
            values=_numeric_values(audio_rows, lambda row: (row.get("semanticTranscript") or {}).get("score")),
            statistic_fn=_mean,
            statistic_name="mean",
            unit="ratio",
            confidence=safe_confidence,
            iterations=safe_iterations,
            rng=rng,
            paper_use="Average deterministic intent/slot/canonical transcript preservation score",
            direction="higher_is_better",
        )
    )
    _add_result_metrics(
        metrics,
        prefix="accepted_audio",
        source="Accepted real-audio set",
        rows=accepted_audio_rows,
        confidence=safe_confidence,
        iterations=safe_iterations,
        rng=rng,
        include_wer=True,
        include_entity=True,
        include_deepgram=True,
    )
    metrics.append(
        _proportion_metric(
            metric_id="combined.pass_rate",
            label="Combined suite pass rate",
            source="Benchmark + speech + real audio",
            values=_numeric_values(combined_rows, lambda row: row.get("passed")),
            confidence=safe_confidence,
            paper_use="Overall task success across latest saved evaluation suites",
        )
    )
    metrics.append(
        _bootstrap_metric(
            metric_id="combined.cost_per_1k_turns",
            label="Combined cost per 1k turns",
            source="Benchmark + speech + real audio",
            values=_numeric_values(combined_rows, lambda row: (_cost_value(row, "vapiStackCost") or 0.0) * 1000.0 if _cost_value(row, "vapiStackCost") is not None else None),
            statistic_fn=_mean,
            statistic_name="mean",
            unit="usd_per_1k_turns",
            confidence=safe_confidence,
            iterations=safe_iterations,
            rng=rng,
            paper_use="Primary composed-stack cost claim with uncertainty bounds",
            direction="lower_is_better",
        )
    )
    metrics.append(
        _bootstrap_metric(
            metric_id="combined.savings_vs_openai_realtime",
            label="Savings vs GPT Realtime",
            source="Benchmark + speech + real audio",
            values=_cost_savings_values(combined_rows, "openaiRealtimeCost"),
            statistic_fn=_mean,
            statistic_name="mean",
            unit="rate",
            confidence=safe_confidence,
            iterations=safe_iterations,
            rng=rng,
            paper_use="Cost delta claim against GPT Realtime baseline",
            direction="higher_is_better",
        )
    )
    metrics.append(
        _bootstrap_metric(
            metric_id="combined.savings_vs_gemini_live",
            label="Savings vs Gemini Live",
            source="Benchmark + speech + real audio",
            values=_cost_savings_values(combined_rows, "geminiLiveCost"),
            statistic_fn=_mean,
            statistic_name="mean",
            unit="rate",
            confidence=safe_confidence,
            iterations=safe_iterations,
            rng=rng,
            paper_use="Cost delta claim against Gemini Live baseline",
            direction="higher_is_better",
        )
    )
    metrics.append(
        _proportion_metric(
            metric_id="audio_robustness.regression_rate",
            label="Audio robustness regression rate",
            source="Audio robustness analyzer",
            values=_numeric_values(robustness_rows, lambda row: 1.0 if row.get("verdict") == "regression" else 0.0),
            confidence=safe_confidence,
            paper_use="Regression rate over paired original-vs-augmented recordings",
            direction="lower_is_better",
        )
    )
    for field, label, unit, direction in [
        ("deltaWer", "Audio robustness delta WER", "delta_wer", "lower_is_better"),
        ("deltaEntityRecall", "Audio robustness delta entity recall", "rate", "higher_is_better"),
        ("deltaTranscriptionLatencyMs", "Audio robustness delta Deepgram latency", "ms", "lower_is_better"),
    ]:
        metrics.append(
            _bootstrap_metric(
                metric_id=f"audio_robustness.{field}",
                label=label,
                source="Audio robustness analyzer",
                values=_numeric_values(robustness_rows, lambda row, field=field: row.get(field)),
                statistic_fn=_mean,
                statistic_name="mean",
                unit=unit,
                confidence=safe_confidence,
                iterations=safe_iterations,
                rng=rng,
                paper_use="Paired acoustic stress delta over generated variants",
                direction=direction,
            )
        )

    run_ids = {
        "benchmark": benchmark.get("runId"),
        "speech": speech.get("runId"),
        "audio": audio.get("runId"),
        "acceptedAudio": audio_accepted.get("runId"),
        "audioRobustness": robustness.get("runId"),
    }
    summary = _summary(metrics, run_ids, safe_confidence, safe_iterations, seed)
    payload = {
        "found": True,
        "runId": datetime.now(timezone.utc).strftime("stats-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suite": "paper_statistics_confidence_pack",
        "summary": summary,
        "metrics": metrics,
        "interpretation": _interpretation(summary),
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "csv": str(LATEST_CSV_PATH.relative_to(ROOT_DIR)),
        },
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with LATEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _write_csv(metrics)
    return payload
