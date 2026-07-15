from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio_accepted import build_audio_accepted_set, load_latest_audio_accepted_set
from .audio_error_analysis import build_audio_error_analysis, load_latest_audio_error_analysis
from .audio_eval import analyze_audio_robustness, build_audio_dataset_manifest, load_latest_audio_eval, load_latest_audio_manifest, load_latest_audio_robustness, run_real_audio_suite
from .audio_quality import load_latest_audio_quality, run_audio_quality_gate
from .benchmark_suite import load_latest_benchmark, run_benchmark_suite
from .case_factory import generate_case_factory, load_latest_case_factory
from .claim_readiness import generate_claim_readiness_pack, load_latest_claim_readiness
from .draft_validation import load_latest_draft_validation, run_draft_validation
from .experiment_planner import generate_experiment_plan, load_latest_experiment_plan
from .speech_eval import load_latest_speech_eval, run_speech_robustness_suite
from .statistics_pack import generate_statistics_pack, load_latest_statistics_pack
from .suite_promotion import load_latest_suite_promotion, run_suite_promotion


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "paper"
LATEST_JSON_PATH = ARTIFACT_DIR / "latest_report.json"
LATEST_MD_PATH = ARTIFACT_DIR / "latest_report.md"
CORE_CSV_PATH = ARTIFACT_DIR / "core_metrics.csv"
COST_CSV_PATH = ARTIFACT_DIR / "cost_comparison.csv"


def load_latest_paper_report() -> dict[str, Any]:
    if not LATEST_JSON_PATH.is_file():
        return {"found": False, "message": "No paper results pack saved yet."}
    with LATEST_JSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["found"] = True
    return payload


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("summary")
    return value if isinstance(value, dict) else {}


def _rate(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return None


def _pct(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value * 100:.{digits}f}%"


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _money(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"${value:.{digits}f}"


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


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


def _case_costs(results: list[dict[str, Any]]) -> dict[str, Any]:
    ours: list[float] = []
    openai: list[float] = []
    gemini: list[float] = []
    voice_latencies: list[float] = []
    for result in results:
        cost = result.get("cost") if isinstance(result.get("cost"), dict) else {}
        if isinstance(cost.get("vapiStackCost"), (int, float)):
            ours.append(float(cost["vapiStackCost"]))
        if isinstance(cost.get("openaiRealtimeCost"), (int, float)):
            openai.append(float(cost["openaiRealtimeCost"]))
        if isinstance(cost.get("geminiLiveCost"), (int, float)):
            gemini.append(float(cost["geminiLiveCost"]))
        if isinstance(result.get("estimatedVoiceLatencyMs"), (int, float)):
            voice_latencies.append(float(result["estimatedVoiceLatencyMs"]))
    return {
        "count": len(results),
        "avgVapiStack": _mean(ours) or 0.0,
        "per1000VapiStack": round((_mean(ours) or 0.0) * 1000, 4),
        "per1000OpenAIRealtime": round((_mean(openai) or 0.0) * 1000, 4),
        "per1000GeminiLive": round((_mean(gemini) or 0.0) * 1000, 4),
        "voiceP95Ms": _percentile(voice_latencies, 0.95),
    }


def _combined_cost(*payloads: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for payload in payloads:
        results.extend(
            item
            for item in payload.get("results") or []
            if isinstance(item, dict) and not item.get("skipped")
        )
    return _case_costs(results)


def _safe_savings(ours: float | None, baseline: float | None) -> float | None:
    if not ours or not baseline or baseline <= 0:
        return None
    return round((baseline - ours) / baseline, 4)


def _latency_score(voice_p95_ms: float | None, target_ms: int = 2500) -> float:
    if not voice_p95_ms or voice_p95_ms <= 0:
        return 0.0
    return round(min(1.0, target_ms / voice_p95_ms), 4)


def _cost_score(ours: float | None, openai: float | None, gemini: float | None) -> float:
    baselines = [value for value in [openai, gemini] if isinstance(value, (int, float)) and value > 0]
    if not ours or not baselines:
        return 0.0
    cheapest = min(baselines)
    return round(1.0 if ours <= cheapest else cheapest / ours, 4)


def _readiness_score(benchmark_summary: dict[str, Any], speech_summary: dict[str, Any], combined_cost: dict[str, Any]) -> dict[str, Any]:
    task_pass = float(benchmark_summary.get("passRate") or 0.0)
    speech_pass = float(speech_summary.get("passRate") or 0.0)
    entity_recall = float(speech_summary.get("avgEntityRecall") or 0.0)
    latency = _latency_score(combined_cost.get("voiceP95Ms"))
    cost = _cost_score(
        combined_cost.get("per1000VapiStack"),
        combined_cost.get("per1000OpenAIRealtime"),
        combined_cost.get("per1000GeminiLive"),
    )
    weighted = (
        task_pass * 0.30
        + speech_pass * 0.25
        + entity_recall * 0.15
        + latency * 0.15
        + cost * 0.15
    )
    return {
        "score": round(weighted * 100, 1),
        "scale": "0-100",
        "weights": {
            "taskPassRate": 0.30,
            "speechPassRate": 0.25,
            "entityRecall": 0.15,
            "latencyP95Target2500Ms": 0.15,
            "costVsRealtimeBaselines": 0.15,
        },
        "components": {
            "taskPassRate": round(task_pass, 4),
            "speechPassRate": round(speech_pass, 4),
            "entityRecall": round(entity_recall, 4),
            "latencyScore": latency,
            "costScore": cost,
        },
    }


def _core_metrics(benchmark_summary: dict[str, Any], speech_summary: dict[str, Any], combined: dict[str, Any], readiness: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "metric": "Task success",
            "value": benchmark_summary.get("passRate"),
            "display": _pct(benchmark_summary.get("passRate")),
            "source": "paper-grade suite",
            "paperUse": "Primary task completion metric",
        },
        {
            "metric": "Speech robustness success",
            "value": speech_summary.get("passRate"),
            "display": _pct(speech_summary.get("passRate")),
            "source": "speech robustness suite",
            "paperUse": "End-to-end success after ASR perturbation",
        },
        {
            "metric": "Average WER",
            "value": speech_summary.get("avgWer"),
            "display": _num(speech_summary.get("avgWer"), 4),
            "source": "speech robustness suite",
            "paperUse": "ASR transcript quality",
        },
        {
            "metric": "Entity recall",
            "value": speech_summary.get("avgEntityRecall"),
            "display": _pct(speech_summary.get("avgEntityRecall")),
            "source": "speech robustness suite",
            "paperUse": "Product/policy slot preservation under ASR noise",
        },
        {
            "metric": "Voice latency p95",
            "value": combined.get("voiceP95Ms"),
            "display": f"{_num(combined.get('voiceP95Ms'), 1)} ms",
            "source": "combined live suites",
            "paperUse": "Interactive voice latency proxy",
        },
        {
            "metric": "Cost per 1k turns",
            "value": combined.get("per1000VapiStack"),
            "display": _money(combined.get("per1000VapiStack"), 2),
            "source": "cost ledger",
            "paperUse": "Economic efficiency metric",
        },
        {
            "metric": "Readiness score",
            "value": readiness.get("score"),
            "display": f"{_num(readiness.get('score'), 1)} / 100",
            "source": "weighted report score",
            "paperUse": "Demo-readiness summary, not a universal benchmark",
        },
    ]


def _cost_rows(combined: dict[str, Any]) -> list[dict[str, Any]]:
    ours = combined.get("per1000VapiStack")
    openai = combined.get("per1000OpenAIRealtime")
    gemini = combined.get("per1000GeminiLive")
    return [
        {
            "architecture": "AislePilot composed stack",
            "mode": "Vapi + Deepgram + GPT-4o-mini + ElevenLabs",
            "per1000Turns": ours,
            "display": _money(ours, 2),
            "savingsVsComposed": 0.0,
        },
        {
            "architecture": "GPT Realtime baseline",
            "mode": "published pricing estimate",
            "per1000Turns": openai,
            "display": _money(openai, 2),
            "savingsVsComposed": _safe_savings(ours, openai),
        },
        {
            "architecture": "Gemini Live Native Audio baseline",
            "mode": "published pricing estimate",
            "per1000Turns": gemini,
            "display": _money(gemini, 2),
            "savingsVsComposed": _safe_savings(ours, gemini),
        },
    ]


def _group_rows(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    groups = summary.get(key)
    if not isinstance(groups, dict):
        return []
    rows: list[dict[str, Any]] = []
    for name, metrics in groups.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "name": name,
                "total": metrics.get("total"),
                "passed": metrics.get("passed"),
                "passRate": metrics.get("passRate"),
                "avgWer": metrics.get("avgWer"),
                "avgEntityRecall": metrics.get("avgEntityRecall"),
            }
        )
    return rows


def _failure_rows(*payloads: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        suite = payload.get("suite") or "suite"
        for result in payload.get("results") or []:
            if not isinstance(result, dict) or result.get("passed"):
                continue
            rows.append(
                {
                    "suite": suite,
                    "id": result.get("id"),
                    "group": result.get("group") or result.get("route") or result.get("type"),
                    "failures": " | ".join(result.get("failures") or []),
                }
            )
    return rows


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    rendered = [[cell(item) for item in row] for row in rows]
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rendered]
    return "\n".join([header, separator, *body])


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    core = report["tables"]["coreMetrics"]
    costs = report["tables"]["costComparison"]
    benchmark_groups = report["tables"]["benchmarkByGroup"]
    speech_conditions = report["tables"]["speechByCondition"]
    audio_prompts = report["tables"].get("audioPromptCoverage", [])
    audio_conditions = report["tables"].get("audioConditionCoverage", [])
    audio_quality_rows = report["tables"].get("audioQualityRows", [])
    audio_accepted_rows = report["tables"].get("audioAcceptedRows", [])
    audio_error_actions = report["tables"].get("audioErrorActionPlan", [])
    audio_error_conditions = report["tables"].get("audioErrorConditionRisks", [])
    audio_robustness = report["tables"].get("audioRobustnessByAugmentation", [])
    statistics_intervals = report["tables"].get("statisticsIntervals", [])
    claim_readiness = report["tables"].get("claimReadiness", [])
    experiment_plan = report["tables"].get("experimentPlan", [])
    case_factory = report["tables"].get("caseFactory", [])
    draft_validation = report["tables"].get("draftValidation", [])
    suite_promotion = report["tables"].get("suitePromotion", [])
    failures = report["tables"]["failureAnalysis"]
    sections = [
        "# AislePilot Paper Results Pack",
        f"Generated: `{report['createdAt']}`",
        "## Executive Metrics",
        _markdown_table(
            ["Metric", "Value", "Source", "Paper use"],
            [[row["metric"], row["display"], row["source"], row["paperUse"]] for row in core],
        ),
        "## Cost Comparison",
        _markdown_table(
            ["Architecture", "Mode", "Cost / 1k turns", "Savings vs baseline"],
            [
                [
                    row["architecture"],
                    row["mode"],
                    row["display"],
                    "-" if row["savingsVsComposed"] in (None, 0.0) else _pct(row["savingsVsComposed"]),
                ]
                for row in costs
            ],
        ),
        "## Statistical Confidence Intervals",
        _markdown_table(
            ["Metric", "Observed", "95% CI", "N", "Method"],
            [
                [
                    row.get("label"),
                    row.get("displayObserved"),
                    row.get("displayCi"),
                    row.get("n"),
                    row.get("method"),
                ]
                for row in statistics_intervals
                if row.get("status") == "ok"
            ][:12] or [["-", "-", "-", "0", "Run the statistics pack after benchmark artifacts exist."]],
        ),
        "## Claim Readiness Gate",
        _markdown_table(
            ["Claim", "Status", "Observed", "CI", "N", "Next action"],
            [
                [
                    row.get("claim"),
                    row.get("status"),
                    row.get("displayObserved"),
                    row.get("displayCi"),
                    row.get("n"),
                    row.get("nextAction"),
                ]
                for row in claim_readiness
            ][:12] or [["-", "-", "-", "-", "0", "Run the claim readiness gate."]],
        ),
        "## Next Experiment Plan",
        _markdown_table(
            ["Lane", "Action", "Add", "Reason"],
            [
                [
                    row.get("lane"),
                    row.get("action"),
                    row.get("addCount"),
                    row.get("reason"),
                ]
                for row in experiment_plan
            ][:10] or [["-", "Run the experiment planner.", "0", "-"]],
        ),
        "## Draft Case Factory",
        _markdown_table(
            ["Artifact", "Count", "Use"],
            [
                [row.get("artifact"), row.get("count"), row.get("paperUse")]
                for row in case_factory
            ] or [["-", "0", "Run the case factory."]],
        ),
        "## Draft Validation Gate",
        _markdown_table(
            ["Artifact", "Ready", "Blocked", "Promotion target"],
            [
                [
                    row.get("artifact"),
                    row.get("ready"),
                    row.get("blocked"),
                    row.get("promotionTarget"),
                ]
                for row in draft_validation
            ] or [["-", "0", "0", "Run the draft validation gate."]],
        ),
        "## Suite Promotion",
        _markdown_table(
            ["Target", "Candidates", "Added", "Skipped", "Mode", "Path"],
            [
                [
                    row.get("target"),
                    row.get("candidateCount"),
                    row.get("addedCount"),
                    row.get("skippedCount"),
                    "dry run" if row.get("dryRun") else "wrote",
                    row.get("targetPath"),
                ]
                for row in suite_promotion
            ] or [["-", "0", "0", "0", "Run the suite promotion preview.", "-"]],
        ),
        "## Benchmark Groups",
        _markdown_table(
            ["Group", "Passed", "Total", "Pass rate"],
            [[row["name"], row.get("passed"), row.get("total"), _pct(row.get("passRate"))] for row in benchmark_groups],
        ),
        "## Speech Conditions",
        _markdown_table(
            ["Condition", "Passed", "Total", "Pass rate", "WER", "Entity recall"],
            [
                [
                    row["name"],
                    row.get("passed"),
                    row.get("total"),
                    _pct(row.get("passRate")),
                    _num(row.get("avgWer"), 4),
                    _pct(row.get("avgEntityRecall")),
                ]
                for row in speech_conditions
            ],
        ),
        "## Audio Dataset Coverage",
        _markdown_table(
            ["Prompt", "Recordings", "Target", "Missing", "Coverage"],
            [
                [
                    row.get("referenceText"),
                    row.get("recordings"),
                    row.get("target"),
                    row.get("missing"),
                    _pct(row.get("coverageRate")),
                ]
                for row in audio_prompts
            ] or [["-", "0", "-", "-", "No audio prompts loaded"]],
        ),
        "## Audio Conditions",
        _markdown_table(
            ["Condition", "Recordings", "Speakers", "Evaluated", "Pass", "WER"],
            [
                [
                    row.get("condition"),
                    row.get("recordings"),
                    row.get("speakerCount"),
                    row.get("evaluated"),
                    _pct(row.get("passRate")),
                    _num(row.get("avgWer"), 4),
                ]
                for row in audio_conditions
            ] or [["-", "0", "0", "0", "-", "-"]],
        ),
        "## Audio QA Retake Queue",
        _markdown_table(
            ["Priority", "Recording", "Score", "Modes", "Instruction"],
            [
                [
                    row.get("priority"),
                    row.get("id"),
                    row.get("qualityScore"),
                    ", ".join(row.get("failureModes") or []),
                    row.get("retakeInstruction"),
                ]
                for row in audio_quality_rows
            ][:12] or [["-", "-", "-", "-", "Run the audio QA gate."]],
        ),
        "## Accepted Audio Set",
        _markdown_table(
            ["Recording", "Prompt", "Accent", "WER", "Entity", "Supersedes"],
            [
                [
                    row.get("recordingId"),
                    row.get("referenceText"),
                    row.get("accent"),
                    _num(row.get("wer"), 4),
                    _pct(row.get("entityRecall")),
                    len(row.get("supersedes") or []),
                ]
                for row in audio_accepted_rows
            ][:12] or [["-", "-", "-", "-", "-", "Run the accepted-set builder."]],
        ),
        "## Audio Error Taxonomy",
        _markdown_table(
            ["Failure", "Family", "Severity", "Affected", "Action"],
            [
                [
                    row.get("failure"),
                    row.get("family"),
                    row.get("severity"),
                    row.get("affectedRecordings"),
                    row.get("recommendedAction"),
                ]
                for row in audio_error_actions
            ][:12] or [["-", "-", "-", "0", "Run the audio error analysis."]],
        ),
        "## Audio Condition Risk",
        _markdown_table(
            ["Condition", "Failed", "Total", "Failure rate", "Top failure", "WER"],
            [
                [
                    row.get("condition"),
                    row.get("failed"),
                    row.get("total"),
                    _pct(row.get("failureRate")),
                    row.get("topFailure"),
                    _num(row.get("avgWer"), 4),
                ]
                for row in audio_error_conditions
            ][:12] or [["-", "0", "0", "-", "-", "-"]],
        ),
        "## Audio Robustness Deltas",
        _markdown_table(
            ["Augmentation", "Compared", "Regression", "Pass", "Baseline pass", "Delta WER", "Delta entity", "Delta ASR ms"],
            [
                [
                    row.get("augmentationLabel") or row.get("augmentationType"),
                    row.get("comparedCount"),
                    _pct(row.get("regressionRate")),
                    _pct(row.get("passRate")),
                    _pct(row.get("baselinePassRate")),
                    _num(row.get("avgDeltaWer"), 4),
                    _num(row.get("avgDeltaEntityRecall"), 4),
                    _num(row.get("avgDeltaTranscriptionLatencyMs"), 1),
                ]
                for row in audio_robustness
            ] or [["-", "0", "-", "-", "-", "-", "-", "-"]],
        ),
        "## Failures",
        _markdown_table(
            ["Suite", "Case", "Group", "Failure"],
            [[row["suite"], row["id"], row["group"], row["failures"]] for row in failures] or [["-", "-", "-", "No failures in latest saved suites."]],
        ),
        "## Interpretation Notes",
        "- Speech robustness currently uses transcript proxies; recorded audio and provider ASR logs should be added before external publication.",
        "- The readiness score is a project-specific dashboard metric, not a universal benchmark score.",
        "- Cost comparisons use the project cost ledger assumptions and should be rerun when provider pricing changes.",
        "## Reproducibility",
        _markdown_table(["Artifact", "Path"], [[name, path] for name, path in report["artifacts"].items()]),
        "## Snapshot",
        f"- Readiness score: `{summary['combined']['readinessScore']}`",
        f"- Combined cost per 1k turns: `{_money(summary['combined']['costPer1000Turns'], 2)}`",
        f"- Combined p95 voice latency: `{_num(summary['combined']['voiceP95Ms'], 1)} ms`",
    ]
    return "\n\n".join(sections) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_paper_results_pack(*, rerun_suites: bool = False, save: bool = True) -> dict[str, Any]:
    benchmark = run_benchmark_suite(limit=None, include_payloads=False, save=True) if rerun_suites else load_latest_benchmark()
    if benchmark.get("found") is False:
        benchmark = run_benchmark_suite(limit=None, include_payloads=False, save=True)
    speech = run_speech_robustness_suite(limit=None, save=True) if rerun_suites else load_latest_speech_eval()
    if speech.get("found") is False:
        speech = run_speech_robustness_suite(limit=None, save=True)
    audio = run_real_audio_suite(limit=None, save=True) if rerun_suites else load_latest_audio_eval()
    if audio.get("found") is False:
        audio = {
            "runId": None,
            "suite": "voice_retail_real_audio_deepgram_suite",
            "summary": {"total": 0, "evaluated": 0, "skipped": 0, "passed": 0, "passRate": 0.0},
            "results": [],
        }
    audio_manifest = build_audio_dataset_manifest(save=True) if rerun_suites else load_latest_audio_manifest()
    if audio_manifest.get("found") is False:
        audio_manifest = build_audio_dataset_manifest(save=True)
    audio_robustness = analyze_audio_robustness(save=True) if rerun_suites else load_latest_audio_robustness()
    if audio_robustness.get("found") is False:
        audio_robustness = analyze_audio_robustness(save=True)
    audio_quality = run_audio_quality_gate(save=True) if rerun_suites else load_latest_audio_quality()
    if audio_quality.get("found") is False:
        audio_quality = run_audio_quality_gate(save=True)
    audio_accepted = build_audio_accepted_set(save=True) if rerun_suites else load_latest_audio_accepted_set()
    if audio_accepted.get("found") is False:
        audio_accepted = build_audio_accepted_set(save=True)
    audio_error = build_audio_error_analysis(save=True) if rerun_suites else load_latest_audio_error_analysis()
    if audio_error.get("found") is False:
        audio_error = build_audio_error_analysis(save=True)
    statistics_pack = generate_statistics_pack(save=True) if rerun_suites else load_latest_statistics_pack()
    if statistics_pack.get("found") is False:
        statistics_pack = generate_statistics_pack(save=True)
    claim_readiness = generate_claim_readiness_pack(regenerate_statistics=rerun_suites, save=True) if rerun_suites else load_latest_claim_readiness()
    if claim_readiness.get("found") is False:
        claim_readiness = generate_claim_readiness_pack(save=True)
    experiment_plan = generate_experiment_plan(refresh_claims=rerun_suites, save=True) if rerun_suites else load_latest_experiment_plan()
    if experiment_plan.get("found") is False:
        experiment_plan = generate_experiment_plan(save=True)
    case_factory = generate_case_factory(refresh_plan=rerun_suites, save=True) if rerun_suites else load_latest_case_factory()
    if case_factory.get("found") is False:
        case_factory = generate_case_factory(save=True)
    draft_validation = run_draft_validation(refresh_factory=rerun_suites, save=True) if rerun_suites else load_latest_draft_validation()
    if draft_validation.get("found") is False:
        draft_validation = run_draft_validation(save=True)
    suite_promotion = run_suite_promotion(dry_run=True, refresh_validation=rerun_suites, save=True) if rerun_suites else load_latest_suite_promotion()
    if suite_promotion.get("found") is False:
        suite_promotion = run_suite_promotion(dry_run=True, save=True)

    benchmark_summary = _summary(benchmark)
    speech_summary = _summary(speech)
    audio_summary = _summary(audio)
    audio_accepted_summary = _summary(audio_accepted)
    audio_accepted_result_summary = audio_accepted.get("acceptedSummary") if isinstance(audio_accepted.get("acceptedSummary"), dict) else {}
    audio_manifest_summary = _summary(audio_manifest)
    audio_robustness_summary = _summary(audio_robustness)
    audio_for_claims = {
        "results": audio_accepted.get("acceptedResults") if isinstance(audio_accepted.get("acceptedResults"), list) else [],
        "summary": audio_accepted_result_summary,
    }
    combined = _combined_cost(benchmark, speech, audio_for_claims)
    readiness = _readiness_score(benchmark_summary, speech_summary, combined)
    core_metrics = _core_metrics(benchmark_summary, speech_summary, combined, readiness)
    audio_total = int(audio_summary.get("total") or 0)
    audio_evaluated = int(audio_summary.get("evaluated") or 0)
    core_metrics.extend(
        [
            {
                "metric": "Real audio evaluated",
                "value": audio_evaluated,
                "display": f"{audio_evaluated} / {audio_total}",
                "source": "real audio eval",
                "paperUse": "Recorded browser audio coverage",
            },
            {
                "metric": "Real audio pass",
                "value": audio_summary.get("passRate") if audio_evaluated else None,
                "display": _pct(audio_summary.get("passRate")) if audio_evaluated else "recordings needed",
                "source": "real audio eval",
                "paperUse": "Raw archive provider-ASR task success",
            },
            {
                "metric": "Accepted audio pass",
                "value": audio_accepted_summary.get("acceptedPassRate"),
                "display": _pct(audio_accepted_summary.get("acceptedPassRate")) if audio_accepted_summary.get("acceptedRecordings") else "accepted fixtures needed",
                "source": "accepted audio set",
                "paperUse": "Publishable provider-ASR task success",
            },
            {
                "metric": "Accepted audio WER",
                "value": audio_accepted_summary.get("acceptedAvgWer"),
                "display": _num(audio_accepted_summary.get("acceptedAvgWer"), 4) if audio_accepted_summary.get("acceptedRecordings") else "accepted fixtures needed",
                "source": "accepted audio set",
                "paperUse": "Publishable provider-ASR quality over curated fixtures",
            },
            {
                "metric": "Accepted-set lift",
                "value": audio_accepted_summary.get("passRateLiftVsRaw"),
                "display": _pct(audio_accepted_summary.get("passRateLiftVsRaw")),
                "source": "accepted audio set",
                "paperUse": "Retake/supersession improvement over raw archive",
            },
            {
                "metric": "Superseded recordings",
                "value": audio_accepted_summary.get("supersededRecordings"),
                "display": str(audio_accepted_summary.get("supersededRecordings", 0)),
                "source": "accepted audio set",
                "paperUse": "Audit trail for old failed or replaced audio takes",
            },
            {
                "metric": "Audio failure buckets",
                "value": (audio_error.get("summary") or {}).get("failureBucketCount"),
                "display": str((audio_error.get("summary") or {}).get("failureBucketCount", 0)),
                "source": "audio error taxonomy",
                "paperUse": "Error-analysis taxonomy breadth for recorded audio",
            },
            {
                "metric": "Audio language mismatch",
                "value": (audio_error.get("summary") or {}).get("languageMismatches"),
                "display": str((audio_error.get("summary") or {}).get("languageMismatches", 0)),
                "source": "audio error taxonomy",
                "paperUse": "Separates accent robustness from multilingual-language mismatch",
            },
            {
                "metric": "Real audio WER",
                "value": audio_summary.get("avgWer") if audio_evaluated else None,
                "display": _num(audio_summary.get("avgWer"), 4) if audio_evaluated else "recordings needed",
                "source": "real audio eval",
                "paperUse": "Provider ASR quality over recorded fixtures",
            },
            {
                "metric": "Multilingual audio turns",
                "value": audio_summary.get("multilingualScored"),
                "display": str(audio_summary.get("multilingualScored", 0)),
                "source": "real audio eval",
                "paperUse": "Spanish-to-canonical-English grounding coverage",
            },
            {
                "metric": "Canonical WER",
                "value": audio_summary.get("avgCanonicalWer") if audio_evaluated else None,
                "display": _num(audio_summary.get("avgCanonicalWer"), 4) if audio_evaluated else "recordings needed",
                "source": "multilingual canonicalizer",
                "paperUse": "ASR quality after language-aware query canonicalization",
            },
            {
                "metric": "Semantic transcript pass",
                "value": audio_summary.get("semanticTranscriptPassRate") if audio_evaluated else None,
                "display": _pct(audio_summary.get("semanticTranscriptPassRate")) if audio_evaluated else "recordings needed",
                "source": "semantic transcript scorer",
                "paperUse": "Intent/slot preservation despite literal WER or paraphrase differences",
            },
            {
                "metric": "Recovered ASR misses",
                "value": audio_summary.get("semanticRecoveredAsrMisses"),
                "display": str(audio_summary.get("semanticRecoveredAsrMisses", 0)),
                "source": "semantic transcript scorer",
                "paperUse": "Cases where downstream task and semantic preservation pass despite strict ASR failure",
            },
            {
                "metric": "Audio dataset coverage",
                "value": audio_manifest_summary.get("coverageRate"),
                "display": _pct(audio_manifest_summary.get("coverageRate")),
                "source": "audio dataset manifest",
                "paperUse": "Recording coverage against target samples per prompt",
            },
            {
                "metric": "Audio usable fixtures",
                "value": (audio_quality.get("summary") or {}).get("usableRate"),
                "display": f"{(audio_quality.get('summary') or {}).get('usableForPaper', 0)} / {(audio_quality.get('summary') or {}).get('totalRecordings', 0)}",
                "source": "audio QA retake gate",
                "paperUse": "Real-audio fixture quality before publication claims",
            },
            {
                "metric": "Complete audio prompts",
                "value": audio_manifest_summary.get("completePrompts"),
                "display": f"{audio_manifest_summary.get('completePrompts', 0)} / {audio_manifest_summary.get('templateCount', 0)}",
                "source": "audio dataset manifest",
                "paperUse": "Prompt-level coverage for real audio experiments",
            },
            {
                "metric": "Audio robustness compared",
                "value": audio_robustness_summary.get("comparedCount"),
                "display": f"{audio_robustness_summary.get('comparedCount', 0)} / {audio_robustness_summary.get('variantCount', 0)}",
                "source": "audio robustness analyzer",
                "paperUse": "Parent-vs-augmentation comparison coverage",
            },
            {
                "metric": "Audio robustness regression",
                "value": audio_robustness_summary.get("regressionRate"),
                "display": _pct(audio_robustness_summary.get("regressionRate")) if audio_robustness_summary.get("comparedCount") else "variants needed",
                "source": "audio robustness analyzer",
                "paperUse": "Acoustic degradation rate over generated stress variants",
            },
            {
                "metric": "Audio robustness WER delta",
                "value": audio_robustness_summary.get("avgDeltaWer"),
                "display": _num(audio_robustness_summary.get("avgDeltaWer"), 4) if audio_robustness_summary.get("comparedCount") else "variants needed",
                "source": "audio robustness analyzer",
                "paperUse": "Mean WER change versus parent recordings",
            },
            {
                "metric": "Statistics intervals",
                "value": (statistics_pack.get("summary") or {}).get("populatedMetricCount"),
                "display": f"{(statistics_pack.get('summary') or {}).get('populatedMetricCount', 0)} / {(statistics_pack.get('summary') or {}).get('metricCount', 0)}",
                "source": "statistics pack",
                "paperUse": "Confidence intervals for publishable benchmark claims",
            },
            {
                "metric": "Publishable claims",
                "value": (claim_readiness.get("summary") or {}).get("publishable"),
                "display": f"{(claim_readiness.get('summary') or {}).get('publishable', 0)} / {(claim_readiness.get('summary') or {}).get('totalClaims', 0)}",
                "source": "claim readiness gate",
                "paperUse": "Claim-level evidence status for paper writing",
            },
            {
                "metric": "Planned next samples",
                "value": (experiment_plan.get("summary") or {}).get("plannedSamples"),
                "display": str((experiment_plan.get("summary") or {}).get("plannedSamples", 0)),
                "source": "experiment planner",
                "paperUse": "Deduplicated sample budget for next experiment run",
            },
            {
                "metric": "Draft generated cases",
                "value": (case_factory.get("summary") or {}).get("totalDraftArtifacts"),
                "display": str((case_factory.get("summary") or {}).get("totalDraftArtifacts", 0)),
                "source": "case factory",
                "paperUse": "Generated draft benchmark, speech, and audio-prompt artifacts",
            },
            {
                "metric": "Promotion-ready drafts",
                "value": (draft_validation.get("summary") or {}).get("promotionReady"),
                "display": f"{(draft_validation.get('summary') or {}).get('promotionReady', 0)} / {(draft_validation.get('summary') or {}).get('totalDraftArtifacts', 0)}",
                "source": "draft validation gate",
                "paperUse": "Quality gate before promoting generated cases into official suites",
            },
            {
                "metric": "Suite promotion preview",
                "value": (suite_promotion.get("summary") or {}).get("totalAdded"),
                "display": f"{(suite_promotion.get('summary') or {}).get('totalAdded', 0)} ready / {(suite_promotion.get('summary') or {}).get('totalSkipped', 0)} skipped",
                "source": "suite promotion gate",
                "paperUse": "Audited bridge from validated drafts to official benchmark suites",
            },
        ]
    )
    cost_comparison = _cost_rows(combined)

    payload = {
        "runId": datetime.now(timezone.utc).strftime("paper-%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suite": "aislepilot_live_paper_results_pack",
        "inputs": {
            "benchmarkRunId": benchmark.get("runId"),
            "speechRunId": speech.get("runId"),
            "audioRunId": audio.get("runId"),
            "audioAcceptedRunId": audio_accepted.get("runId"),
            "benchmarkSuite": benchmark.get("suite"),
            "speechSuite": speech.get("suite"),
        },
        "summary": {
            "benchmark": {
                "total": benchmark_summary.get("total"),
                "passed": benchmark_summary.get("passed"),
                "passRate": _rate(benchmark_summary.get("passRate")),
                "voiceP95Ms": (benchmark_summary.get("latency") or {}).get("voiceP95Ms"),
                "costPer1000Turns": (benchmark_summary.get("cost") or {}).get("per1000VapiStack"),
            },
            "speech": {
                "total": speech_summary.get("total"),
                "passed": speech_summary.get("passed"),
                "passRate": _rate(speech_summary.get("passRate")),
                "asrPassRate": _rate(speech_summary.get("asrPassRate")),
                "downstreamTaskSuccess": _rate(speech_summary.get("downstreamTaskSuccess")),
                "avgWer": speech_summary.get("avgWer"),
                "avgEntityRecall": speech_summary.get("avgEntityRecall"),
                "voiceP95Ms": (speech_summary.get("latency") or {}).get("voiceP95Ms"),
                "costPer1000Turns": (speech_summary.get("cost") or {}).get("per1000VapiStack"),
            },
            "audioDataset": {
                "templateCount": audio_manifest_summary.get("templateCount"),
                "recordingCount": audio_manifest_summary.get("recordingCount"),
                "requiredRecordings": audio_manifest_summary.get("requiredRecordings"),
                "completePrompts": audio_manifest_summary.get("completePrompts"),
                "coverageRate": audio_manifest_summary.get("coverageRate"),
                "promptCoverageRate": audio_manifest_summary.get("promptCoverageRate"),
                "speakerCount": audio_manifest_summary.get("speakerCount"),
                "conditionCount": audio_manifest_summary.get("conditionCount"),
                "evaluatedRecordings": audio_manifest_summary.get("evaluatedRecordings"),
            },
            "audioQuality": {
                "totalRecordings": (audio_quality.get("summary") or {}).get("totalRecordings"),
                "usableForPaper": (audio_quality.get("summary") or {}).get("usableForPaper"),
                "usableRate": (audio_quality.get("summary") or {}).get("usableRate"),
                "retakeNeeded": (audio_quality.get("summary") or {}).get("retakeNeeded"),
                "urgentRetakes": (audio_quality.get("summary") or {}).get("urgentRetakes"),
                "highRetakes": (audio_quality.get("summary") or {}).get("highRetakes"),
                "emptyTranscripts": (audio_quality.get("summary") or {}).get("emptyTranscripts"),
                "wrongPromptOrUnintelligible": (audio_quality.get("summary") or {}).get("wrongPromptOrUnintelligible"),
                "avgQualityScore": (audio_quality.get("summary") or {}).get("avgQualityScore"),
            },
            "audioAcceptedSet": {
                "rawRecordings": audio_accepted_summary.get("rawRecordings"),
                "rawEvaluated": audio_accepted_summary.get("rawEvaluated"),
                "rawPassed": audio_accepted_summary.get("rawPassed"),
                "rawPassRate": _rate(audio_accepted_summary.get("rawPassRate")),
                "groupCount": audio_accepted_summary.get("groupCount"),
                "acceptedRecordings": audio_accepted_summary.get("acceptedRecordings"),
                "acceptedCoverageRate": _rate(audio_accepted_summary.get("acceptedCoverageRate")),
                "acceptedPassRate": _rate(audio_accepted_summary.get("acceptedPassRate")),
                "acceptedAvgWer": audio_accepted_summary.get("acceptedAvgWer"),
                "acceptedAvgRawWer": audio_accepted_summary.get("acceptedAvgRawWer"),
                "acceptedAvgEntityRecall": audio_accepted_summary.get("acceptedAvgEntityRecall"),
                "acceptedAvgRawEntityRecall": audio_accepted_summary.get("acceptedAvgRawEntityRecall"),
                "acceptedDeepgramP95Ms": audio_accepted_summary.get("acceptedDeepgramP95Ms"),
                "acceptedCostPer1000Turns": audio_accepted_summary.get("acceptedCostPer1000Turns"),
                "supersededRecordings": audio_accepted_summary.get("supersededRecordings"),
                "rejectedRetakes": audio_accepted_summary.get("rejectedRetakes"),
                "groupsNeedingRetake": audio_accepted_summary.get("groupsNeedingRetake"),
                "passRateLiftVsRaw": audio_accepted_summary.get("passRateLiftVsRaw"),
            },
            "audioErrorAnalysis": {
                "totalRecordings": (audio_error.get("summary") or {}).get("totalRecordings"),
                "failed": (audio_error.get("summary") or {}).get("failed"),
                "failureBucketCount": (audio_error.get("summary") or {}).get("failureBucketCount"),
                "asrOnlyFailures": (audio_error.get("summary") or {}).get("asrOnlyFailures"),
                "downstreamFailures": (audio_error.get("summary") or {}).get("downstreamFailures"),
                "languageMismatches": (audio_error.get("summary") or {}).get("languageMismatches"),
                "promptDriftOrWrongClip": (audio_error.get("summary") or {}).get("promptDriftOrWrongClip"),
                "coverageGaps": (audio_error.get("summary") or {}).get("coverageGaps"),
                "topFailure": (audio_error.get("summary") or {}).get("topFailure"),
                "topRecommendedAction": (audio_error.get("summary") or {}).get("topRecommendedAction"),
                "highestPriorityFailure": (audio_error.get("summary") or {}).get("highestPriorityFailure"),
                "highestPriorityAction": (audio_error.get("summary") or {}).get("highestPriorityAction"),
            },
            "audioRobustness": {
                "baselineCount": audio_robustness_summary.get("baselineCount"),
                "variantCount": audio_robustness_summary.get("variantCount"),
                "comparedCount": audio_robustness_summary.get("comparedCount"),
                "stableCount": audio_robustness_summary.get("stableCount"),
                "regressionCount": audio_robustness_summary.get("regressionCount"),
                "regressionRate": _rate(audio_robustness_summary.get("regressionRate")),
                "avgDeltaWer": audio_robustness_summary.get("avgDeltaWer"),
                "avgDeltaEntityRecall": audio_robustness_summary.get("avgDeltaEntityRecall"),
                "avgDeltaTranscriptionLatencyMs": audio_robustness_summary.get("avgDeltaTranscriptionLatencyMs"),
            },
            "statistics": {
                "metricCount": (statistics_pack.get("summary") or {}).get("metricCount"),
                "populatedMetricCount": (statistics_pack.get("summary") or {}).get("populatedMetricCount"),
                "missingMetricCount": (statistics_pack.get("summary") or {}).get("missingMetricCount"),
                "coverageRate": (statistics_pack.get("summary") or {}).get("coverageRate"),
                "widestCiMetric": (statistics_pack.get("summary") or {}).get("widestCiMetric"),
            },
            "claims": {
                "totalClaims": (claim_readiness.get("summary") or {}).get("totalClaims"),
                "publishable": (claim_readiness.get("summary") or {}).get("publishable"),
                "needsMoreData": (claim_readiness.get("summary") or {}).get("needsMoreData"),
                "needsSystemWork": (claim_readiness.get("summary") or {}).get("needsSystemWork"),
                "missingEvidence": (claim_readiness.get("summary") or {}).get("missingEvidence"),
                "paperReady": (claim_readiness.get("summary") or {}).get("paperReady"),
                "claimReadinessScore": (claim_readiness.get("summary") or {}).get("claimReadinessScore"),
                "additionalSamplesRecommended": (claim_readiness.get("summary") or {}).get("additionalSamplesRecommended"),
                "topAction": (claim_readiness.get("summary") or {}).get("topAction"),
            },
            "experimentPlan": {
                "plannedSamples": (experiment_plan.get("summary") or {}).get("plannedSamples"),
                "rawClaimRecommendedSamples": (experiment_plan.get("summary") or {}).get("rawClaimRecommendedSamples"),
                "deduplicationSavings": (experiment_plan.get("summary") or {}).get("deduplicationSavings"),
                "benchmarkCasesToAdd": (experiment_plan.get("summary") or {}).get("benchmarkCasesToAdd"),
                "speechProxyCasesToAdd": (experiment_plan.get("summary") or {}).get("speechProxyCasesToAdd"),
                "realAudioRecordingsToAdd": (experiment_plan.get("summary") or {}).get("realAudioRecordingsToAdd"),
                "stressPairsToEvaluate": (experiment_plan.get("summary") or {}).get("stressPairsToEvaluate"),
                "providerEvalCallsNeeded": (experiment_plan.get("summary") or {}).get("providerEvalCallsNeeded"),
            },
            "caseFactory": {
                "benchmarkDraftCases": (case_factory.get("summary") or {}).get("benchmarkDraftCases"),
                "speechDraftCases": (case_factory.get("summary") or {}).get("speechDraftCases"),
                "audioRecordingPrompts": (case_factory.get("summary") or {}).get("audioRecordingPrompts"),
                "totalDraftArtifacts": (case_factory.get("summary") or {}).get("totalDraftArtifacts"),
                "duplicateIds": (case_factory.get("summary") or {}).get("duplicateIds"),
            },
            "draftValidation": {
                "totalDraftArtifacts": (draft_validation.get("summary") or {}).get("totalDraftArtifacts"),
                "promotionReady": (draft_validation.get("summary") or {}).get("promotionReady"),
                "blocked": (draft_validation.get("summary") or {}).get("blocked"),
                "promotionReadyRate": (draft_validation.get("summary") or {}).get("promotionReadyRate"),
                "benchmarkPromotionReady": (draft_validation.get("summary") or {}).get("benchmarkPromotionReady"),
                "speechPromotionReady": (draft_validation.get("summary") or {}).get("speechPromotionReady"),
                "audioRecordingReady": (draft_validation.get("summary") or {}).get("audioRecordingReady"),
                "schemaBlocked": (draft_validation.get("summary") or {}).get("schemaBlocked"),
                "scoringBlocked": (draft_validation.get("summary") or {}).get("scoringBlocked"),
            },
            "suitePromotion": {
                "dryRun": (suite_promotion.get("summary") or {}).get("dryRun"),
                "totalCandidates": (suite_promotion.get("summary") or {}).get("totalCandidates"),
                "totalAdded": (suite_promotion.get("summary") or {}).get("totalAdded"),
                "totalSkipped": (suite_promotion.get("summary") or {}).get("totalSkipped"),
                "benchmarkAdded": (suite_promotion.get("summary") or {}).get("benchmarkAdded"),
                "speechAdded": (suite_promotion.get("summary") or {}).get("speechAdded"),
                "audioQueued": (suite_promotion.get("summary") or {}).get("audioQueued"),
                "wroteFiles": (suite_promotion.get("summary") or {}).get("wroteFiles"),
                "readyToWrite": (suite_promotion.get("summary") or {}).get("readyToWrite"),
            },
            "realAudio": {
                "total": audio_summary.get("total"),
                "evaluated": audio_summary.get("evaluated"),
                "skipped": audio_summary.get("skipped"),
                "passed": audio_summary.get("passed"),
                "passRate": _rate(audio_summary.get("passRate")),
                "asrPassRate": _rate(audio_summary.get("asrPassRate")),
                "downstreamTaskSuccess": _rate(audio_summary.get("downstreamTaskSuccess")),
                "avgWer": audio_summary.get("avgWer"),
                "avgEntityRecall": audio_summary.get("avgEntityRecall"),
                "avgSurfaceWer": audio_summary.get("avgSurfaceWer"),
                "avgSurfaceEntityRecall": audio_summary.get("avgSurfaceEntityRecall"),
                "avgCanonicalWer": audio_summary.get("avgCanonicalWer"),
                "avgCanonicalEntityRecall": audio_summary.get("avgCanonicalEntityRecall"),
                "multilingualScored": audio_summary.get("multilingualScored"),
                "byLanguage": audio_summary.get("byLanguage"),
                "semanticTranscriptPassed": audio_summary.get("semanticTranscriptPassed"),
                "semanticTranscriptPassRate": _rate(audio_summary.get("semanticTranscriptPassRate")),
                "semanticRecoveredAsrMisses": audio_summary.get("semanticRecoveredAsrMisses"),
                "semanticRecoveryRate": _rate(audio_summary.get("semanticRecoveryRate")),
                "avgSemanticScore": audio_summary.get("avgSemanticScore"),
                "avgSemanticIntentScore": audio_summary.get("avgSemanticIntentScore"),
                "avgSemanticSlotScore": audio_summary.get("avgSemanticSlotScore"),
                "avgSemanticCanonicalScore": audio_summary.get("avgSemanticCanonicalScore"),
                "bySemanticLabel": audio_summary.get("bySemanticLabel"),
                "deepgramP95Ms": (audio_summary.get("latency") or {}).get("deepgramP95Ms"),
                "voiceP95Ms": (audio_summary.get("latency") or {}).get("voiceP95Ms"),
                "costPer1000Turns": (audio_summary.get("cost") or {}).get("per1000VapiStack"),
            },
            "combined": {
                "totalCases": combined.get("count"),
                "costPer1000Turns": combined.get("per1000VapiStack"),
                "openaiRealtimePer1000Turns": combined.get("per1000OpenAIRealtime"),
                "geminiLivePer1000Turns": combined.get("per1000GeminiLive"),
                "savingsVsOpenAIRealtime": _safe_savings(combined.get("per1000VapiStack"), combined.get("per1000OpenAIRealtime")),
                "savingsVsGeminiLive": _safe_savings(combined.get("per1000VapiStack"), combined.get("per1000GeminiLive")),
                "voiceP95Ms": combined.get("voiceP95Ms"),
                "readinessScore": readiness.get("score"),
            },
        },
        "readiness": readiness,
        "tables": {
            "coreMetrics": core_metrics,
            "costComparison": cost_comparison,
            "benchmarkByGroup": _group_rows(benchmark_summary, "byGroup"),
            "speechByCondition": _group_rows(speech_summary, "byCondition"),
            "realAudioByCondition": _group_rows(audio_summary, "byCondition"),
            "audioPromptCoverage": audio_manifest.get("promptCoverage", []),
            "audioConditionCoverage": audio_manifest.get("conditionCoverage", []),
            "audioQualityRows": audio_quality.get("retakeQueue", [])[:24],
            "audioAcceptedRows": audio_accepted.get("acceptedRows", [])[:24],
            "audioAcceptedGroups": audio_accepted.get("groups", []),
            "audioErrorRows": audio_error.get("rows", [])[:24],
            "audioErrorActionPlan": audio_error.get("actionPlan", []),
            "audioErrorByFailure": audio_error.get("byFailure", []),
            "audioErrorConditionRisks": audio_error.get("conditionRisks", []),
            "audioErrorCoverageGaps": audio_error.get("acceptedCoverageGaps", []),
            "audioRobustnessByAugmentation": audio_robustness.get("byAugmentation", []),
            "audioRobustnessRows": audio_robustness.get("rows", []),
            "statisticsIntervals": statistics_pack.get("metrics", []),
            "claimReadiness": claim_readiness.get("claims", []),
            "claimActionPlan": claim_readiness.get("actionPlan", []),
            "experimentPlan": experiment_plan.get("workItems", []),
            "experimentPhases": experiment_plan.get("phases", []),
            "recordingQueue": experiment_plan.get("recordingQueue", []),
            "caseFactory": [
                {
                    "artifact": "Benchmark draft cases",
                    "count": (case_factory.get("summary") or {}).get("benchmarkDraftCases", 0),
                    "paperUse": "Candidate additions for task-success and latency claims",
                },
                {
                    "artifact": "Speech proxy draft cases",
                    "count": (case_factory.get("summary") or {}).get("speechDraftCases", 0),
                    "paperUse": "Candidate accent/noise transcript proxy additions",
                },
                {
                    "artifact": "Audio recording prompts",
                    "count": (case_factory.get("summary") or {}).get("audioRecordingPrompts", 0),
                    "paperUse": "Prompt queue for provider-ASR recording collection",
                },
            ],
            "caseFactoryRows": [
                *case_factory.get("benchmarkCases", [])[:8],
                *case_factory.get("speechCases", [])[:8],
                *case_factory.get("audioRecordingPrompts", [])[:8],
            ],
            "draftValidation": [
                {
                    "artifact": "Benchmark draft cases",
                    "ready": (draft_validation.get("summary") or {}).get("benchmarkPromotionReady", 0),
                    "blocked": (draft_validation.get("summary") or {}).get("benchmarkDrafts", 0) - (draft_validation.get("summary") or {}).get("benchmarkPromotionReady", 0),
                    "promotionTarget": "benchmarks/eval_cases.json",
                },
                {
                    "artifact": "Speech proxy draft cases",
                    "ready": (draft_validation.get("summary") or {}).get("speechPromotionReady", 0),
                    "blocked": (draft_validation.get("summary") or {}).get("speechDrafts", 0) - (draft_validation.get("summary") or {}).get("speechPromotionReady", 0),
                    "promotionTarget": "benchmarks/speech_cases.json",
                },
                {
                    "artifact": "Audio recording prompts",
                    "ready": (draft_validation.get("summary") or {}).get("audioRecordingReady", 0),
                    "blocked": (draft_validation.get("summary") or {}).get("audioPrompts", 0) - (draft_validation.get("summary") or {}).get("audioRecordingReady", 0),
                    "promotionTarget": "Real Audio Eval recording queue",
                },
            ],
            "draftValidationRows": draft_validation.get("rows", [])[:24],
            "suitePromotion": suite_promotion.get("rows", []),
            "suitePromotionSkipped": suite_promotion.get("skipped", [])[:24],
            "failureAnalysis": _failure_rows(benchmark, speech, audio),
        },
        "limitations": [
            "Speech robustness uses transcript proxies until recorded audio fixtures are collected.",
            "Provider cost baselines are ledger estimates and should be refreshed for final paper submission.",
            "The current live suite is curated for repeatable demos; larger generated task pools should be used for external claims.",
        ],
        "artifacts": {
            "json": str(LATEST_JSON_PATH.relative_to(ROOT_DIR)),
            "markdown": str(LATEST_MD_PATH.relative_to(ROOT_DIR)),
            "coreMetricsCsv": str(CORE_CSV_PATH.relative_to(ROOT_DIR)),
            "costComparisonCsv": str(COST_CSV_PATH.relative_to(ROOT_DIR)),
            "audioDatasetManifest": (audio_manifest.get("artifacts") or {}).get("json"),
            "audioDatasetCsv": (audio_manifest.get("artifacts") or {}).get("csv"),
            "audioQualityJson": (audio_quality.get("artifacts") or {}).get("json"),
            "audioQualityCsv": (audio_quality.get("artifacts") or {}).get("csv"),
            "retakeQueueJson": (audio_quality.get("artifacts") or {}).get("retakeQueueJson"),
            "retakeQueueCsv": (audio_quality.get("artifacts") or {}).get("retakeQueueCsv"),
            "audioAcceptedJson": (audio_accepted.get("artifacts") or {}).get("json"),
            "audioAcceptedCsv": (audio_accepted.get("artifacts") or {}).get("csv"),
            "audioErrorJson": (audio_error.get("artifacts") or {}).get("json"),
            "audioErrorCsv": (audio_error.get("artifacts") or {}).get("csv"),
            "audioErrorActionPlanCsv": (audio_error.get("artifacts") or {}).get("actionPlanCsv"),
            "audioRobustnessJson": (audio_robustness.get("artifacts") or {}).get("json"),
            "audioRobustnessCsv": (audio_robustness.get("artifacts") or {}).get("csv"),
            "statisticsJson": (statistics_pack.get("artifacts") or {}).get("json"),
            "statisticsCsv": (statistics_pack.get("artifacts") or {}).get("csv"),
            "claimsJson": (claim_readiness.get("artifacts") or {}).get("json"),
            "claimsCsv": (claim_readiness.get("artifacts") or {}).get("csv"),
            "experimentPlanJson": (experiment_plan.get("artifacts") or {}).get("json"),
            "experimentPlanCsv": (experiment_plan.get("artifacts") or {}).get("csv"),
            "caseFactoryJson": (case_factory.get("artifacts") or {}).get("json"),
            "caseFactoryCsv": (case_factory.get("artifacts") or {}).get("csv"),
            "benchmarkDrafts": (case_factory.get("artifacts") or {}).get("benchmarkDrafts"),
            "speechDrafts": (case_factory.get("artifacts") or {}).get("speechDrafts"),
            "audioPromptQueue": (case_factory.get("artifacts") or {}).get("audioPromptQueue"),
            "draftValidationJson": (draft_validation.get("artifacts") or {}).get("json"),
            "draftValidationCsv": (draft_validation.get("artifacts") or {}).get("csv"),
            "promotionManifestJson": (draft_validation.get("artifacts") or {}).get("promotionManifestJson"),
            "promotionManifestCsv": (draft_validation.get("artifacts") or {}).get("promotionManifestCsv"),
            "suitePromotionJson": (suite_promotion.get("artifacts") or {}).get("json"),
            "suitePromotionCsv": (suite_promotion.get("artifacts") or {}).get("csv"),
            "promotedAudioQueue": (suite_promotion.get("artifacts") or {}).get("promotedAudioQueue"),
        },
    }

    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with LATEST_JSON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        LATEST_MD_PATH.write_text(_markdown_report(payload), encoding="utf-8")
        _write_csv(CORE_CSV_PATH, core_metrics)
        _write_csv(COST_CSV_PATH, cost_comparison)
    return payload
