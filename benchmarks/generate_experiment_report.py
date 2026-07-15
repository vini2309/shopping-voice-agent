from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_MANIFEST = ROOT / "benchmarks" / "generated" / "voice_retail_paper_tasks.manifest.json"
DEFAULT_OUT_JSON = DEFAULT_ARTIFACTS / "voice_retail_experiment_report.json"
DEFAULT_OUT_MD = DEFAULT_ARTIFACTS / "voice_retail_experiment_report.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def summary(data: dict[str, Any]) -> dict[str, Any]:
    value = data.get("summary")
    return value if isinstance(value, dict) else {}


def pct(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value * 100:.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    rendered_rows = [[cell(item) for item in row] for row in rows]
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rendered_rows]
    return "\n".join([header, separator, *body])


def condition_rows(asr_eval: dict[str, Any]) -> list[list[str]]:
    by_condition = asr_eval.get("by_condition")
    if not isinstance(by_condition, dict):
        return []
    rows: list[list[str]] = []
    for condition, metrics in by_condition.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            [
                condition,
                num(metrics.get("transcriptPairs"), 0),
                pct(metrics.get("averageWer")),
                pct(metrics.get("averageEntityWer")),
                pct(metrics.get("averageEntityRecall")),
                pct(metrics.get("taskSuccessRate")),
            ]
        )
    return rows


def dataset_rows(text_eval: dict[str, Any]) -> list[list[str]]:
    by_dataset = text_eval.get("by_dataset")
    if not isinstance(by_dataset, dict):
        return []
    rows: list[list[str]] = []
    for dataset, metrics in by_dataset.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            [
                dataset,
                num(metrics.get("total"), 0),
                pct(metrics.get("success_rate")),
                pct(metrics.get("action_accuracy")),
                pct(metrics.get("item_accuracy")),
                pct(metrics.get("source_accuracy")),
            ]
        )
    return rows


def top_asr_risks(asr_eval: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    results = asr_eval.get("results")
    if not isinstance(results, list):
        return []
    ranked = sorted(
        [item for item in results if isinstance(item, dict)],
        key=lambda item: (
            item.get("entity_wer") if isinstance(item.get("entity_wer"), (int, float)) else -1,
            item.get("wer") if isinstance(item.get("wer"), (int, float)) else -1,
        ),
        reverse=True,
    )
    return ranked[:limit]


def build_report(
    *,
    manifest: dict[str, Any],
    text_eval: dict[str, Any],
    asr_eval: dict[str, Any],
    catalog_eval: dict[str, Any],
    trace_replay: dict[str, Any],
    trace_trust: dict[str, Any],
    adversarial: dict[str, Any],
) -> dict[str, Any]:
    text_summary = summary(text_eval)
    asr_summary = summary(asr_eval)
    catalog_summary = summary(catalog_eval)
    replay_summary = summary(trace_replay)
    trust_summary = summary(trace_trust)
    adversarial_summary = summary(adversarial)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    multi_tool_total = text_summary.get("multi_tool_total")
    multi_tool_rate = text_summary.get("multi_tool_success_rate")
    multi_tool_success = (
        round(multi_tool_total * multi_tool_rate)
        if isinstance(multi_tool_total, (int, float)) and isinstance(multi_tool_rate, (int, float))
        else None
    )

    return {
        "generatedAt": generated_at,
        "benchmark": {
            "version": manifest.get("version"),
            "tasks": manifest.get("task_count"),
            "transcriptPairs": manifest.get("transcript_pair_count"),
            "datasets": manifest.get("by_dataset", {}),
            "splits": manifest.get("by_split", {}),
            "accents": manifest.get("by_accent", {}),
            "noise": manifest.get("by_noise", {}),
        },
        "textTasks": text_summary,
        "asrRobustness": asr_summary,
        "catalogIntelligence": catalog_summary,
        "traceReplay": replay_summary,
        "traceTrust": trust_summary,
        "adversarialTrust": adversarial_summary,
        "derived": {
            "paperSuiteSuccess": f"{text_summary.get('success')}/{text_summary.get('total')}",
            "multiToolSuccess": f"{multi_tool_success}/{multi_tool_total}",
            "ragRecallAtK": text_summary.get("rag_source_recall_at_k"),
            "asrEntityRecall": asr_summary.get("averageEntityRecall"),
            "catalogRelationCoverage": catalog_summary.get("relationCoverage"),
            "traceTrustAverage": trust_summary.get("averageTrustScore"),
            "adversarialPassRate": (
                adversarial_summary.get("passCount") / adversarial_summary.get("generated")
                if adversarial_summary.get("generated")
                else None
            ),
        },
        "topAsrRisks": top_asr_risks(asr_eval),
    }


def markdown_report(report: dict[str, Any], source_paths: dict[str, str], text_eval: dict[str, Any], asr_eval: dict[str, Any]) -> str:
    benchmark = report["benchmark"]
    text = report["textTasks"]
    asr = report["asrRobustness"]
    catalog = report["catalogIntelligence"]
    replay = report["traceReplay"]
    trust = report["traceTrust"]
    adversarial = report["adversarialTrust"]

    coverage_rows = [
        ["Benchmark version", benchmark.get("version")],
        ["Total tasks", num(benchmark.get("tasks"), 0)],
        ["Transcript pairs", num(benchmark.get("transcriptPairs"), 0)],
        ["Datasets", num(len(benchmark.get("datasets", {})), 0)],
        ["Accent buckets", num(len(benchmark.get("accents", {})), 0)],
        ["Noise buckets", num(len(benchmark.get("noise", {})), 0)],
    ]
    text_rows = [
        ["Task success", f"{text.get('success')}/{text.get('total')}", pct(text.get("success_rate"))],
        ["Action accuracy", "-", pct(text.get("action_accuracy"))],
        ["Item accuracy", "-", pct(text.get("item_accuracy"))],
        ["Aisle accuracy", "-", pct(text.get("aisle_accuracy"))],
        ["Source accuracy", "-", pct(text.get("source_accuracy"))],
        ["Multi-tool success", report["derived"]["multiToolSuccess"], pct(text.get("multi_tool_success_rate"))],
    ]
    rag_rows = [
        ["RAG tasks", num(text.get("rag_total"), 0)],
        ["Recall@k", pct(text.get("rag_source_recall_at_k"))],
        ["Precision@k", pct(text.get("rag_source_precision_at_k"))],
        ["Top-1 source accuracy", pct(text.get("rag_source_top1_accuracy"))],
        ["MRR", num(text.get("rag_source_mrr"))],
        ["nDCG@k", num(text.get("rag_source_ndcg_at_k"))],
        ["Average retrieval confidence", num(text.get("avg_retrieval_confidence"))],
    ]
    asr_rows = [
        ["Transcript pairs", num(asr.get("transcriptPairs"), 0)],
        ["Unique tasks", num(asr.get("uniqueTasks"), 0)],
        ["Average WER", pct(asr.get("averageWer"))],
        ["Average entity-WER", pct(asr.get("averageEntityWer"))],
        ["Average entity recall", pct(asr.get("averageEntityRecall"))],
        ["Task success under transcript perturbation", pct(asr.get("taskSuccessRate"))],
        ["Max WER", pct(asr.get("maxWer"))],
        ["Max entity-WER", pct(asr.get("maxEntityWer"))],
    ]
    catalog_rows = [
        ["Catalog products", num(catalog.get("totalProducts"), 0)],
        ["Summary consistency", catalog.get("summaryConsistent")],
        ["Relation coverage", pct(catalog.get("relationCoverage"))],
        ["Alternative count", num(catalog.get("alternativeCount"), 0)],
        ["Alternative availability", pct(catalog.get("alternativeAvailabilityRate"))],
        ["Alternative semantic validity", pct(catalog.get("alternativeSemanticValidityRate"))],
        ["Complement count", num(catalog.get("complementCount"), 0)],
        ["Complement availability", pct(catalog.get("complementAvailabilityRate"))],
        ["Complement semantic validity", pct(catalog.get("complementSemanticValidityRate"))],
    ]
    trace_rows = [
        ["Trace replay deterministic match", f"{replay.get('deterministicMatches')}/{replay.get('toolCalls')}", pct(replay.get("deterministicMatchRate"))],
        ["Trace trust average", num(trust.get("averageTrustScore"), 1), "score"],
        ["Trace trust pass/review/fail", f"{trust.get('passCount')}/{trust.get('reviewCount')}/{trust.get('failCount')}", "counts"],
        ["Adversarial generated", num(adversarial.get("generated"), 0), "traces"],
        ["Adversarial pass/review/fail", f"{adversarial.get('passCount')}/{adversarial.get('reviewCount')}/{adversarial.get('failCount')}", "counts"],
        ["Adversarial average trust", num(adversarial.get("averageTrustScore"), 1), "score"],
    ]
    risk_rows = [
        [
            item.get("task_id", "-"),
            pct(item.get("wer")),
            pct(item.get("entity_wer")),
            str(item.get("reference_text", ""))[:80],
            str(item.get("transcript_text", ""))[:80],
        ]
        for item in report.get("topAsrRisks", [])
    ]

    sections = [
        "# VoiceRetailBench Experiment Report",
        f"Generated: `{report['generatedAt']}`",
        "## Benchmark Coverage",
        table(["Metric", "Value"], coverage_rows),
        "## Text Task Performance",
        table(["Metric", "Count", "Rate"], text_rows),
        "## RAG Retrieval",
        table(["Metric", "Value"], rag_rows),
        "## ASR Transcript Robustness",
        table(["Metric", "Value"], asr_rows),
        "## ASR By Condition",
        table(["Condition", "Pairs", "WER", "Entity-WER", "Entity Recall", "Task Success"], condition_rows(asr_eval)),
        "## Catalog Intelligence",
        table(["Metric", "Value"], catalog_rows),
        "## Dataset Breakdown",
        table(["Dataset", "Tasks", "Success", "Action", "Item", "Source"], dataset_rows(text_eval)),
        "## Trace And Trust",
        table(["Metric", "Value", "Unit"], trace_rows),
        "## Highest ASR Entity Risk Cases",
        table(["Task", "WER", "Entity-WER", "Reference", "Transcript"], risk_rows),
        "## Interpretation Notes",
        "- The current ASR numbers are transcript-proxy results, not yet real provider ASR over recorded audio.",
        "- The trace replay rate includes older live traces that did not persist full tool results, so adversarial traces are cleaner than the aggregate replay rate suggests.",
        "- The report is a baseline snapshot for paper tables; architecture comparisons should write additional reports with the same schema.",
        "## Source Artifacts",
        table(["Artifact", "Path"], [[name, path] for name, path in source_paths.items()]),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a paper-ready VoiceRetailBench experiment report.")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    paths = {
        "manifest": args.manifest,
        "text_eval": args.artifacts / "voice_retail_paper_eval.json",
        "asr_eval": args.artifacts / "asr_transcript_eval.json",
        "trace_replay": args.artifacts / "trace_replay_eval.json",
        "trace_trust": args.artifacts / "trace_trust_eval.json",
        "adversarial": args.artifacts / "adversarial_trace_eval.json",
        "catalog_intelligence": args.artifacts / "catalog_intelligence_eval.json",
    }
    manifest = read_json(paths["manifest"])
    text_eval = read_json(paths["text_eval"])
    asr_eval = read_json(paths["asr_eval"])
    catalog_eval = read_json(paths["catalog_intelligence"])
    trace_replay = read_json(paths["trace_replay"])
    trace_trust = read_json(paths["trace_trust"])
    adversarial = read_json(paths["adversarial"])

    report = build_report(
        manifest=manifest,
        text_eval=text_eval,
        asr_eval=asr_eval,
        catalog_eval=catalog_eval,
        trace_replay=trace_replay,
        trace_trust=trace_trust,
        adversarial=adversarial,
    )
    source_paths = {name: str(path) for name, path in paths.items()}
    report["sourceArtifacts"] = source_paths

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(markdown_report(report, source_paths, text_eval, asr_eval), encoding="utf-8")

    print(
        json.dumps(
            {
                "outJson": str(args.out_json),
                "outMarkdown": str(args.out_md),
                "tasks": report["benchmark"].get("tasks"),
                "transcriptPairs": report["benchmark"].get("transcriptPairs"),
                "successRate": report["textTasks"].get("success_rate"),
                "averageWer": report["asrRobustness"].get("averageWer"),
                "catalogRelationCoverage": report["catalogIntelligence"].get("relationCoverage"),
                "averageTrustScore": report["traceTrust"].get("averageTrustScore"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
