from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .audio_accepted import build_audio_accepted_set, load_latest_audio_accepted_set
from .audio_error_analysis import build_audio_error_analysis, load_latest_audio_error_analysis
from .audio_eval import analyze_audio_robustness, build_audio_dataset_manifest, load_audio_cases, load_latest_audio_accent_sweep, load_latest_audio_eval, load_latest_audio_manifest, load_latest_audio_robustness, recording_audio_path, run_deepgram_accent_sweep, run_real_audio_suite, save_audio_recording
from .audio_quality import load_latest_audio_quality, run_audio_quality_gate
from .benchmark_suite import load_benchmark_cases, load_latest_benchmark, run_benchmark_suite
from .case_factory import generate_case_factory, load_latest_case_factory
from .claim_readiness import generate_claim_readiness_pack, load_latest_claim_readiness
from .draft_validation import load_latest_draft_validation, run_draft_validation
from .experiment_planner import generate_experiment_plan, load_latest_experiment_plan
from .inventory import catalog_summary, load_inventory, product_relations_payload, tool_payload
from .knowledge import compare_knowledge_retrieval, evaluate_answer_faithfulness, generate_evidence_gated_answer, search_knowledge
from .paper_report import generate_paper_results_pack, load_latest_paper_report
from .suite_promotion import load_latest_suite_promotion, run_suite_promotion
from .statistics_pack import generate_statistics_pack, load_latest_statistics_pack
from .pricing import PRICING_SOURCES, estimate_costs
from .speech_eval import load_latest_speech_eval, load_speech_cases, run_speech_robustness_suite
from .trace_eval import evaluate_saved_traces, evaluate_trace_id
from .traces import list_traces, load_trace, replay_trace, save_trace


ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
FEEDBACK_DIR = ROOT_DIR / "artifacts" / "feedback"
FEEDBACK_JSONL_PATH = FEEDBACK_DIR / "calls.jsonl"
FEEDBACK_SUMMARY_PATH = FEEDBACK_DIR / "summary.json"
load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / "backend" / ".env")

PII_REDACTIONS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED_CARD]"),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
]


def _frontend_origins() -> list[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://localhost:8000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _static_dir() -> Path:
    if (FRONTEND_DIST_DIR / "index.html").is_file():
        return FRONTEND_DIST_DIR
    return FRONTEND_DIR


def _message_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    message = payload.get("message")
    return message if isinstance(message, dict) else payload


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {"query": value}
    return {}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _feedback_text(value: Any, limit: int = 600) -> str:
    text = str(value or "").strip()
    for pattern, replacement in PII_REDACTIONS:
        text = pattern.sub(replacement, text)
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _load_feedback_rows() -> list[dict[str, Any]]:
    if not FEEDBACK_JSONL_PATH.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in FEEDBACK_JSONL_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _feedback_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row.get("score") or 0) for row in rows if isinstance(row.get("score"), (int, float))]
    total = len(scores)
    happy = sum(1 for score in scores if score >= 4)
    return {
        "total": total,
        "happy": happy,
        "unhappy": total - happy,
        "happyRate": round(happy / total, 4) if total else 0.0,
        "averageScore": round(sum(scores) / total, 3) if total else 0.0,
        "latestScore": scores[-1] if scores else None,
        "updatedAt": _utc_now(),
    }


def _save_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    score = int(payload.get("score") or 0)
    if score < 1 or score > 5:
        raise HTTPException(status_code=422, detail="score must be between 1 and 5")
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "feedbackId": f"feedback-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}",
        "createdAt": _utc_now(),
        "score": score,
        "happy": score >= 4,
        "comment": _feedback_text(payload.get("comment")),
        "endReason": _feedback_text(payload.get("endReason"), limit=80),
        "callSeconds": max(0, int(payload.get("callSeconds") or 0)),
        "turnCount": max(0, int(payload.get("turnCount") or 0)),
        "userText": _feedback_text(payload.get("userText")),
        "answer": _feedback_text(payload.get("answer")),
        "traceId": _feedback_text(payload.get("traceId"), limit=120),
    }
    with FEEDBACK_JSONL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    rows = _load_feedback_rows()
    summary = _feedback_summary(rows)
    FEEDBACK_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "saved": True,
        "feedback": row,
        "summary": summary,
        "artifacts": {
            "jsonl": str(FEEDBACK_JSONL_PATH.relative_to(ROOT_DIR)),
            "summary": str(FEEDBACK_SUMMARY_PATH.relative_to(ROOT_DIR)),
        },
    }


def _public_origin(request: Request) -> str:
    forwarded_host = request.headers.get("x-forwarded-host")
    host = forwarded_host or request.headers.get("host") or request.url.netloc
    forwarded_proto = request.headers.get("x-forwarded-proto")
    proto = (forwarded_proto.split(",")[0].strip() if forwarded_proto else request.url.scheme) or "https"
    if not host:
        return str(request.base_url).rstrip("/")
    return f"{proto}://{host}".rstrip("/")


def _public_tool_service_url(request: Request) -> str:
    configured_url = os.getenv("PUBLIC_TOOL_SERVICE_URL", "").strip()
    if configured_url:
        return configured_url
    return f"{_public_origin(request)}/api/vapi/webhook"


LOCAL_DOCS_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AislePilot API Docs</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f7f4ec;
        --panel: #fffdf8;
        --ink: #1d252c;
        --muted: #5f6b74;
        --line: #ded8ca;
        --accent: #145c57;
        --post: #2768a7;
        --get: #178052;
        --shadow: 0 16px 48px rgba(38, 32, 20, 0.12);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: var(--bg);
        color: var(--ink);
      }
      main {
        width: min(1120px, calc(100% - 32px));
        margin: 0 auto;
        padding: 32px 0 56px;
      }
      header {
        display: flex;
        justify-content: space-between;
        gap: 24px;
        align-items: end;
        margin-bottom: 24px;
      }
      h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1;
      }
      p { color: var(--muted); }
      a { color: var(--accent); font-weight: 700; }
      .links {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .links a, button {
        border: 1px solid var(--line);
        background: var(--panel);
        color: var(--ink);
        border-radius: 8px;
        padding: 10px 12px;
        text-decoration: none;
        cursor: pointer;
      }
      .toolbar {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 12px;
        margin-bottom: 16px;
      }
      input, textarea {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        padding: 12px 14px;
        font: inherit;
      }
      textarea {
        min-height: 150px;
        resize: vertical;
        font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
        line-height: 1.45;
      }
      .meta {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 18px;
      }
      .stat, details {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
      }
      .stat { padding: 14px; }
      .stat span {
        display: block;
        color: var(--muted);
        font-size: 0.82rem;
      }
      .stat strong {
        display: block;
        font-size: 1.25rem;
        margin-top: 4px;
      }
      .endpoints {
        display: grid;
        gap: 12px;
      }
      summary {
        display: grid;
        grid-template-columns: 74px 1fr;
        gap: 12px;
        align-items: center;
        padding: 14px;
        cursor: pointer;
      }
      summary::-webkit-details-marker { display: none; }
      .method {
        display: inline-flex;
        justify-content: center;
        border-radius: 6px;
        padding: 6px 8px;
        color: white;
        font-weight: 800;
        letter-spacing: 0;
      }
      .GET { background: var(--get); }
      .POST { background: var(--post); }
      .path {
        font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
        font-size: 0.98rem;
        overflow-wrap: anywhere;
      }
      .summary {
        color: var(--muted);
        margin-top: 4px;
      }
      .body {
        border-top: 1px solid var(--line);
        padding: 0 14px 14px;
      }
      pre {
        overflow: auto;
        background: #101820;
        color: #eef7f2;
        border-radius: 8px;
        padding: 12px;
        line-height: 1.45;
      }
      .try-panel {
        display: grid;
        gap: 12px;
        margin-top: 14px;
      }
      .param-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }
      .field {
        display: grid;
        gap: 6px;
      }
      .field label {
        color: var(--muted);
        font-size: 0.84rem;
        font-weight: 700;
      }
      .try-actions {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
      }
      .try-actions button {
        background: var(--accent);
        color: white;
        border-color: var(--accent);
        font-weight: 800;
      }
      .result-status {
        color: var(--muted);
        font-size: 0.92rem;
      }
      .sample-note {
        margin: 0;
        color: var(--muted);
        font-size: 0.9rem;
      }
      .try-output {
        min-height: 72px;
        white-space: pre-wrap;
      }
      .empty {
        border: 1px dashed var(--line);
        border-radius: 8px;
        padding: 24px;
        color: var(--muted);
        text-align: center;
      }
      @media (max-width: 720px) {
        header, .toolbar { grid-template-columns: 1fr; display: grid; align-items: start; }
        .meta { grid-template-columns: 1fr; }
        .param-grid { grid-template-columns: 1fr; }
        summary { grid-template-columns: 1fr; }
        .method { width: max-content; }
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <div>
          <h1>AislePilot API Docs</h1>
          <p id="subtitle">Loading local OpenAPI schema...</p>
        </div>
        <nav class="links" aria-label="API shortcuts">
          <a href="/">App</a>
          <a href="/openapi.json">OpenAPI JSON</a>
          <a href="/health">Health</a>
        </nav>
      </header>
      <section class="toolbar">
        <input id="filter" type="search" placeholder="Filter endpoints, for example traces or inventory" />
        <button id="expand" type="button">Expand all</button>
      </section>
      <section class="meta" aria-label="API summary">
        <div class="stat"><span>Version</span><strong id="version">-</strong></div>
        <div class="stat"><span>Endpoints</span><strong id="count">-</strong></div>
        <div class="stat"><span>Schema</span><strong>Local</strong></div>
      </section>
      <section id="endpoints" class="endpoints" aria-label="Endpoints"></section>
    </main>
    <script>
      const endpointsEl = document.querySelector("#endpoints");
      const filterEl = document.querySelector("#filter");
      const expandButton = document.querySelector("#expand");
      let endpoints = [];
      let expanded = false;

      function compactSchema(value) {
        if (!value) return null;
        if (value.$ref) return value.$ref;
        if (value.type) return value;
        return value;
      }

      function operationRows(schema) {
        const rows = [];
        for (const [path, methods] of Object.entries(schema.paths || {})) {
          for (const [method, operation] of Object.entries(methods || {})) {
            if (!["get", "post", "put", "patch", "delete"].includes(method)) continue;
            const requestBody = operation.requestBody?.content?.["application/json"]?.schema || null;
            const response = operation.responses?.["200"]?.content?.["application/json"]?.schema || null;
            rows.push({
              method: method.toUpperCase(),
              path,
              summary: operation.summary || operation.operationId || "",
              operationId: operation.operationId || "",
              parameters: operation.parameters || [],
              requestBody: compactSchema(requestBody),
              response: compactSchema(response)
            });
          }
        }
        return rows
          .sort((a, b) => `${a.path} ${a.method}`.localeCompare(`${b.path} ${b.method}`))
          .map((row, index) => ({ ...row, id: index }));
      }

      function escapeHtml(value) {
        return String(value)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#039;");
      }

      function parameterDefault(endpoint, parameter) {
        if (parameter.name === "trace_id") return "adv-injection-inventory";
        if (parameter.name === "limit") return endpoint.path.includes("/evaluation/") ? "100" : "20";
        return "";
      }

      function sampleBody(endpoint) {
        if (endpoint.method === "GET") return null;
        if (endpoint.path === "/api/inventory/lookup") return { query: "paper towels" };
        if (endpoint.path === "/api/knowledge/search") return { query: "shelf empty but system shows stock", limit: 3 };
        if (endpoint.path === "/api/knowledge/answer") return { query: "shelf empty but system shows stock", limit: 4 };
        if (endpoint.path === "/api/evaluation/rag-ablation") return { query: "shelf empty but system shows stock", limit: 4 };
        if (endpoint.path === "/api/evaluation/suite/run") return { groups: [], limit: 20, includePayloads: false, save: true };
        if (endpoint.path === "/api/evaluation/speech/run") return { groups: [], conditions: [], limit: 13, save: true };
        if (endpoint.path === "/api/evaluation/audio/recording") return { caseId: "audio_inv_paper_towels", audioBase64: "data:audio/webm;base64,...", mimeType: "audio/webm", durationMs: 2500 };
        if (endpoint.path === "/api/evaluation/audio/run") return { caseIds: [], limit: 20, allowReferenceFallback: false, deepgramConfig: { id: "accent_aware_keyterms_repair" }, enableTranscriptRepair: true, save: true };
        if (endpoint.path === "/api/evaluation/audio/accent-sweep") return { caseIds: [], limit: 8, includePassed: false, configs: [], allowReferenceFallback: false, save: true };
        if (endpoint.path === "/api/evaluation/audio/robustness") return { save: true };
        if (endpoint.path === "/api/evaluation/audio/quality") return { save: true };
        if (endpoint.path === "/api/evaluation/audio/accepted-set") return { save: true };
        if (endpoint.path === "/api/evaluation/audio/error-analysis") return { save: true };
        if (endpoint.path === "/api/evaluation/audio/manifest") return { targetPerPrompt: 3, save: true };
        if (endpoint.path === "/api/evaluation/report/run") return { rerunSuites: false, save: true };
        if (endpoint.path === "/api/evaluation/statistics/run") return { iterations: 1000, confidence: 0.95, save: true };
        if (endpoint.path === "/api/evaluation/claims/run") return { regenerateStatistics: false, save: true };
        if (endpoint.path === "/api/evaluation/experiment-plan/run") return { refreshClaims: false, save: true };
        if (endpoint.path === "/api/evaluation/case-factory/run") return { refreshPlan: false, save: true };
        if (endpoint.path === "/api/evaluation/draft-validation/run") return { refreshFactory: false, limit: null, includePayloads: false, save: true };
        if (endpoint.path === "/api/evaluation/promotion/run") return { dryRun: true, replaceFactoryCases: true, includeBenchmark: true, includeSpeech: true, includeAudioQueue: true, refreshValidation: false, save: true };
        if (endpoint.path === "/api/evaluation/faithfulness") return { query: "shelf empty but system shows stock", answer: "An associate should check topstock and nearby misplaced items before saying it is unavailable.", limit: 4 };
        if (endpoint.path === "/api/cost-estimate") {
          return {
            callMs: 60000,
            userText: "Where are paper towels?",
            answer: "Bounty Paper Towels are on aisle A twelve, bay four, with eighteen units available.",
            totalLatencyMs: 1200
          };
        }
        if (endpoint.path === "/api/vapi/webhook") {
          return {
            message: {
              type: "tool-calls",
              toolCallList: [
                {
                  id: "docs-lookup-1",
                  function: {
                    name: "lookup_inventory",
                    arguments: JSON.stringify({ query: "paper towels" })
                  }
                }
              ]
            }
          };
        }
        if (endpoint.path === "/api/traces") {
          return {
            traceId: "docs-demo-trace",
            source: "local-docs",
            architecture: "vapi-managed-cascade",
            status: "completed",
            durationMs: 1200,
            turnCount: 1,
            events: [
              { type: "transcript", relativeMs: 100, payload: { role: "user", text: "Where are paper towels?", transcriptType: "final" } },
              { type: "tool_call", relativeMs: 320, payload: { name: "lookup_inventory", query: "paper towels", arguments: { query: "paper towels" } } },
              { type: "tool_result", relativeMs: 520, payload: { tool: "lookup_inventory", query: "paper towels", found: true } },
              { type: "transcript", relativeMs: 900, payload: { role: "assistant", text: "Bounty Paper Towels are on aisle A twelve, bay four.", transcriptType: "final" } }
            ]
          };
        }
        return null;
      }

      function renderParameters(endpoint) {
        if (!endpoint.parameters.length) return '<p class="sample-note">No path or query parameters.</p>';
        return `
          <div class="param-grid">
            ${endpoint.parameters.map((parameter) => `
              <div class="field">
                <label for="param-${endpoint.id}-${parameter.name}">
                  ${escapeHtml(parameter.name)} (${escapeHtml(parameter.in || "query")})
                </label>
                <input
                  id="param-${endpoint.id}-${parameter.name}"
                  data-param-name="${escapeHtml(parameter.name)}"
                  data-param-in="${escapeHtml(parameter.in || "query")}"
                  value="${escapeHtml(parameterDefault(endpoint, parameter))}"
                />
              </div>
            `).join("")}
          </div>
        `;
      }

      function renderTryPanel(endpoint) {
        const body = sampleBody(endpoint);
        const bodyEditor = body
          ? `<div class="field">
              <label for="body-${endpoint.id}">JSON request body</label>
              <textarea id="body-${endpoint.id}" data-body>${escapeHtml(JSON.stringify(body, null, 2))}</textarea>
            </div>`
          : '<p class="sample-note">This endpoint does not need a request body.</p>';
        return `
          <section class="try-panel" data-endpoint-card="${endpoint.id}">
            <h3>Try it out</h3>
            ${renderParameters(endpoint)}
            ${bodyEditor}
            <div class="try-actions">
              <button type="button" onclick="tryEndpoint(${endpoint.id})">Try endpoint</button>
              <span id="status-${endpoint.id}" class="result-status">Ready</span>
            </div>
            <pre id="output-${endpoint.id}" class="try-output">Response will appear here.</pre>
          </section>
        `;
      }

      function render() {
        const needle = filterEl.value.trim().toLowerCase();
        const visible = endpoints.filter((endpoint) => {
          const haystack = `${endpoint.method} ${endpoint.path} ${endpoint.summary} ${endpoint.operationId}`.toLowerCase();
          return haystack.includes(needle);
        });
        document.querySelector("#count").textContent = String(visible.length);
        if (!visible.length) {
          endpointsEl.innerHTML = '<div class="empty">No endpoints match that filter.</div>';
          return;
        }
        endpointsEl.innerHTML = visible.map((endpoint, index) => `
          <details ${expanded ? "open" : ""}>
            <summary>
              <span class="method ${endpoint.method}">${endpoint.method}</span>
              <span>
                <span class="path">${endpoint.path}</span>
                <span class="summary">${endpoint.summary}</span>
              </span>
            </summary>
            <div class="body">
              <p><strong>Operation:</strong> ${endpoint.operationId || "-"}</p>
              ${renderTryPanel(endpoint)}
              <p><strong>Parameters:</strong></p>
              <pre>${JSON.stringify(endpoint.parameters, null, 2)}</pre>
              <p><strong>Request body:</strong></p>
              <pre>${JSON.stringify(endpoint.requestBody || "none", null, 2)}</pre>
              <p><strong>200 response:</strong></p>
              <pre>${JSON.stringify(endpoint.response || "schema not declared", null, 2)}</pre>
            </div>
          </details>
        `).join("");
      }

      async function tryEndpoint(id) {
        const endpoint = endpoints.find((item) => item.id === id);
        const card = document.querySelector(`[data-endpoint-card="${id}"]`);
        const status = document.querySelector(`#status-${id}`);
        const output = document.querySelector(`#output-${id}`);
        if (!endpoint || !card || !status || !output) return;

        let path = endpoint.path;
        const query = new URLSearchParams();
        for (const input of card.querySelectorAll("[data-param-name]")) {
          const name = input.dataset.paramName;
          const location = input.dataset.paramIn;
          const value = input.value.trim();
          if (location === "path") {
            path = path.replace(`{${name}}`, encodeURIComponent(value));
          } else if (value) {
            query.set(name, value);
          }
        }

        const url = `${path}${query.toString() ? `?${query.toString()}` : ""}`;
        const options = {
          method: endpoint.method,
          headers: { Accept: "application/json" }
        };
        const bodyInput = card.querySelector("[data-body]");
        if (bodyInput && bodyInput.value.trim()) {
          try {
            JSON.parse(bodyInput.value);
          } catch (error) {
            status.textContent = "Invalid JSON body";
            output.textContent = error.message;
            return;
          }
          options.headers["Content-Type"] = "application/json";
          options.body = bodyInput.value;
        }

        status.textContent = `Running ${endpoint.method} ${url}`;
        output.textContent = "Loading...";
        try {
          const startedAt = performance.now();
          const response = await fetch(url, options);
          const elapsedMs = Math.round(performance.now() - startedAt);
          const text = await response.text();
          let rendered = text;
          try {
            rendered = JSON.stringify(JSON.parse(text), null, 2);
          } catch (_) {}
          status.textContent = `${response.status} ${response.statusText} in ${elapsedMs} ms`;
          output.textContent = rendered || "(empty response)";
        } catch (error) {
          status.textContent = "Request failed";
          output.textContent = error.message;
        }
      }

      window.tryEndpoint = tryEndpoint;

      async function boot() {
        try {
          const response = await fetch("/openapi.json", { cache: "no-store" });
          const schema = await response.json();
          endpoints = operationRows(schema);
          document.querySelector("#subtitle").textContent = `${schema.info?.title || "API"} uses OpenAPI ${schema.openapi || ""}`;
          document.querySelector("#version").textContent = schema.info?.version || "-";
          render();
        } catch (error) {
          endpointsEl.innerHTML = `<div class="empty">Could not load /openapi.json: ${error.message}</div>`;
        }
      }

      filterEl.addEventListener("input", render);
      expandButton.addEventListener("click", () => {
        expanded = !expanded;
        expandButton.textContent = expanded ? "Collapse all" : "Expand all";
        render();
      });
      boot();
    </script>
  </body>
</html>
"""


app = FastAPI(title="AislePilot Vapi Tool Server", version="2.0.0", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/docs", include_in_schema=False)
@app.get("/docs/", include_in_schema=False)
async def local_docs() -> HTMLResponse:
    return HTMLResponse(LOCAL_DOCS_HTML, headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "shopping-voice-agent",
        "ready": True,
    }


@app.get("/api/catalog")
async def catalog() -> list[dict[str, Any]]:
    return load_inventory()


@app.get("/api/catalog/summary")
async def catalog_summary_endpoint() -> dict[str, Any]:
    return catalog_summary()


@app.post("/api/catalog/relations")
async def catalog_relations(request: Request) -> dict[str, Any]:
    payload = await request.json()
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    limit = int(payload.get("limit") or 4)
    return product_relations_payload(query, limit=max(1, min(limit, 8)))


@app.post("/api/inventory/lookup")
async def inventory_lookup(request: Request) -> dict[str, Any]:
    payload = await request.json()
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    return tool_payload(query)


@app.post("/api/knowledge/search")
async def knowledge_search(request: Request) -> dict[str, Any]:
    payload = await request.json()
    query = str(payload.get("query") or payload.get("question") or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    limit = int(payload.get("limit") or 3)
    return search_knowledge(query, limit=max(1, min(limit, 5)))


@app.post("/api/knowledge/answer")
async def knowledge_answer(request: Request) -> dict[str, Any]:
    payload = await request.json()
    query = str(payload.get("query") or payload.get("question") or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    limit = int(payload.get("limit") or 4)
    return generate_evidence_gated_answer(query, limit=max(1, min(limit, 6)))


@app.post("/api/vapi/webhook")
async def vapi_webhook(request: Request) -> dict[str, Any]:
    payload = await request.json()
    message = _message_from_payload(payload)
    message_type = message.get("type")

    if message_type in {"tool-calls", "function-call"}:
        calls = message.get("toolCallList") or message.get("toolCalls") or []
        if not calls and message.get("functionCall"):
            calls = [message["functionCall"]]

        results: list[dict[str, Any]] = []
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name") or call.get("name") or ""
            call_id = call.get("id") or call.get("toolCallId") or "lookup_inventory"
            arguments = _parse_arguments(function.get("arguments") or call.get("parameters"))

            query = str(arguments.get("query") or arguments.get("item") or "").strip()
            if name == "lookup_inventory":
                result = tool_payload(query)
            elif name == "search_knowledge":
                result = generate_evidence_gated_answer(query, limit=int(arguments.get("limit") or 4))
            else:
                results.append(
                    {
                        "toolCallId": call_id,
                        "name": name,
                        "error": f"Unsupported tool: {name}",
                    }
                )
                continue

            results.append(
                {
                    "toolCallId": call_id,
                    "name": name,
                    "result": json.dumps(result, separators=(",", ":")),
                    "metadata": {"matched": result.get("found", False)},
                }
            )

        return {"results": results}

    # Vapi sends other server messages when assistant.server is configured.
    return {"received": True}


@app.post("/api/cost-estimate")
async def cost_estimate(request: Request) -> dict[str, Any]:
    payload = await request.json()
    return estimate_costs(
        call_ms=int(payload.get("callMs") or 0),
        user_text=str(payload.get("userText") or ""),
        answer=str(payload.get("answer") or ""),
        total_latency_ms=payload.get("totalLatencyMs"),
    )


@app.get("/api/pricing")
async def pricing() -> dict[str, Any]:
    return {"sources": PRICING_SOURCES}


@app.get("/api/public-config")
async def public_config(request: Request) -> dict[str, Any]:
    public_key = os.getenv("PUBLIC_VOICE_AGENT_KEY") or os.getenv("VITE_VAPI_PUBLIC_KEY") or ""
    assistant_id = os.getenv("PUBLIC_VOICE_ASSISTANT_ID") or os.getenv("VITE_VAPI_ASSISTANT_ID") or ""
    voice_profile_id = os.getenv("PUBLIC_VOICE_PROFILE_ID") or os.getenv("VITE_ELEVENLABS_VOICE_ID") or ""
    service_url = _public_tool_service_url(request)
    return {
        "ready": bool(public_key and service_url and (assistant_id or voice_profile_id)),
        "voiceAgentPublicKey": public_key,
        "voiceAssistantId": assistant_id,
        "voiceProfileId": voice_profile_id,
        "serviceUrl": service_url,
    }


@app.post("/api/feedback")
async def feedback_save(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="feedback payload must be an object")
    return _save_feedback(payload)


@app.get("/api/feedback/summary")
async def feedback_summary() -> dict[str, Any]:
    rows = _load_feedback_rows()
    return {
        "found": bool(rows),
        "summary": _feedback_summary(rows),
        "artifacts": {
            "jsonl": str(FEEDBACK_JSONL_PATH.relative_to(ROOT_DIR)),
            "summary": str(FEEDBACK_SUMMARY_PATH.relative_to(ROOT_DIR)),
        },
    }


@app.post("/api/traces")
async def traces_save(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="trace payload must be an object")
    return save_trace(payload)


@app.get("/api/traces")
async def traces_list(limit: int = 20) -> dict[str, Any]:
    return list_traces(limit=max(1, min(limit, 100)))


@app.get("/api/traces/{trace_id}")
async def traces_get(trace_id: str) -> dict[str, Any]:
    try:
        return load_trace(trace_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="trace not found") from None


@app.post("/api/traces/{trace_id}/replay")
async def traces_replay(trace_id: str) -> dict[str, Any]:
    try:
        return replay_trace(trace_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="trace not found") from None




@app.post("/api/evaluation/faithfulness")
async def faithfulness_eval(request: Request) -> dict[str, Any]:
    payload = await request.json()
    query = str(payload.get("query") or payload.get("question") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    if not answer:
        raise HTTPException(status_code=422, detail="answer is required")
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else None
    limit = int(payload.get("limit") or 4)
    return evaluate_answer_faithfulness(query, answer, evidence=evidence, limit=max(1, min(limit, 6)))

@app.post("/api/evaluation/rag-ablation")
async def rag_ablation(request: Request) -> dict[str, Any]:
    payload = await request.json()
    query = str(payload.get("query") or payload.get("question") or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    limit = int(payload.get("limit") or 4)
    return compare_knowledge_retrieval(query, limit=max(1, min(limit, 6)))


@app.get("/api/evaluation/suite/cases")
async def benchmark_cases() -> dict[str, Any]:
    cases = load_benchmark_cases()
    groups = sorted({str(case.get("group") or case.get("type") or "unknown") for case in cases})
    return {"cases": cases, "count": len(cases), "groups": groups}


@app.get("/api/evaluation/suite/latest")
async def benchmark_latest() -> dict[str, Any]:
    return load_latest_benchmark()


@app.post("/api/evaluation/suite/run")
async def benchmark_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    groups = payload.get("groups") if isinstance(payload.get("groups"), list) else None
    case_ids = payload.get("caseIds") if isinstance(payload.get("caseIds"), list) else None
    limit_value = payload.get("limit")
    limit = int(limit_value) if limit_value not in (None, "") else None
    include_payloads = bool(payload.get("includePayloads", False))
    save = bool(payload.get("save", True))
    return run_benchmark_suite(
        groups=[str(group) for group in groups] if groups else None,
        case_ids=[str(case_id) for case_id in case_ids] if case_ids else None,
        limit=limit,
        include_payloads=include_payloads,
        save=save,
    )

@app.get("/api/evaluation/speech/cases")
async def speech_eval_cases() -> dict[str, Any]:
    cases = load_speech_cases()
    groups = sorted({str(case.get("group") or case.get("route") or "unknown") for case in cases})
    conditions = sorted(
        {
            f"{(case.get('condition') or {}).get('accent', 'unknown')}|"
            f"{(case.get('condition') or {}).get('noise', 'unknown')}|"
            f"barge:{bool((case.get('condition') or {}).get('bargeIn'))}"
            for case in cases
        }
    )
    return {"cases": cases, "count": len(cases), "groups": groups, "conditions": conditions}


@app.get("/api/evaluation/speech/latest")
async def speech_eval_latest() -> dict[str, Any]:
    return load_latest_speech_eval()


@app.post("/api/evaluation/speech/run")
async def speech_eval_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    groups = payload.get("groups") if isinstance(payload.get("groups"), list) else None
    conditions = payload.get("conditions") if isinstance(payload.get("conditions"), list) else None
    limit_value = payload.get("limit")
    limit = int(limit_value) if limit_value not in (None, "") else None
    save = bool(payload.get("save", True))
    return run_speech_robustness_suite(
        groups=[str(group) for group in groups] if groups else None,
        conditions=[str(condition) for condition in conditions] if conditions else None,
        limit=limit,
        save=save,
    )

@app.get("/api/evaluation/audio/cases")
async def audio_eval_cases() -> dict[str, Any]:
    return load_audio_cases(include_templates=True)


@app.get("/api/evaluation/audio/manifest/latest")
async def audio_manifest_latest() -> dict[str, Any]:
    return load_latest_audio_manifest()


@app.post("/api/evaluation/audio/manifest")
async def audio_manifest_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    target = int(payload.get("targetPerPrompt") or 3)
    save = bool(payload.get("save", True))
    return build_audio_dataset_manifest(target_per_prompt=target, save=save)

@app.get("/api/evaluation/audio/latest")
async def audio_eval_latest() -> dict[str, Any]:
    return load_latest_audio_eval()


@app.get("/api/evaluation/audio/robustness/latest")
async def audio_robustness_latest() -> dict[str, Any]:
    return load_latest_audio_robustness()


@app.get("/api/evaluation/audio/accent-sweep/latest")
async def audio_accent_sweep_latest() -> dict[str, Any]:
    return load_latest_audio_accent_sweep()


@app.post("/api/evaluation/audio/accent-sweep")
async def audio_accent_sweep_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    raw_case_ids = payload.get("caseIds") if isinstance(payload.get("caseIds"), list) else []
    raw_configs = payload.get("configs") if isinstance(payload.get("configs"), list) else []
    limit_value = payload.get("limit")
    limit = int(limit_value) if limit_value not in (None, "") else None
    return run_deepgram_accent_sweep(
        case_ids=[str(case_id) for case_id in raw_case_ids] if raw_case_ids else None,
        limit=limit,
        configs=raw_configs,
        include_passed=bool(payload.get("includePassed", False)),
        allow_reference_fallback=bool(payload.get("allowReferenceFallback", False)),
        save=bool(payload.get("save", True)),
    )


@app.post("/api/evaluation/audio/robustness")
async def audio_robustness_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    save = bool(payload.get("save", True))
    return analyze_audio_robustness(save=save)


@app.get("/api/evaluation/audio/quality/latest")
async def audio_quality_latest() -> dict[str, Any]:
    return load_latest_audio_quality()


@app.post("/api/evaluation/audio/quality")
async def audio_quality_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    save = bool(payload.get("save", True))
    return run_audio_quality_gate(save=save)


@app.get("/api/evaluation/audio/accepted-set/latest")
async def audio_accepted_set_latest() -> dict[str, Any]:
    return load_latest_audio_accepted_set()


@app.post("/api/evaluation/audio/accepted-set")
async def audio_accepted_set_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    save = bool(payload.get("save", True))
    return build_audio_accepted_set(save=save)


@app.get("/api/evaluation/audio/error-analysis/latest")
async def audio_error_analysis_latest() -> dict[str, Any]:
    return load_latest_audio_error_analysis()


@app.post("/api/evaluation/audio/error-analysis")
async def audio_error_analysis_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    save = bool(payload.get("save", True))
    return build_audio_error_analysis(save=save)


@app.post("/api/evaluation/audio/recording")
async def audio_eval_recording(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        return save_audio_recording(payload)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.get("/api/evaluation/audio/recordings/{recording_id}/file", include_in_schema=False)
async def audio_eval_recording_file(recording_id: str) -> FileResponse:
    try:
        path, mime_type = recording_audio_path(recording_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="recording audio not found") from None
    return FileResponse(path, media_type=mime_type)

@app.post("/api/evaluation/audio/run")
async def audio_eval_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    case_ids = payload.get("caseIds") if isinstance(payload.get("caseIds"), list) else None
    limit_value = payload.get("limit")
    limit = int(limit_value) if limit_value not in (None, "") else None
    allow_reference_fallback = bool(payload.get("allowReferenceFallback", False))
    save = bool(payload.get("save", True))
    return run_real_audio_suite(
        case_ids=[str(case_id) for case_id in case_ids] if case_ids else None,
        limit=limit,
        allow_reference_fallback=allow_reference_fallback,
        deepgram_config=payload.get("deepgramConfig") if isinstance(payload.get("deepgramConfig"), dict) else None,
        enable_transcript_repair=payload.get("enableTranscriptRepair") if "enableTranscriptRepair" in payload else None,
        save=save,
    )

@app.get("/api/evaluation/statistics/latest")
async def statistics_pack_latest() -> dict[str, Any]:
    return load_latest_statistics_pack()


@app.post("/api/evaluation/statistics/run")
async def statistics_pack_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    iterations = int(payload.get("iterations", 1000) or 1000)
    confidence = float(payload.get("confidence", 0.95) or 0.95)
    save = bool(payload.get("save", True))
    return generate_statistics_pack(iterations=iterations, confidence=confidence, save=save)


@app.get("/api/evaluation/claims/latest")
async def claim_readiness_latest() -> dict[str, Any]:
    return load_latest_claim_readiness()


@app.post("/api/evaluation/claims/run")
async def claim_readiness_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    regenerate_statistics = bool(payload.get("regenerateStatistics", False))
    save = bool(payload.get("save", True))
    return generate_claim_readiness_pack(regenerate_statistics=regenerate_statistics, save=save)


@app.get("/api/evaluation/experiment-plan/latest")
async def experiment_plan_latest() -> dict[str, Any]:
    return load_latest_experiment_plan()


@app.post("/api/evaluation/experiment-plan/run")
async def experiment_plan_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    refresh_claims = bool(payload.get("refreshClaims", False))
    save = bool(payload.get("save", True))
    return generate_experiment_plan(refresh_claims=refresh_claims, save=save)


@app.get("/api/evaluation/case-factory/latest")
async def case_factory_latest() -> dict[str, Any]:
    return load_latest_case_factory()


@app.post("/api/evaluation/case-factory/run")
async def case_factory_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    refresh_plan = bool(payload.get("refreshPlan", False))
    save = bool(payload.get("save", True))
    return generate_case_factory(refresh_plan=refresh_plan, save=save)


@app.get("/api/evaluation/draft-validation/latest")
async def draft_validation_latest() -> dict[str, Any]:
    return load_latest_draft_validation()


@app.post("/api/evaluation/draft-validation/run")
async def draft_validation_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    refresh_factory = bool(payload.get("refreshFactory", False))
    include_payloads = bool(payload.get("includePayloads", False))
    save = bool(payload.get("save", True))
    raw_limit = payload.get("limit")
    limit = int(raw_limit) if raw_limit not in (None, "", 0) else None
    return run_draft_validation(refresh_factory=refresh_factory, limit=limit, include_payloads=include_payloads, save=save)


@app.get("/api/evaluation/promotion/latest")
async def suite_promotion_latest() -> dict[str, Any]:
    return load_latest_suite_promotion()


@app.post("/api/evaluation/promotion/run")
async def suite_promotion_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    return run_suite_promotion(
        dry_run=bool(payload.get("dryRun", True)),
        replace_factory_cases=bool(payload.get("replaceFactoryCases", True)),
        include_benchmark=bool(payload.get("includeBenchmark", True)),
        include_speech=bool(payload.get("includeSpeech", True)),
        include_audio_queue=bool(payload.get("includeAudioQueue", True)),
        refresh_validation=bool(payload.get("refreshValidation", False)),
        save=bool(payload.get("save", True)),
    )


@app.get("/api/evaluation/report/latest")
async def paper_report_latest() -> dict[str, Any]:
    return load_latest_paper_report()


@app.post("/api/evaluation/report/run")
async def paper_report_run(request: Request) -> dict[str, Any]:
    payload = await request.json()
    rerun_suites = bool(payload.get("rerunSuites", False))
    save = bool(payload.get("save", True))
    return generate_paper_results_pack(rerun_suites=rerun_suites, save=save)

@app.get("/api/evaluation/traces")
async def traces_evaluate_all(limit: int = 100) -> dict[str, Any]:
    return evaluate_saved_traces(limit=max(1, min(limit, 200)))


@app.get("/api/evaluation/traces/{trace_id}")
async def traces_evaluate(trace_id: str) -> dict[str, Any]:
    try:
        return evaluate_trace_id(trace_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="trace not found") from None


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_static_dir() / "index.html")


@app.get("/{asset_path:path}", include_in_schema=False)
async def static_asset(asset_path: str) -> FileResponse:
    static_dir = _static_dir().resolve()
    path = (static_dir / asset_path).resolve()
    if path.is_file() and static_dir in path.parents:
        return FileResponse(path)
    return FileResponse(static_dir / "index.html")






