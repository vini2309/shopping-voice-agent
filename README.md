---
title: Shopping Voice Agent
emoji: 🛒
colorFrom: gray
colorTo: pink
sdk: gradio
python_version: '3.12'
app_file: app.py
pinned: false
---

# AislePilot Shopping Agent

A brand-neutral shopping voice agent for one narrow task: help a customer find products, availability, aisle, bay, and nearby alternatives inside a synthetic retail store.

## Active Pipeline

- Orchestrator: Vapi Web SDK
- VAD: Silero V5 in the browser
- STT: Deepgram Nova-3 through Vapi
- LLM: OpenAI GPT-4o-mini through Vapi
- Tool: FastAPI `lookup_inventory` webhook with 127 enriched synthetic products, availability summaries, evidence rows, substitutes, and complements
- RAG: FastAPI `search_knowledge` webhook over synthetic store policy/SOP documents using advanced local hybrid retrieval, reciprocal-rank fusion, metadata reranking, context compression, and evidence validation
- Multilingual grounding: deterministic Spanish language detection plus retail phrase canonicalization before inventory/RAG tool calls
- TTS: ElevenLabs Turbo v2.5 through Vapi

Silero is functional, not decorative. Its speech callbacks gate the Vapi microphone path. Vapi then orchestrates Deepgram, GPT-4o-mini, the inventory webhook, the knowledge-search webhook, and ElevenLabs.

The app now uses a hybrid enterprise pattern:

- `lookup_inventory` for structured product, stock, aisle, bay, price facts, and review-grounded best-option recommendations.
- `catalog intelligence` for list-all/category questions, in-stock alternatives, complementary products, evidence rows, customer-review recommendation signals, and catalog coverage metrics.
- `search_knowledge` for unstructured policies, pickup/curbside, returns, price matching, accessibility, and associate SOPs.
- `trace recorder` for call events, transcripts, tool calls, evidence, costs, latency, and deterministic replay.
- `trust evaluator` for privacy, prompt-injection attempts, unsupported-answer handling, tool consistency, RAG grounding, replay stability, latency, and cost.
- `RAG Evidence Lab` for running messy policy questions and inspecting intent, query transformations, retrieval confidence, margin, validation, citations, compressed chunks, the prompt contract, and answer faithfulness grading.
- `RAG retrieval ablation` for comparing raw hybrid top-k, transformed RRF, and advanced metadata-aware reranking on the same query.
- `accepted audio set` for separating the raw recording archive from latest passing publishable fixtures after retakes.
- `audio error taxonomy` for separating language mismatch, prompt drift, ASR-only metric misses, downstream failures, and accepted-set coverage gaps.
- `multilingual canonicalization` for scoring Spanish utterances against English grounded tools while retaining literal surface-WER diagnostics.
- `semantic transcript scorer` for intent-equivalent ASR evaluation: strict WER remains, but paraphrases that preserve intent, slots, canonical query, and downstream task success are counted separately.
- `Accent sweep` for comparing Deepgram default, keyterm, accent-aware, multilingual, and retail transcript-repair profiles on the same saved audio clips.

The knowledge tool now behaves like a production evidence pipeline rather than a simple top-k demo: it plans intent/topic, rewrites and decomposes the query, retrieves with sparse+dense signals, reranks candidates, compresses evidence, returns citations, and abstains when support is weak.

## Vapi Setup

1. Create a Vapi account and copy the public key from the dashboard.
2. Create a dashboard assistant for the shopping agent and copy its assistant ID.
3. Configure that assistant with the provider credentials and the deployed `/api/vapi/webhook` tool URL.
4. Deploy the backend so `/api/vapi/webhook` has a public HTTPS URL. Vapi cannot call a localhost webhook.
5. Set `PUBLIC_VOICE_AGENT_KEY` and `PUBLIC_VOICE_ASSISTANT_ID` on the hosted service.

The public site starts the dashboard assistant by ID. The local/private lab can still create a transient assistant from `frontend/assistant-config.js` for experimentation, but no private provider key is placed in the browser.

## Showcase Website

The first screen is a brand-neutral shopping assistant website instead of the raw engineering console. Click **Ask AI** to open a simple live voice call card that behaves like a shopper-facing assistant.

The research dashboard is separate at `?lab=1`. Use that private link for traces, benchmark results, audio evaluation, RAG evidence inspection, and the cost/latency ledger.

The public website loads safe call configuration from the backend. Shoppers only click **Ask AI** and grant microphone permission. The owner sets the voice public key and dashboard assistant ID once in the hosted environment; private provider credentials and any later Featherless key stay in provider dashboards or the private service layer, not in the webpage.

## Research Track

See `research/` and `benchmarks/` for the paper-grade direction:

- VoiceRetailBench paper blueprint
- dataset and metric plan
- paper-level experiment matrix
- hybrid tool + RAG strategy
- generated paper benchmark rebuilt from the current enriched shopping catalog
- catalog-intelligence evaluator for relation coverage, available alternatives, complements, and evidence consistency
- text evaluator for inventory, knowledge retrieval, multi-tool routing, ASR-proxy, grouped metrics, RAG ranking metrics, and error taxonomy
- ASR transcript evaluator for WER, entity-WER, entity recall, and task success under transcript perturbations
- saved voice trace replay from `artifacts/traces` with deterministic tool-call match rates
- trace trust scoring for paper-ready safety and reliability tables
- adversarial trust traces for prompt injection, PII redaction, unsupported requests, and grounded RAG answers
- latest paper-suite result: 1223/1223 text-task success, 14/14 multi-tool success, RAG recall@k 1.0, 189 transcript-pair ASR robustness evaluation, and 98.4% catalog relation coverage

Generate adversarial trace tests:

```powershell
python benchmarks/generate_adversarial_traces.py --out artifacts/adversarial_trace_eval.json
```

Replay saved traces:

```powershell
python benchmarks/replay_traces.py --out artifacts/trace_replay_eval.json
```

Evaluate saved traces:

```powershell
python benchmarks/evaluate_traces.py --out artifacts/trace_trust_eval.json
```

Evaluate ASR transcript robustness:

```powershell
python benchmarks/evaluate_asr_transcripts.py --tasks benchmarks/generated/voice_retail_paper_tasks.jsonl --out artifacts/asr_transcript_eval.json
```

Evaluate catalog intelligence:

```powershell
python benchmarks/evaluate_catalog_intelligence.py --out artifacts/catalog_intelligence_eval.json
```

Generate the combined paper-table report:

```powershell
python benchmarks/generate_experiment_report.py
```

## Local Development

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd frontend
copy .env.example .env.local
pnpm install
pnpm dev
```

Open `http://localhost:5173`. A real Vapi inventory tool call requires `VITE_VAPI_TOOL_SERVER_URL` to point to the deployed HTTPS backend, not localhost.

Backend API docs are available at `http://localhost:8000/docs`. This project uses an interactive local docs page with built-in request examples, so endpoint testing still works when Swagger UI CDN assets are blocked.

Use the RAG Evidence Lab on `http://localhost:8000` to test policy questions such as `The shelf is empty but your system shows stock.` or `Can I buy a fishing license here?`; the panel shows whether retrieval is supported or should abstain.

## Deploy

Recommended public demo path:

- Deploy the root project with the root `render.yaml`. It builds the Vite website and FastAPI backend into one service.
- Set `PUBLIC_VOICE_AGENT_KEY` in the hosted environment.
- Set `PUBLIC_VOICE_ASSISTANT_ID` to the preconfigured voice assistant from your dashboard. This is the best public mode because the orchestration recipe stays out of the main website bundle.
- Keep `PUBLIC_VOICE_PROFILE_ID=burt` only for the private/local transient-assistant fallback.
- Do not set `PUBLIC_TOOL_SERVICE_URL` for the normal one-service deployment; the backend derives `https://your-site/api/vapi/webhook` automatically.
- Keep private provider keys in Vapi/provider dashboards. `DEEPGRAM_API_KEY` is only needed on the backend for the private recorded-audio evaluation tools.

Optional split deployment:

- Use Vercel for the static frontend and Render for the backend only if you specifically want two services.
- In that mode, set the frontend/backend origins explicitly and point the voice service URL at the backend HTTPS `/api/vapi/webhook`.

For local public testing, ngrok can expose `http://localhost:8000` as HTTPS. Open the ngrok HTTPS URL rather than `localhost` so the browser config and webhook are both public and secure.

## Featherless Later

The LLM selection is isolated in `buildModelConfig()` inside `frontend/assistant-config.js`. Replace the OpenAI model object with Vapi's `custom-llm` configuration and store the Featherless credential in Vapi; do not expose the Featherless API key in frontend environment variables.





