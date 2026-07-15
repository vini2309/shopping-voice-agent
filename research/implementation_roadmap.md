# Implementation Roadmap

## Phase 0 - Freeze Current Vapi Baseline

Status: mostly done.

Deliverables:

- Current Vapi app remains available.
- Log provider stack and ledger.
- Export call traces:
  - transcript
  - tool calls
  - answer text
  - latency segments
  - cost estimate

Purpose:

This is the managed-orchestrator baseline.

## Phase 1 - Benchmark Harness

Build before changing architecture.

Deliverables:

- `benchmarks/` folder.
- Dataset schema for products, tasks, audio variants, and traces.
- Offline text-task evaluator.
- Audio replay evaluator.
- Trace logger.

Minimum tests:

- 500 text tasks from generated retail scenarios.
- Current 50-item inventory as dev split.
- Expanded 1,000-item inventory as benchmark split.

Why first:

Without a harness, every architecture comparison is anecdotal.

## Phase 2 - Expanded Shopping Dataset

Status: enriched local catalog implemented.

Deliverables:

- Product catalog expansion from open sources and synthetic fields.
- Catalog-intelligence layer for broad category requests, available substitutes, complements, and evidence rows.
- Knowledge-base expansion from synthetic policy/SOP/product-guide documents.
- Generated user tasks with labels.
- Gold evaluator for exact, category, policy, and source-grounded answers.

Data sources:

- Open Food Facts / Open Pet Food Facts.
- Synthetic products for non-food departments.
- tau retail task style for multi-turn goals.

## Phase 2.5 - Enterprise RAG Layer

Status: first local baseline implemented.

Deliverables:

- `search_knowledge` tool.
- Source-level retrieval scoring.
- RAG tasks for returns, pickup, price match, substitutions, accessibility, and SOPs.
- Routing labels:
  - inventory tool
  - knowledge search
  - both tools
  - clarification
  - cannot-answer

Next upgrades:

- Expand catalog from 127 enriched products to 1,000+ products.
- Add relation labels for substitutes, complements, bundles, and unavailable-item recovery.
- Embedding retrieval and vector index.
- Retrieval confidence threshold.
- RAG faithfulness evaluator.
- RAG retrieval ablation endpoint and live UI comparing raw hybrid top-k, transformed RRF, and advanced reranking.
- Accent/noise stress tests for ASR-to-RAG degradation.

## Phase 2.6 - Retrieval Ablation And Faithfulness

Status: ablation endpoint and live UI implemented.

Deliverables:

- `/api/evaluation/rag-ablation` compares raw retrieval, query-transformed RRF, and full advanced reranking.
- RAG Evidence Lab displays support status, confidence, margin, source count, latency, and winner.
- Next: add answer faithfulness grading against compressed evidence and source signatures.
- Next: add stale-evidence and access-level-filter tests.

Purpose:

This turns advanced RAG from an implementation claim into a measurable ablation table for the paper.

## Phase 3 - LiveKit Orchestrator Variant

Deliverables:

- New backend service using LiveKit Agents.
- Same inventory tool.
- Same knowledge-search tool.
- Same frontend or a LiveKit-specific frontend mode.
- Same metrics trace format as Vapi.

Target:

- Lower orchestrator cost than Vapi.
- Similar or better latency.
- Better trace control.

## Phase 4 - Provider Ablations

STT:

- Deepgram Nova-3 baseline.
- AssemblyAI Universal-Streaming comparison.
- Sarvam for Indic/Indian accent if available.

LLM:

- GPT-4o-mini baseline.
- Groq/Featherless open model.
- vLLM/SGLang served open model.

TTS:

- ElevenLabs quality baseline.
- Deepgram Aura-2 cost baseline.
- Cartesia Sonic latency baseline.

## Phase 5 - Open LLM GPU Path

Deliverables:

- OpenAI-compatible vLLM or SGLang endpoint.
- Model candidates:
  - Qwen2.5/3 instruct family.
  - Llama 3.x instruct family.
  - smaller distilled model for TTFT.
- TTFT and cost-per-success comparison.

Important:

Self-host LLM first. Do not self-host every audio component immediately.

## Phase 6 - Accent And Noise Evaluation

Deliverables:

- Audio generation pipeline.
- FLEURS/Common Voice/AESRC real-speech probes where possible.
- Noise augmentation.
- Accent/noise tables.

## Phase 7 - Barge-In And Endpointing Evaluation

Deliverables:

- Scripted interruption cases.
- Measure stop-audio latency.
- Compare push-to-talk, fixed VAD, semantic endpointing, and full-duplex.

## Phase 8 - Speech-To-Speech Baselines

Deliverables:

- Gemini Live benchmark.
- GPT Realtime benchmark.
- Optional Moshi/Mini-Omni/Qwen-Omni feasibility runs.

Purpose:

Speech-to-speech is the future-facing baseline, not necessarily the production winner.

## Phase 9 - Paper Experiments

Experiment set:

1. Text vs voice task gap.
2. Vapi vs LiveKit vs self-host orchestration.
3. Structured tool vs RAG routing accuracy.
4. RAG source accuracy under clean/accent/noise conditions.
5. STT ablation by accent.
6. LLM ablation by cost and TTFT.
7. TTS ablation by first-audio and cost.
8. Noise and barge-in robustness.
9. Cost-per-success Pareto frontier.

## Phase 10 - Paper Writing

Write in this order:

1. Related work and benchmark section.
2. System and metrics section.
3. Dataset section.
4. Experimental setup.
5. Results once benchmark runs exist.
6. Abstract and conclusion last.

