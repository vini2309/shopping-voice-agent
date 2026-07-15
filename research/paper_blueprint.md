# Paper Blueprint

## Working Title

VoiceRetailBench: Cost-Aware and Accent-Robust Evaluation of Tool-Calling Voice Agents for Grounded Retail Assistance

## Core Thesis

Modern voice-agent papers usually optimize only one part of the problem: speech-to-speech latency, tool-calling accuracy, or ASR robustness. A deployed retail voice agent must optimize all of them at once. We propose a reproducible benchmark and system study that evaluates cost, latency, accent robustness, and grounded tool success together.

## What Makes This Unique

The contribution should not be "we built a shopping voice agent." That is a demo. The paper contribution should be:

1. A domain-grounded retail voice benchmark with verifiable inventory/tool outcomes.
2. A cost-latency ledger that reports cost per successful task, not only cost per minute.
3. Accent/noise-conditioned evaluation using public speech datasets and controlled audio conversion.
4. A component ablation across orchestrators, STT, LLM, TTS, and open/self-hosted models.
5. An error taxonomy separating ASR errors, tool-decision errors, slot errors, endpointing errors, and grounded-response errors.
6. A hybrid structured-tool and RAG benchmark for enterprise knowledge, not only product rows.

## Proposed System Name

ARES-Retail:

Accent-Robust, Economical, Streaming Retail voice agent.

## Research Questions

RQ1. How much of retail voice-agent failure comes from ASR, tool-call policy, reasoning, endpointing, or speech generation?

RQ2. How do managed orchestration, open orchestration, and self-hosted components trade off cost, latency, and task success?

RQ3. Can a modular cascaded pipeline approach realtime speech-to-speech latency while retaining transcripts, tools, custom voices, and cost control?

RQ4. Which accent/noise conditions create the largest gap between text-agent task success and voice-agent task success?

RQ5. Is cost per successful task a better deployment metric than cost per minute?

RQ6. How much does ASR error under accent/noise degrade RAG source retrieval and grounded policy answering?

## System Variants

### V0 - Current Demo Baseline

- Orchestrator: Vapi
- VAD: Silero browser VAD
- STT: Deepgram Nova-3
- LLM: GPT-4o-mini
- TTS: ElevenLabs
- Tool: local inventory lookup
- RAG: local synthetic store knowledge base

Purpose: working demo and managed-orchestrator baseline.

### V1 - Open Orchestrator

- Orchestrator: LiveKit Agents
- VAD: Silero
- Turn detection: LiveKit turn detector
- STT/TTS/LLM same as V0 initially

Purpose: remove Vapi platform cost while keeping comparable provider stack.

### V2 - Cost-Optimized Cascade

- Orchestrator: LiveKit Agents
- STT: Deepgram Nova-3 or AssemblyAI Universal-Streaming
- LLM: GPT-4o-mini or Groq/Featherless open model
- TTS: Deepgram Aura-2 or Cartesia Sonic

Purpose: best cost/latency cascade with external providers.

### V3 - Open LLM Cascade

- Orchestrator: LiveKit Agents or Pipecat
- LLM: Qwen/Llama served through vLLM or SGLang on GPU
- STT/TTS remain provider-hosted

Purpose: measure whether self-hosting the brain matters when audio boxes dominate cost.

### V4 - Self-Hosted Audio Experiments

- STT: faster-whisper or another streaming-capable open ASR where feasible
- TTS: Kokoro/Piper/IndicF5-style open TTS where feasible
- LLM: vLLM/SGLang

Purpose: scale-path ablation, not first production target.

### V5 - Speech-to-Speech Baseline

- Gemini Live / GPT Realtime as hosted S2S baselines
- Optional open S2S: Moshi, Mini-Omni, Qwen-Omni if realtime serving is feasible

Purpose: compare "collapse the stack" against cascaded observability and cost.

## Key Claim To Test

For grounded retail tool tasks, a streaming cascaded architecture can achieve competitive latency and lower cost-per-success than native realtime speech-to-speech systems, while preserving tool observability and accent-specific component tuning.

This is a claim, not a conclusion. The benchmark decides whether it survives.

## Paper Skeleton

1. Abstract
2. Introduction
   - Retail voice agents need low latency, high task accuracy, accent robustness, and low cost.
   - Existing work evaluates pieces separately.
   - We introduce VoiceRetailBench and ARES-Retail.
3. Related Work
   - Speech-to-speech models: Moshi, Mini-Omni, Qwen2.5-Omni.
   - Practical cascaded voice agents: enterprise realtime voice-agent tutorial.
   - Voice/tool evaluation: From Text to Voice, When2Call, CONFETTI, tau-Voice.
   - Enterprise retrieval: RAG, REALM, Self-RAG, CRAG, Ragas, SpeechRAG.
   - Accent and multilingual speech datasets: FLEURS, AESRC, Common Voice.
4. Benchmark
   - Retail inventory tasks.
   - Data sources.
   - Audio generation and real-accent evaluation.
   - Noise and interruption conditions.
5. System
   - Vapi baseline.
   - LiveKit/open orchestrator.
   - Component variants.
   - Logging and ledger.
6. Metrics
   - Task success.
   - Slot accuracy.
   - WER.
   - End-to-first-audio latency.
   - Barge-in recovery.
   - Cost per minute and cost per successful task.
7. Experiments
   - Text vs voice gap.
   - Accent/noise robustness.
   - Orchestrator comparison.
   - STT/LLM/TTS ablation.
   - Cost-latency Pareto frontier.
8. Results
   - Use `artifacts/voice_retail_experiment_report.md` as the reproducible baseline table snapshot.
9. Error Analysis
10. Discussion
11. Limitations
12. Conclusion

## Honest Limitations To Include

- "All accents" cannot be claimed; we can claim performance over evaluated accent groups.
- Synthetic TTS accent data does not fully replace real accented speech.
- Public product datasets may require license care.
- Full self-hosted speech-to-speech is still difficult to serve in realtime on modest GPUs.
- Live demos can understate tail latency; paper results should report P50, P90, and P95.
