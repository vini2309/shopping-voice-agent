# VoiceRetailBench Research Track

This folder turns the current AislePilot demo into a paper-grade research project.

Working title:

> VoiceRetailBench: Cost-Aware and Accent-Robust Evaluation of Tool-Calling Voice Agents for Grounded Retail Assistance

The current Vapi demo is the baseline, not the final contribution. The paper should stand out by evaluating four dimensions together:

1. Task completion for grounded retail inventory questions.
2. Accent and noise robustness.
3. End-of-user-speech to first-audio latency.
4. Cost per minute and cost per successful task.

The current RAG implementation has moved beyond demo top-k retrieval. It now uses query planning, query rewriting/decomposition, hybrid sparse+dense retrieval, reciprocal-rank fusion, reranking, context compression, evidence validation, and cited prompt contracts.

## Files

- `paper_blueprint.md` - draft paper structure, thesis, claims, and contribution.
- `datasets_and_protocol.md` - datasets to use, what the referenced papers used, and how to adapt them.
- `implementation_roadmap.md` - one-by-one build plan from current Vapi baseline to LiveKit/open-model/self-host variants.
- `experiment_matrix.md` - paper-level benchmark suites, metrics, ablations, and architecture comparisons.
- `metrics.md` - exact metrics, logging fields, and evaluation tables.
- `rag_strategy.md` - hybrid structured-tool + RAG architecture for enterprise knowledge.
