# VoiceRetailBench Experiment Matrix

## Thesis

VoiceRetailBench evaluates retail voice agents as complete systems:

```text
speech or text input -> routing -> structured tool / RAG -> grounded answer -> speech output
```

The benchmark is designed to expose the tradeoff among accuracy, source grounding, latency, reliability, accent/noise robustness, and cost per successful task.

## Implemented Benchmark Suites

The current generated suite is produced by:

```powershell
python benchmarks/generate_paper_benchmark.py
```

It writes:

- `benchmarks/generated/voice_retail_paper_tasks.jsonl`
- `benchmarks/generated/voice_retail_paper_tasks.manifest.json`

Current generated scale:

- 1223 total tasks
- 1086 inventory/tool tasks
- 118 knowledge/RAG tasks
- 14 multi-tool product-plus-policy tasks
- 189 ASR reference/transcript pairs
- 127 catalog-intelligence relation probes
- 5 clarification tasks
- 30 multi-turn follow-up tasks
- 30 barge-in proxy tasks
- accent/noise proxy buckets for Indian English, Spanish L1 English, fast speech, store ambient noise, checkout beeps, and freezer hum

Latest advanced-RAG local result:

- seed suite: 8/8
- ASR-proxy RAG stress suite: 7/7
- generated paper suite: 1223/1223
- paper-suite multi-tool success: 14/14
- paper-suite RAG source recall@k: 1.0
- paper-suite RAG top-1 source accuracy: 0.788
- paper-suite RAG MRR: 0.879
- paper-suite RAG nDCG@k: 0.910
- ASR transcript pairs: 189
- ASR average WER: 0.014
- ASR average entity-WER: 0.026
- ASR average entity recall: 0.988
- ASR transcript task success: 1.0
- catalog relation coverage: 0.984
- catalog alternative availability: 1.0
- catalog complement availability: 1.0
- catalog alternative/complement non-self rate: 1.0

## Current Paper Metrics

Run:

```powershell
python benchmarks/evaluate_text_tasks.py --tasks benchmarks/generated/voice_retail_paper_tasks.jsonl --out artifacts/voice_retail_paper_eval.json --min-success-rate 0
```

The evaluator reports:

- task success
- action accuracy
- item accuracy
- aisle accuracy
- RAG source recall@k
- RAG source precision@k
- RAG top-1 source accuracy
- MRR
- nDCG@k
- average retrieval confidence
- average retrieval margin
- grouped metrics by dataset, tool, accent/noise condition
- tool-decision confusion matrix
- error taxonomy

Run transcript robustness:

```powershell
python benchmarks/evaluate_asr_transcripts.py --tasks benchmarks/generated/voice_retail_paper_tasks.jsonl --out artifacts/asr_transcript_eval.json
```

The transcript evaluator reports:

- WER
- entity-WER
- entity recall
- task success under transcript perturbation
- grouped transcript robustness by dataset and accent/noise condition

Run catalog intelligence:

```powershell
python benchmarks/evaluate_catalog_intelligence.py --out artifacts/catalog_intelligence_eval.json
```

The catalog evaluator reports:

- catalog summary consistency
- lookup found rate over all catalog items
- relation coverage
- alternative count and complement count
- in-stock alternative/complement rate
- non-self recommendation rate
- semantic validity for alternatives and complements

Generate combined paper tables:

```powershell
python benchmarks/generate_experiment_report.py
```

It writes:

- `artifacts/voice_retail_experiment_report.md`
- `artifacts/voice_retail_experiment_report.json`

Live-call traces are now recorded as replayable artifacts:

- browser events: VAD, transcripts, tool calls, tool results, assistant speech, ledger snapshots
- backend storage: `artifacts/traces/*.json`
- API: `/api/traces`, `/api/traces/{trace_id}`, `/api/traces/{trace_id}/replay`
- CLI: `python benchmarks/replay_traces.py --out artifacts/trace_replay_eval.json`
- trust CLI: `python benchmarks/evaluate_traces.py --out artifacts/trace_trust_eval.json`
- adversarial CLI: `python benchmarks/generate_adversarial_traces.py --out artifacts/adversarial_trace_eval.json`

This makes architecture comparisons repeatable: the same user turns and tool intents can be replayed after changing the orchestrator, model, retriever, or tool logic.

## Paper-Level Test Cases

### Structured Inventory

Purpose: test grounded tool calling and slot correctness.

Cases:

- exact product name
- synonym product name
- category wording
- list-all/category intent
- stock lookup
- zero-stock item
- unsupported product
- noisy ASR-like product spelling
- multi-turn follow-up with omitted referent
- barge-in proxy where final user intent replaces interrupted intent

Metrics:

- tool-call decision accuracy
- item ID accuracy
- aisle accuracy
- cannot-answer accuracy
- follow-up success
- barge-in proxy success

### Catalog Intelligence

Purpose: test whether the shopping agent can move beyond one exact SKU while still staying grounded in catalog evidence.

Implemented catalog layer:

- availability summary for every lookup result
- retrieval evidence rows with SKU, score, department, category, aisle, bay, stock, and tags
- in-stock substitutes ranked by product similarity, shelf proximity, and availability
- complementary related products ranked from hand-curated retail intent plus retrieval evidence
- catalog summary endpoint for product, department, aisle, and stock coverage

Current local result:

- catalog summary consistency: true
- lookup found rate: 1.0
- relation coverage: 0.984
- alternative availability rate: 1.0
- complement availability rate: 1.0
- alternative non-self rate: 1.0
- complement non-self rate: 1.0

Cases:

- broad category request, such as dog food
- exact product request with related alternatives
- item out of stock with available substitutes
- complementary add-on request
- evidence-only request where the assistant must cite product rows
- category inventory summary request

Metrics:

- relation coverage
- alternative availability rate
- complement availability rate
- non-self recommendation rate
- semantic relation validity
- evidence-row presence
- catalog summary consistency
- price, stock, and location consistency

### Enterprise RAG

Purpose: test policy/SOP retrieval beyond rows and columns.

Implemented retrieval pipeline:

- semantic chunk ingestion with source/page/section metadata
- intent and topic planning
- query expansion, decomposition, and HyDE-style hypothetical query text
- hybrid sparse+dense retrieval
- reciprocal-rank fusion
- metadata and evidence-density reranking
- context compression
- confidence/coverage/PII/schema evidence validation
- cited prompt contract for the answer model

Cases:

- returns and exchanges
- opened/used item policy
- electronics exception
- online pickup and curbside
- substitutions
- price differences
- competitor price match
- shelf-empty handling
- zero-on-hand handling
- accessibility services
- language help
- carryout help
- associate SOP
- unsupported services

Metrics:

- source recall@k
- source precision@k
- top-1 source accuracy
- MRR
- nDCG@k
- retrieval confidence
- retrieval margin
- abstention accuracy
- groundedness/support validation
- compressed-context token budget


### Retrieval Ablation

Purpose: prove which retrieval layers actually improve grounding for messy spoken retail questions.

Implemented comparison:

- raw hybrid top-k from the original query
- query expansion/decomposition/HyDE-style transformed queries with RRF
- full advanced retrieval with metadata-aware reranking, evidence density, compression, and validation

Metrics:

- support status
- confidence
- margin
- candidate count
- source count
- source overlap with expected labels when benchmark labels are available
- retrieval latency
- abstention behavior for unsupported requests

### Answer Faithfulness

Purpose: separate good retrieval from grounded final answering.

Implemented grader:

- maps each answer sentence to the strongest compressed evidence chunk
- reports the supporting evidence signature and source
- scores token and number support
- labels each sentence as supported, weakly supported, or unsupported
- returns faithful, review, unsupported, or no-answer verdicts

Metrics:

- faithfulness score
- grounded verdict rate
- unsupported-claim rate
- weak-support rate
- evidence-signature usage
- unsupported-number rate
### Routing

Purpose: test whether the agent chooses the correct action.

Labels:

- `lookup_inventory`
- `search_knowledge`
- `multi_tool`, for questions that require both inventory and policy, such as "The shelf is empty but the system says you have it."
- `request_info`
- `cannot_answer`

Metrics:

- routing confusion matrix
- over-retrieval rate
- under-retrieval rate
- unsupported-answer rate
- trace trust score
- PII leak rate
- prompt-injection attempt rate
- tool-answer consistency
- RAG grounding pass rate
- multi-tool action accuracy
- multi-tool item accuracy
- multi-tool source accuracy

### Adversarial Trust

Purpose: test whether the agent remains useful when customer turns contain unsafe or noisy instructions.

Implemented trace cases:

- prompt injection combined with a valid inventory lookup
- prompt injection combined with PII and a valid inventory lookup
- raw customer PII that must be redacted before trace storage
- unsupported service request that must abstain
- shelf-empty RAG question that must cite operational evidence

Metrics:

- raw PII leak rate
- prompt-injection leakage rate
- useful-answer preservation under attack
- unsupported-answer abstention accuracy
- trust score by attack category
- deterministic replay stability after backend changes

### Accent And Noise

Purpose: test ASR-to-tool and ASR-to-RAG degradation.

Current proxy conditions:

- Indian English proxy
- Spanish L1 English proxy
- fast-speech proxy
- store ambient proxy
- checkout beep proxy
- freezer hum proxy

Next real-audio conditions:

- clean recorded human speech
- TTS-generated variants
- Common Voice / FLEURS speaker probes where licensing permits
- MUSAN or store-noise overlays at 20 dB, 10 dB, and 5 dB SNR

Metrics:

- WER
- entity-WER
- entity recall
- task success drop from clean text
- tool argument accuracy drop
- RAG source accuracy drop
- cost and latency by condition

### Reliability

Purpose: avoid single-demo claims.

Run each architecture multiple times:

- `pass@1`
- `pass@3`
- `pass@5`
- `pass^k` for repeated reliability
- variance in latency and cost
- trace replay deterministic-match rate
- trace trust score variance

### Latency

Purpose: compare composed pipelines vs speech-to-speech APIs.

Measure:

- VAD start/end
- STT partial latency
- STT final latency
- retrieval/tool latency
- LLM first-token latency
- TTS first-audio latency
- end-of-user-speech to first audible audio
- full answer completion latency
- barge-in audio-stop latency

### Cost

Purpose: make the paper stand out beyond app-gluing.

Measure:

- cost per minute
- cost per turn
- cost per successful task
- provider cost split
- orchestrator cost split
- cost-latency-success Pareto frontier

## Architecture Comparisons

Run the same benchmark against:

1. Vapi managed cascade: Silero + Deepgram + GPT-4o-mini + ElevenLabs.
2. Custom orchestrator cascade: browser/client VAD + Deepgram/AssemblyAI + Groq/Featherless/vLLM + Cartesia/Deepgram Aura/ElevenLabs.
3. GPT Realtime.
4. Gemini Live.
5. Open-source LLM served through vLLM or SGLang.
6. Optional audio-aware RAG prototype for high-WER turns.

## Related Benchmark Inspirations

- RAG and REALM: external knowledge for knowledge-intensive tasks.
- Self-RAG and CRAG: retrieval decision, confidence, and abstention.
- RAGAS: retrieval quality and faithfulness metrics.
- SpeechRAG, WavRAG, VoxRAG: ASR degradation in spoken RAG.
- VoiceBench and EVA-Bench: end-to-end voice-agent evaluation.
- Full-Duplex-Bench: turn-taking, interruption, and latency.
- tau-bench: realistic tool-agent tasks with repeated reliability.
- FLEURS, Common Voice, and MUSAN: speech, accent, language, and noise robustness.

## Next Experiments To Implement

1. Add recorded/TTS audio generation and provider transcript ingestion.
2. Add final-answer semantic grading for spoken responses.
3. Add adversarial trust categories for tool poisoning, stale evidence, and access-level filtering.
4. Add custom orchestrator baseline using the same `/api/inventory` and `/api/knowledge` tools.
5. Add neural embedding RAG + reranker and compare against the local hybrid retriever.
6. Add cost-per-success tables for each architecture.
7. Expand multi-tool tasks into multi-turn customer goals with final-answer grading.


