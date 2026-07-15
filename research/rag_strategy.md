# Hybrid Tool + RAG Strategy

## Why RAG Is Required

The current inventory table is intentionally narrow: exact product, aisle, bay, stock, and notes. Real companies do not only answer table questions. They answer from:

- return policies
- promotions
- SOPs
- pickup and curbside instructions
- product manuals
- safety notices
- internal associate guides
- legal/compliance rules
- unstructured PDFs, web pages, chats, and tickets

That means the paper should not frame AislePilot as only a structured tool agent. It should frame the system as:

> a hybrid voice agent that chooses between structured tools and RAG under latency, cost, and accent-noise constraints.

## What We Implemented

The repo now has a local knowledge-base retrieval tool:

- `backend/app/knowledge.py`
- `backend/app/data/knowledge/*.md`
- `/api/knowledge/search`
- Vapi tool: `search_knowledge`
- benchmark source-level scoring in `benchmarks/evaluate_text_tasks.py`
- ASR-proxy stress tasks in `benchmarks/rag_asr_stress_tasks.jsonl`

The current version is an advanced local evidence pipeline:

- semantic section chunking from markdown knowledge documents
- per-chunk metadata for source ID, page/section, document type, top topic, access level, and timestamp
- query intent and topic planning before retrieval
- query expansion, query decomposition, and HyDE-style hypothetical query text
- hybrid sparse+dense candidate retrieval using exact token evidence plus local hashed token/character vectors
- reciprocal-rank fusion across original, rewritten, decomposed, and hypothetical queries
- metadata-aware reranking for topic fit, section fit, and evidence density
- context compression to keep only the most relevant sentences and evidence signatures
- evidence validation for confidence, coverage, PII pattern checks, schema validity, and abstention
- prompt-contract output that tells the LLM to answer only from cited evidence

It is still local and reproducible, so it can run without a paid embedding API or vector database. The production upgrade path is to swap the local vector scorer for a neural embedding index and add a learned cross-encoder or LLM reranker while preserving the same evidence contract.

Latest local benchmark result:

- seed suite: 7/7
- ASR-proxy RAG stress suite: 7/7
- generated paper suite: 593/593
- paper-suite RAG recall@k: 1.0
- paper-suite RAG MRR: 0.873
- paper-suite RAG nDCG@k: 0.901

## Paper-Grounded RAG Progression

### RAG / REALM

RAG and REALM establish the core idea: model parameters are not enough for knowledge-intensive tasks, and retrieval gives modular, interpretable, updateable external memory.

Use in our paper:

- Justify why enterprise voice agents need external knowledge.
- Compare parametric-only answering vs retrieved-context answering.
- Measure citation/source accuracy.

### Self-RAG

Self-RAG argues that always retrieving a fixed number of passages can hurt. The model should decide when retrieval is needed and critique whether passages support the answer.

Use in our paper:

- Add an ablation for tool choice:
  - no retrieval
  - always retrieve
  - route between inventory, RAG, clarification, and cannot-answer
- Measure over-retrieval and under-retrieval.

### CRAG

CRAG introduces a retrieval evaluator that checks whether retrieved documents are good enough and triggers corrective actions when they are not.

Use in our paper:

- Add retrieval confidence.
- If confidence is low, ask a clarification or abstain instead of hallucinating.
- Measure unsupported-answer rate.

Implemented now:

- `validation.needsClarification` is raised when confidence or coverage is weak.
- unsupported services such as fishing licenses, money transfers, auto-center warranties, pharmacy appointments, photo copyright rules, and check-cashing fees abstain instead of retrieving weak fuzzy matches.

### HyDE And Query Expansion

HyDE and query2doc show that generated or expanded pseudo-documents can improve retrieval, especially when the original question is short, messy, or mismatched with document wording.

Use in our paper:

- Compare raw-query retrieval against rewritten/decomposed/HyDE-style retrieval.
- Measure ASR-proxy robustness for phrases such as "curb side", "wheel chair", and "electric item".

Implemented now:

- Query expansion maps spoken variants and retail synonyms into retrievable evidence terms.
- Hypothetical query text is only added when topic inference is strong, which prevents generic policy words from creating false support.

### Ragas

Ragas separates retrieval and generation evaluation: context relevance, faithfulness, answer quality, and retrieval quality.

Use in our paper:

- Add RAG metrics alongside voice metrics:
  - context precision
  - context recall
  - faithfulness
- answer relevancy
- source accuracy

Implemented now:

- source recall@k, precision@k, top-1 accuracy, MRR, nDCG@k, confidence, margin, and unsupported-answer errors are logged by `benchmarks/evaluate_text_tasks.py`.

### Context Compression

LongLLMLingua motivates compressing long context to reduce cost and latency while preserving key evidence.

Use in our paper:

- Compare full chunks versus compressed evidence snippets.
- Measure whether compression improves latency and keeps source accuracy stable.

Implemented now:

- The retrieval tool returns full text plus `compressedText` and an evidence signature for every cited chunk.

### SpeechRAG / WavRAG / VoxRAG

These papers point to the main voice-specific RAG problem: if ASR is wrong, text RAG retrieves the wrong passages. SpeechRAG and WavRAG explore direct speech/audio retrieval to bypass or reduce ASR propagation errors.

Use in our paper:

- Start with ASR -> text RAG baseline.
- Add accent/noise stress tests.
- Later prototype audio-aware retrieval:
  - store audio variants of KB questions
  - retrieve over speech embeddings
  - compare against ASR-text retrieval

## Proposed Novel Contribution

VoiceRetailBench should evaluate:

1. Structured tool success.
2. RAG source retrieval success.
3. Routing accuracy between inventory tool, RAG, clarification, and cannot-answer.
4. ASR-to-RAG degradation under accent/noise.
5. Cost and latency of retrieval in the voice loop.

That creates a stronger paper than a table-only inventory agent.

## Architecture Target

```text
User audio
  -> VAD / endpointing
  -> Streaming STT
  -> Router
       -> lookup_inventory for structured item facts
       -> search_knowledge for policies/SOPs/docs
       -> clarify when query is underspecified
       -> cannot_answer when unsupported
  -> LLM grounded answer
  -> Streaming TTS
```

## Next Implementation Steps

1. Add retrieval latency logging separately from LLM latency.
2. Add neural embeddings plus a reranker ablation while keeping the local retriever as the cheap baseline.
3. Add clean human audio, accented audio, and noisy audio variants.
4. Compare ASR-text RAG against neural text embeddings.
5. Prototype audio-aware retrieval for high-WER turns.
6. Measure source accuracy, abstention quality, cost, and end-to-end latency.

## Sources

- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks: https://arxiv.org/abs/2005.11401
- REALM: Retrieval-Augmented Language Model Pre-Training: https://arxiv.org/abs/2002.08909
- Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection: https://arxiv.org/abs/2310.11511
- Corrective Retrieval Augmented Generation: https://arxiv.org/abs/2401.15884
- Ragas: Automated Evaluation of Retrieval Augmented Generation: https://arxiv.org/abs/2309.15217
- Precise Zero-Shot Dense Retrieval without Relevance Labels / HyDE: https://arxiv.org/abs/2212.10496
- Query2doc: Query Expansion with Large Language Models: https://arxiv.org/abs/2303.07678
- LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression: https://arxiv.org/abs/2310.06839
- A Hybrid Retrieval and Reranking Framework for Evidence-Grounded RAG: https://arxiv.org/abs/2605.01664
- Speech Retrieval-Augmented Generation without Automatic Speech Recognition: https://arxiv.org/abs/2412.16500
- WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models: https://arxiv.org/abs/2502.14727
- VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering: https://arxiv.org/abs/2505.17326
