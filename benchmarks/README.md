# VoiceRetailBench Benchmark Artifacts

This folder starts the benchmark side of the paper.

- `voice_retail_task_schema.json` defines the task contract.
- `seed_tasks.jsonl` contains small dev examples mapped to the current AislePilot inventory.
- `replay_traces.py` replays saved browser/Vapi traces from `artifacts/traces` against the current tool and RAG implementation.
- `evaluate_traces.py` scores saved traces for privacy, injection attempts, unsupported-answer handling, tool consistency, RAG grounding, replay stability, latency, and cost.
- `generate_adversarial_traces.py` creates saved traces for prompt injection, PII redaction, unsupported requests, and grounded inventory/RAG answers.
- `evaluate_asr_transcripts.py` scores reference-vs-transcript pairs for WER, entity-WER, entity recall, and downstream task success.
- `evaluate_catalog_intelligence.py` scores catalog relation coverage, in-stock alternatives, complements, non-self recommendations, and evidence consistency.
- `generate_experiment_report.py` combines benchmark artifacts into paper-ready Markdown and JSON result tables.

Implemented evaluation paths:

1. Text evaluator for seed, ASR-proxy, and generated paper tasks.
2. Live trace recorder in the browser app.
3. Trace persistence through `/api/traces`.
4. Deterministic replay of saved tool calls through `benchmarks/replay_traces.py`.
5. Trust scoring through `benchmarks/evaluate_traces.py`.
6. Adversarial trace generation through `benchmarks/generate_adversarial_traces.py`.
7. Multi-tool text tasks where both `lookup_inventory` and `search_knowledge` must succeed.
8. Transcript robustness scoring through `benchmarks/evaluate_asr_transcripts.py`.
9. Catalog intelligence scoring through `benchmarks/evaluate_catalog_intelligence.py`.
10. Paper-table report generation through `benchmarks/generate_experiment_report.py`.

The seed tasks are intentionally tiny; the generated suite now has 1,223 text/interaction tasks, including 14 multi-tool product-plus-policy tasks and 189 reference/transcript pairs. The catalog relation evaluator currently covers 127 enriched products. The paper dataset should still expand to 1,000+ products and broader recorded audio turns before serious external experiments.

## Live Paper-Grade Suite

The browser Benchmark Lab and FastAPI endpoint use `benchmarks/eval_cases.json` as the curated live suite. Run it from the UI or with:

```powershell
$body = @{ limit = 16; includePayloads = $false; save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/suite/run -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/benchmarks/latest.json`
- `artifacts/benchmarks/latest.csv`

The suite reports task success, slot coverage, retrieval source match, evidence-gate decision, faithfulness score, runtime latency, estimated voice latency, and estimated cost.
## Real Speech Robustness Harness

The browser Speech Robustness Lab and FastAPI endpoint use `benchmarks/speech_cases.json` as a transcript-proxy speech suite. It measures ASR-style WER, entity-WER, entity recall, downstream inventory/RAG success, evidence-gate correctness, estimated voice latency, and estimated cost.

Run it from the UI or with:

```powershell
$body = @{ limit = 13; save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/speech/run -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/benchmarks/speech_latest.json`
- `artifacts/benchmarks/speech_latest.csv`

The current cases are transcript proxies rather than recorded audio. They are designed to exercise accent spellings, store noise substitutions, category questions, unsupported requests, prompt-injection speech, and a barge-in follow-up while keeping the harness runnable without external ASR calls.
## Live Paper Results Pack

The browser Paper Export panel and FastAPI endpoint combine the latest paper-grade task suite and speech robustness suite into paper-ready tables.

Run it from the UI or with:

```powershell
$body = @{ rerunSuites = $false; save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/report/run -Method Post -ContentType 'application/json' -Body $body
```

Set `rerunSuites = $true` to regenerate benchmark and speech results before exporting.

Saved artifacts:

- `artifacts/paper/latest_report.json`
- `artifacts/paper/latest_report.md`
- `artifacts/paper/core_metrics.csv`
- `artifacts/paper/cost_comparison.csv`
- `artifacts/paper/statistics_latest.json`
- `artifacts/paper/statistics_latest.csv`

The report includes executive metrics, statistical confidence intervals, condition breakdowns, failure analysis, cost comparison against GPT Realtime and Gemini Live baselines, and explicit limitations for publication.

## Real-Audio Error Taxonomy

The Real Audio Eval panel can turn the latest Deepgram audio run, retake queue, and accepted-set state into a paper-ready error analysis. It separates actual task failures from non-blocking latency notes, and it labels language mismatch separately from accent robustness.

The real-audio evaluator now also records language-aware fields: `detectedLanguage`, `canonicalTranscriptText`, `surfaceWer`, `canonicalWer`, and `multilingualScoring`. Spanish shopping utterances are canonicalized into English tool queries before inventory/RAG scoring, while literal Spanish-vs-English WER remains available as a diagnostic.

It also records `semanticTranscript`, a task-aware transcript score over intent, required shopping slots, canonical-query similarity, downstream recovery, and entity preservation. This keeps strict WER intact while allowing paper tables to separate true ASR failures from intent-equivalent paraphrases.

Run it from the UI with **Analyze errors** or with:

```powershell
$body = @{ save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/audio/error-analysis -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/audio_eval/error_analysis_latest.json`
- `artifacts/audio_eval/error_analysis_latest.csv`
- `artifacts/audio_eval/error_action_plan_latest.csv`

Use the action-plan CSV as the paper's error-analysis table: it groups failures by root cause, affected recordings, recommended system/data fix, and paper treatment.

## Paper Statistics Pack

The Paper Export panel can generate Wilson confidence intervals for pass rates and deterministic bootstrap intervals for WER, entity recall, latency, cost, savings, and audio robustness deltas.

Run it from the UI with **Run stats** or with:

```powershell
$body = @{ iterations = 1000; confidence = 0.95; save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/statistics/run -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/paper/statistics_latest.json`
- `artifacts/paper/statistics_latest.csv`

Rerun the task, speech, and real-audio suites before final paper export so the intervals reflect the latest experiment data.

## Paper Claim Readiness Gate

The Paper Export panel can convert statistics into paper-claim decisions: `publishable`, `needs_more_data`, `needs_system_work`, or `missing_evidence`. It checks claim thresholds against confidence intervals, minimum sample counts, dataset coverage, and robustness evidence, then writes a concrete action plan.

Run it from the UI with **Run claims** or with:

```powershell
$body = @{ regenerateStatistics = $false; save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/claims/run -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/paper/claims_latest.json`
- `artifacts/paper/claims_latest.csv`

The gate is intentionally stricter than the demo dashboard: a metric can look strong but still be marked `needs_more_data` if the lower/upper confidence bound or minimum sample size is not strong enough for a paper claim.

## Experiment Planner

The Paper Export panel can turn claim gaps into a deduplicated next-experiment plan. It groups work into text benchmark expansion, speech proxy expansion, real-audio collection, acoustic stress variants, and analysis reruns so the same new sample can satisfy multiple weak claims.

Run it from the UI with **Plan next** or with:

```powershell
$body = @{ refreshClaims = $false; save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/experiment-plan/run -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/paper/experiment_plan_latest.json`
- `artifacts/paper/experiment_plan_latest.csv`

The planner is a protocol artifact: it does not mutate benchmark files or record audio automatically. It tells you what to add next and keeps the sample budget reproducible for paper methods and ablation planning.

## Case Factory

The Paper Export panel can generate draft benchmark cases, speech proxy cases, and real-audio recording prompts from the latest experiment plan. The factory writes reviewable artifacts but does not automatically mutate `benchmarks/eval_cases.json` or `benchmarks/speech_cases.json`.

Run it from the UI with **Build cases** or with:

```powershell
$body = @{ refreshPlan = $false; save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/case-factory/run -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/paper/case_factory_latest.json`
- `artifacts/paper/case_factory_latest.csv`
- `artifacts/paper/generated_eval_cases.json`
- `artifacts/paper/generated_speech_cases.json`
- `artifacts/paper/generated_audio_recording_queue.json`

Review the draft expected fields before promotion, especially policy source IDs and recommendation `bestItemId` values. The generated audio queue is meant to be recorded through the Real Audio Eval panel.

## Draft Validation and Promotion Gate

The Paper Export panel can score generated drafts through the same benchmark and speech scorers used by the official suites. The gate writes a promotion manifest that separates ready cases from blocked drafts without mutating the official benchmark files.

Run it from the UI with **Validate drafts** or with:

```powershell
$body = @{ refreshFactory = $false; limit = $null; includePayloads = $false; save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/draft-validation/run -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/paper/draft_validation_latest.json`
- `artifacts/paper/draft_validation_latest.csv`
- `artifacts/paper/promotion_manifest_latest.json`
- `artifacts/paper/promotion_manifest_latest.csv`

Only promote draft cases after reviewing `promotion_manifest_latest.json`; audio prompts should still be recorded through the Real Audio Eval panel.

## Validated Suite Promotion

The Paper Export panel can preview or write validated draft cases into the official benchmark suites. The default mode is a dry run: it reports how many benchmark, speech, and audio recording prompts would be promoted without changing suite files. If `dryRun` is set to `false`, the backend backs up the existing suite JSON before writing promoted cases.

Preview from the UI with **Preview promote** or with:

```powershell
$body = @{
  dryRun = $true
  replaceFactoryCases = $true
  includeBenchmark = $true
  includeSpeech = $true
  includeAudioQueue = $true
  refreshValidation = $false
  save = $true
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/promotion/run -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/paper/suite_promotion_latest.json`
- `artifacts/paper/suite_promotion_latest.csv`
- `artifacts/audio_eval/promoted_recording_queue.json` when write mode is used
- `artifacts/paper/suite_backups/` when official suite files are written

Use write mode only after the dry run shows zero skipped cases.

## Real Audio Evaluation Harness

The browser Real Audio Eval panel records short audio fixtures against `benchmarks/audio_cases.json`, stores them under `artifacts/audio_eval/recordings`, and evaluates them through Deepgram prerecorded STT before running the transcript through the inventory/RAG task scorer.

Backend environment required for provider transcription:

```powershell
DEEPGRAM_API_KEY=your_deepgram_key
# optional
DEEPGRAM_API_URL=https://api.deepgram.com/v1/listen
DEEPGRAM_MODEL=nova-3
```

Run it from the UI or with:

```powershell
$body = @{ caseIds = @(); allowReferenceFallback = $false; save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/audio/run -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/audio_eval/audio_cases.local.json`
- `artifacts/audio_eval/latest.json`
- `artifacts/audio_eval/latest.csv`
- `artifacts/audio_eval/audio_dataset_manifest.json`
- `artifacts/audio_eval/audio_dataset_manifest.csv`
- `artifacts/audio_eval/recordings/*`

The `allowReferenceFallback` flag is useful only for local wiring checks. Keep it `false` for paper-quality real audio metrics.
Build the dataset manifest with:

```powershell
$body = @{ targetPerPrompt = 3; save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/audio/manifest -Method Post -ContentType 'application/json' -Body $body
```

The manifest reports prompt coverage, condition coverage, speaker/device/noise diversity, and joins each recording to the latest Deepgram evaluation result when available.

### Real Audio QA and Retake Queue

After running the real audio suite, build the audio QA gate to classify failures and create a retake queue. It labels empty transcripts, wrong-prompt or unintelligible clips, high WER, entity misses, downstream failures, low confidence, and tail latency. Use this queue to rerecord weak fixtures before making publishable real-audio claims.

Run it from the UI with **Build retake queue** or from PowerShell:

```powershell
$body = @{ save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/audio/quality -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/audio_eval/quality_latest.json`
- `artifacts/audio_eval/quality_latest.csv`
- `artifacts/audio_eval/retake_queue_latest.json`
- `artifacts/audio_eval/retake_queue_latest.csv`

### Accepted Audio Benchmark Set

After retakes, build the accepted-set artifact to separate the raw recording archive from publishable fixtures. The builder groups recordings by prompt, accent, noise, barge-in state, and augmentation type; accepts the latest passing recording in each group; labels older attempts as superseded; and leaves failed newer retakes visible as rejected retakes.

Run it from the UI with **Build accepted set** or from PowerShell:

```powershell
$body = @{ save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/audio/accepted-set -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/audio_eval/accepted_set_latest.json`
- `artifacts/audio_eval/accepted_set_latest.csv`

Use accepted-set metrics for headline paper claims, and raw archive metrics for limitations and retake-improvement analysis.

### Accent-Aware Deepgram Sweep

The Real Audio Eval panel can compare Deepgram transcription profiles on the same saved recordings with **Accent sweep**. The sweep benchmarks default Nova-3, `en-US` plus retail keyterms, accent-aware language hints plus retail keyterms, multilingual keyterms, and accent-aware keyterms plus a transparent retail transcript-repair layer.

Run it from the UI or from PowerShell:

```powershell
$body = @{
  caseIds = @()
  limit = 8
  includePassed = $false
  configs = @()
  allowReferenceFallback = $false
  save = $true
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/audio/accent-sweep -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/audio_eval/accent_sweep_latest.json`
- `artifacts/audio_eval/accent_sweep_latest.csv`

Report raw ASR WER/entity recall separately from repaired transcript success. The sweep is meant for ablation evidence, not for hiding provider errors.

### Audio Stress Variants

The Real Audio Eval panel can now generate controlled stress variants from a saved browser recording. Select a prompt with at least one saved recording, click **Generate stress variants**, then run **Run audio suite** to score the original plus generated WAV variants through Deepgram.

Generated variants are saved as normal recordings with `parentRecordingId`, `variantOf`, and `augmentation` metadata. The current browser generator creates synthetic store noise, low-volume speech, fast speech, slow speech, and mild clipping variants. These are useful for paper-level robustness tables because the JSON/CSV artifacts can group failures by acoustic condition and augmentation type without spending provider money until you run the audio suite.

### Audio Robustness Degradation Analysis

After running the real audio suite, run robustness analysis from the UI with **Analyze robustness** or from PowerShell:

```powershell
$body = @{ save = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/evaluation/audio/robustness -Method Post -ContentType 'application/json' -Body $body
```

Saved artifacts:

- `artifacts/audio_eval/robustness_latest.json`
- `artifacts/audio_eval/robustness_latest.csv`

The analyzer compares each generated variant with its parent original recording and reports pass-rate drops, WER deltas, entity-recall deltas, ASR latency deltas, cost deltas, regression verdicts, and aggregate breakdowns by augmentation type. It does not call Deepgram again; it only derives paper tables from the latest audio evaluation run.
