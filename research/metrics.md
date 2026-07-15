# Metrics

## Primary Metrics

### Task Success

Binary success for each user goal:

- correct item or category
- correct aisle/bay when available
- correct stock status
- no hallucinated unavailable item
- correct follow-up behavior

Report:

- success rate
- pass@1
- pass@k for stochastic runs
- success under clean/accent/noise conditions

### Slot Accuracy

Measure each field separately:

- item_id accuracy
- aisle accuracy
- bay accuracy
- stock status accuracy
- stock count tolerance
- tool-decision accuracy

### Voice Tool-Call Accuracy

Break down:

- should call vs should not call
- correct function name
- correct query argument
- correct clarification decision
- correct cannot-answer decision

### RAG Source Accuracy

Measure whether retrieved sources include the expected passages:

- source recall@k
- source precision@k
- context relevance
- grounded answer faithfulness
- unsupported-answer rate

For voice RAG, report the same metrics by accent and noise bucket.

### Latency

Report P50/P90/P95:

- speech_end_to_stt_final_ms
- speech_end_to_tool_call_ms
- speech_end_to_first_llm_token_ms
- speech_end_to_first_audio_ms
- speech_end_to_answer_complete_ms
- barge_in_detection_ms
- barge_in_audio_stop_ms

The headline latency is:

> end-of-user-speech to first audible assistant audio.

### Cost

Report:

- cost_per_minute
- cost_per_turn
- cost_per_successful_task
- cost split by orchestrator / STT / LLM / TTS

Cost per successful task:

```text
cost_per_success = total_cost / successful_tasks
```

This is the metric that can make the paper stand out.

### Accent Robustness

For each accent/language bucket:

- WER
- entity-WER
- entity recall
- task success
- tool argument accuracy
- latency
- no-call/cannot-answer accuracy

Useful derived metric:

```text
accent_drop = clean_reference_success - accent_group_success
```

### Robustness Under Noise

Report by SNR:

- clean
- 20 dB
- 10 dB
- cafeteria/store noise

### Transcript Robustness

For each reference/transcript pair:

- WER between clean reference text and ASR/proxy transcript
- entity-WER over product, aisle, policy, and retrieval-query terms
- entity recall
- downstream task success
- grouped results by dataset, accent, noise, and SNR

### Barge-In Quality

Measure:

- interruption detection rate
- false interruption rate
- time to stop assistant audio
- ability to answer new user intent
- transcript contamination rate

## Error Taxonomy

Each failed turn should be assigned one primary error:

- ASR_ERROR: transcript wrong enough to change intent.
- ENDPOINT_ERROR: user cut off or endpoint too late.
- TOOL_DECISION_ERROR: wrong call / no call / unnecessary call.
- SLOT_ERROR: tool query argument wrong.
- TOOL_RESULT_ERROR: backend/search returned wrong product.
- RAG_RETRIEVAL_ERROR: knowledge search missed the required source.
- RAG_FAITHFULNESS_ERROR: retrieved source was correct but answer added unsupported policy.
- RESPONSE_GROUNDING_ERROR: tool result correct but answer wrong.
- TTS_OR_AUDIO_ERROR: answer text correct but user-facing speech failed.
- INTERRUPTION_ERROR: failed barge-in handling.

## Tables To Include In Paper

1. Architecture cost-latency table.
2. Component ablation table.
3. Accent/noise and transcript robustness table.
4. Tool-call decision confusion matrix.
5. RAG source-retrieval and faithfulness table.
6. Error taxonomy table.
7. Cost-per-success Pareto frontier.
