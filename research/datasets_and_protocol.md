# Datasets And Evaluation Protocol

## Dataset Strategy

The current 127-product enriched synthetic catalog is enough for a stronger prototype, but the paper should still grow toward a larger, externally auditable benchmark. The paper needs four layers of data:

1. Retail/product grounding data.
2. Tool-calling/task-decision data.
3. Speech/accent data.
4. Voice-agent interaction data with latency, barge-in, and noise.

## What The Relevant Papers Used

### From Text to Voice

Used audio-converted versions of CONFETTI and When2Call. The key idea is important for us: take a verified text/tool benchmark, convert user turns to audio with speaker variation and environmental noise, and preserve the original labels.

Use in our project:

- Convert retail inventory queries and tau-style tasks into audio.
- Keep gold labels: item ID, aisle, bay, stock, should-call/should-not-call, answer type.
- Add speaker/accent/noise variants without changing the ground truth.

### CONFETTI

CONFETTI has 109 human-simulated conversations, 313 user turns, and 86 APIs. It targets follow-ups, goal correction, implicit goals, ambiguity, and chained calls.

Use in our project:

- Borrow the evaluation style: turn-level scoring, dialog-act annotations, and follow-up handling.
- Do not use it as the retail dataset directly unless its license/data access is confirmed.

### When2Call

When2Call evaluates when to call a tool, when to ask a follow-up, and when to admit the tool cannot answer. Its released data includes labels such as `direct`, `tool_call`, `request_for_info`, and `cannot_answer`.

Use in our project:

- Create a retail-specific When2Call subset:
  - tool_call: "Where are paper towels?"
  - request_for_info: "Do you have that cereal?" with missing referent.
  - cannot_answer: "Do you sell PS5 controllers?" when the specific product is absent from the catalog.
  - direct: "Thanks" or "What can you help with?"

### tau / tau2 / tau3 Bench

tau-bench evaluates tool-agent-user interaction in real-world domains and compares final database state to an annotated goal. tau3 adds voice full-duplex evaluation, realistic accents/noise, and domains including retail.

Use in our project:

- Use tau3 retail as a strong external benchmark if installation and license are acceptable.
- Use its pass@k and state-change evaluation style.
- For our narrower store-associate task, define a simpler but fully verifiable state: correct item, aisle, stock, and follow-up behavior.

### FLEURS

FLEURS covers 102 languages with about 12 hours per language and supports ASR, language ID, translation, and retrieval.

Use in our project:

- Evaluate STT robustness across English, Indian English if available through selected subsets, and multilingual retail queries.
- Use as ASR/accent/language stress data, not as inventory task data.

### AESRC 2020

AESRC provides 160 hours of accented English training data from 8 countries and a 20-hour test set including unseen accents.

Use in our project:

- Accent robustness benchmark for STT.
- If data access is practical, map utterances into our ASR evaluation.
- Use WER/CER by accent, then correlate with task success.

### Common Voice

Common Voice is broad, public, and includes demographic/accent metadata in many language releases. It is useful but noisy.

Use in our project:

- Create an accent-balanced real-speech probe set.
- Prefer validated clips.
- Use license metadata carefully.

### Open Food Facts / Open Products / Open Pet Food Facts

Open Food Facts and sibling projects provide open product metadata. Open Pet Food Facts is useful for categories like dog food.

Use in our project:

- Expand the inventory from 50 rows to 1,000+ realistic products.
- Generate synthetic aisle/bay/stock fields while preserving real product names/categories.
- License note: Open Food Facts is ODbL, so derived public databases likely need attribution and share-alike handling.

## Proposed VoiceRetailBench Data Layers

### Layer A - Product Catalog

Target size: 1,000 to 5,000 products.

Fields:

- sku
- product_name
- brand
- category
- department
- aisle
- bay
- stock
- price
- synonyms
- source
- license

Sources:

- Open Food Facts for groceries.
- Open Pet Food Facts for pet products.
- Synthetic non-food household/health/baby products.

### Layer B - Text Task Set

Target size: 1,500 to 3,000 user turns.

Task classes:

- exact lookup
- category lookup
- stock check
- aisle-only question
- out-of-catalog question
- ambiguous reference
- follow-up
- correction
- list/category inventory
- policy question
- no-tool conversational turn

Labels:

- expected_action: direct | tool_call | request_info | cannot_answer
- expected_tool
- expected_query
- expected_item_ids
- expected_aisles
- required_slots
- allowed_response_patterns

### Layer C - Audio Task Set

For each text task, generate or collect audio variants:

- clean TTS voice
- TTS with accent/speaker variation
- noisy environment: 20 dB / 10 dB SNR
- real accented speech subset where available
- interruption/barge-in scenario subset

### Layer D - Interaction Tasks

Multi-turn scenarios:

- "Where are paper towels?" -> "What about the cheaper one?"
- "Do you have dog food?" -> "List what you have."
- "I said diapers, not wipes."
- "Do you have PS5 controllers?" -> agent should not invent.
- User interrupts assistant mid-answer and changes item.

## Minimum Publishable Dataset

If time is tight, release this:

- 1,000 product inventory.
- 500 text turns.
- 500 clean audio turns.
- 200 accent/noise audio turns.
- 100 multi-turn conversations.
- 50 barge-in scenarios.

This is already much stronger than the current demo.

## Full Dataset Goal

- 5,000 product inventory.
- 3,000 text turns.
- 12,000 audio variants.
- 500 multi-turn conversations.
- 200 barge-in conversations.

## Data We Should Not Use

- Scraped private retailer pages unless terms allow it.
- Proprietary product/inventory data without permission.
- Any speech dataset whose license blocks redistribution or derived benchmark release.
