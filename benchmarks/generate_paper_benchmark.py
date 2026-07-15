from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.inventory import load_inventory  # noqa: E402
from backend.app.knowledge import load_knowledge_chunks  # noqa: E402


RANDOM_SEED = 20260701
DEFAULT_OUTPUT = Path(__file__).with_name("generated") / "voice_retail_paper_tasks.jsonl"

BASE_CONDITION = {
    "accent": "reference_us",
    "language": "en-US",
    "noise": "clean",
    "snr_db": None,
    "barge_in": False,
}

ASR_PROXY_CONDITIONS = [
    {"accent": "indian_english_proxy", "language": "en-US", "noise": "clean", "snr_db": None, "barge_in": False},
    {"accent": "spanish_l1_proxy", "language": "en-US", "noise": "clean", "snr_db": None, "barge_in": False},
    {"accent": "fast_speech_proxy", "language": "en-US", "noise": "store_ambient_proxy", "snr_db": 10, "barge_in": False},
]

NOISE_CONDITIONS = [
    {"accent": "reference_us", "language": "en-US", "noise": "store_ambient_proxy", "snr_db": 20, "barge_in": False},
    {"accent": "reference_us", "language": "en-US", "noise": "checkout_beeps_proxy", "snr_db": 10, "barge_in": False},
    {"accent": "reference_us", "language": "en-US", "noise": "freezer_hum_proxy", "snr_db": 5, "barge_in": False},
]

INVENTORY_TEMPLATES = [
    "Where is {alias}?",
    "Do you have {alias}?",
    "Can you check stock for {alias}?",
    "What aisle has {alias}?",
]

LIST_TEMPLATES = [
    "What {alias} options do you have?",
    "Can you list the {alias} inventory?",
    "Show me the {alias} items in stock.",
]

ASR_SUBSTITUTIONS = {
    "aluminum": "a loom in um",
    "bounty": "boun tee",
    "cetirizine": "set iris in",
    "clorox": "chlor ox",
    "coca-cola": "coca cola",
    "digiorno": "de giorno",
    "honeycrisp": "honey crisp",
    "huggies": "hug ease",
    "kellogg's": "kellogs",
    "mccormick": "mc cormick",
    "oral-b": "oral b",
    "reynolds": "renolds",
    "sargento": "sar jento",
    "smucker's": "smuckers",
    "tylenol": "tie len all",
    "yoplait": "yo play",
    "ziploc": "zip lock",
}

UNSUPPORTED_INVENTORY_QUERIES = [
    "PS5 controller",
    "Nintendo Switch OLED",
    "fishing license",
    "car battery installation",
    "men's winter coat",
    "garden hose reel",
    "printer ink for Epson 802",
    "propane tank exchange",
    "live bait worms",
    "passport photo service",
]

AMBIGUOUS_QUERIES = [
    "Do you have that cereal?",
    "Where is the thing from the ad?",
    "Can you find the one I bought last time?",
    "Do you have it in stock?",
    "Where is that medicine?",
]

RAG_QUERY_BANK = {
    "returns_and_exchanges#1": [
        "standard return window with receipt",
        "how many days for unopened merchandise return",
        "where should a customer go for a return",
    ],
    "returns_and_exchanges#2": [
        "opened personal care item return",
        "used item return eligibility",
        "damaged product refund review",
    ],
    "returns_and_exchanges#3": [
        "opened electronics return",
        "gaming controller return policy",
        "phone or tablet return service desk",
    ],
    "online_pickup#1": [
        "online pickup order ready location",
        "curbside pickup where customer should go",
        "customer arrived for grocery pickup",
    ],
    "online_pickup#2": [
        "app says pickup order ready but missing",
        "online order not ready what should associate do",
        "pickup support when order cannot be found",
    ],
    "online_pickup#3": [
        "grocery pickup substitutions",
        "customer wants to reject substitution",
        "swap grocery pickup item",
    ],
    "price_match#1": [
        "shelf tag and register price disagree",
        "wrong price at checkout",
        "price check tag date register lead",
    ],
    "price_match#2": [
        "competitor price match",
        "can store match other retailer price",
        "customer asks for competitor promotion",
    ],
    "price_match#3": [
        "app price different from shelf",
        "online app shows lower price",
        "price differs by delivery pickup local store",
    ],
    "out_of_stock#1": [
        "shelf empty but system shows stock",
        "item not on shelf check topstock",
        "where to check when inventory shows units",
    ],
    "out_of_stock#2": [
        "zero units available restock detail",
        "item out of stock truck note",
        "can associate promise arrival time",
    ],
    "out_of_stock#3": [
        "similar item for category request",
        "offer closest matching item or aisle",
        "do not invent alternative product",
    ],
    "accessibility_services#1": [
        "wheelchair cart near entrance",
        "mobility cart availability",
        "front end help if no mobility scooter",
    ],
    "accessibility_services#2": [
        "customer needs language help",
        "Spanish support at service desk",
        "manager for language assistance",
    ],
    "accessibility_services#3": [
        "large item carryout help",
        "heavy bulky item assistance",
        "front end team carryout",
    ],
    "associate_sop#1": [
        "associate should greet and answer briefly",
        "store associate response style",
        "give aisle bay and stock count",
    ],
    "associate_sop#2": [
        "when to use inventory tool",
        "when to use knowledge search tool",
        "choose inventory or policy lookup",
        "what should associate say before tool lookup",
    ],
    "associate_sop#3": [
        "knowledge base does not cover answer",
        "do not invent policy details",
        "do not promise restock dates",
    ],
}

UNSUPPORTED_RAG_QUERIES = [
    "fishing license permit",
    "money transfer hours",
    "auto center tire warranty",
    "pharmacy vaccine appointment",
    "vision center insurance plan",
    "firearm background check",
    "photo printing copyright rule",
    "check cashing fee",
]


def _transcript_variant(text: str, condition: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "id": "asr_proxy_transcript",
        "provider": "synthetic-asr-proxy",
        "text": text,
        "condition": condition,
        "notes": notes,
    }


def _turn(
    text: str,
    role: str = "user",
    *,
    interrupt: int | None = None,
    reference_text: str | None = None,
    transcript_variants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    turn = {"role": role, "text": text, "audio_uri": None, "interrupts_turn_index": interrupt}
    if reference_text is not None:
        turn["reference_text"] = reference_text
    if transcript_variants:
        turn["transcript_variants"] = transcript_variants
    return turn


def _source(dataset: str, notes: str) -> dict[str, str]:
    return {"dataset": dataset, "license": "synthetic", "notes": notes}


def _task(
    task_id: str,
    split: str,
    modality: str,
    condition: dict[str, Any],
    turns: list[dict[str, Any]],
    expected: dict[str, Any],
    dataset: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "split": split,
        "modality": modality,
        "condition": condition,
        "turns": turns,
        "expected": expected,
        "source": _source(dataset, notes),
    }


def _inventory_expected(query: str, sku: str, aisle: str) -> dict[str, Any]:
    return {
        "action": "tool_call",
        "tool_name": "lookup_inventory",
        "query": query,
        "item_ids": [sku],
        "source_ids": [],
        "aisles": [aisle],
        "required_slots": ["item", "aisle", "bay", "stock"],
    }


def _rag_expected(query: str, sources: list[str]) -> dict[str, Any]:
    return {
        "action": "tool_call",
        "tool_name": "search_knowledge",
        "query": query,
        "item_ids": [],
        "source_ids": sources,
        "aisles": [],
        "required_slots": ["knowledge_source", "policy_summary"],
    }


def _cannot_answer_expected(tool_name: str, query: str) -> dict[str, Any]:
    return {
        "action": "cannot_answer",
        "tool_name": tool_name,
        "query": query,
        "item_ids": [],
        "source_ids": [],
        "aisles": [],
        "required_slots": ["cannot_answer"],
    }


def _multi_tool_expected(
    *,
    inventory_query: str,
    knowledge_query: str,
    sku: str,
    aisle: str,
    sources: list[str],
) -> dict[str, Any]:
    return {
        "action": "multi_tool",
        "tool_name": None,
        "query": f"{inventory_query} | {knowledge_query}",
        "tool_calls": [
            {
                "tool_name": "lookup_inventory",
                "query": inventory_query,
                "item_ids": [sku],
                "source_ids": [],
                "aisles": [aisle],
                "required_slots": ["item", "aisle", "bay", "stock"],
            },
            {
                "tool_name": "search_knowledge",
                "query": knowledge_query,
                "item_ids": [],
                "source_ids": sources,
                "aisles": [],
                "required_slots": ["knowledge_source", "policy_summary"],
            },
        ],
        "item_ids": [sku],
        "source_ids": sources,
        "aisles": [aisle],
        "required_slots": ["item", "aisle", "bay", "stock", "knowledge_source", "policy_summary"],
    }


def _asr_proxy(value: str) -> str:
    normalized = value
    for before, after in ASR_SUBSTITUTIONS.items():
        normalized = normalized.replace(before, after)
        normalized = normalized.replace(before.title(), after)
    return normalized


def _item_aliases(item: dict[str, Any]) -> list[str]:
    aliases = [item["name"], *item.get("synonyms", [])]
    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        key = alias.lower()
        if key not in seen:
            seen.add(key)
            result.append(alias)
    return result[:4]


def inventory_tasks(rng: random.Random) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    items = load_inventory()
    for index, item in enumerate(items, start=1):
        aliases = _item_aliases(item)
        for template_index, template in enumerate(INVENTORY_TEMPLATES, start=1):
            alias = aliases[(template_index - 1) % len(aliases)]
            user_text = template.format(alias=alias)
            task_id = f"paper_inv_exact_{index:03d}_{template_index:02d}"
            tasks.append(
                _task(
                    task_id,
                    "test",
                    "text",
                    BASE_CONDITION,
                    [_turn(user_text)],
                    _inventory_expected(alias, item["sku"], item["aisle"]),
                    "voice-retailbench-inventory-clean",
                    "Clean inventory lookup with exact and synonym product mentions.",
                )
            )

        list_alias = aliases[-1]
        for template_index, template in enumerate(LIST_TEMPLATES, start=1):
            user_text = template.format(alias=list_alias)
            task_id = f"paper_inv_list_{index:03d}_{template_index:02d}"
            tasks.append(
                _task(
                    task_id,
                    "test",
                    "text",
                    BASE_CONDITION,
                    [_turn(user_text)],
                    _inventory_expected(list_alias, item["sku"], item["aisle"]),
                    "voice-retailbench-inventory-category",
                    "Category/list wording for a known catalog item.",
                )
            )

        asr_alias = _asr_proxy(aliases[0])
        condition = rng.choice(ASR_PROXY_CONDITIONS)
        reference_text = f"Can you check {aliases[0]} for me?"
        transcript_text = f"Can you check {asr_alias} for me?"
        task_id = f"paper_inv_asr_{index:03d}"
        tasks.append(
            _task(
                task_id,
                "stress",
                "text",
                condition,
                [
                    _turn(
                        transcript_text,
                        reference_text=reference_text,
                        transcript_variants=[
                            _transcript_variant(
                                transcript_text,
                                condition,
                                "Product-name ASR proxy generated from the clean reference text.",
                            )
                        ],
                    )
                ],
                _inventory_expected(asr_alias, item["sku"], item["aisle"]),
                "voice-retailbench-inventory-asr-proxy",
                "ASR-proxy spelling/segmentation perturbation for product names.",
            )
        )

    for index, query in enumerate(UNSUPPORTED_INVENTORY_QUERIES, start=1):
        tasks.append(
            _task(
                f"paper_inv_unsupported_{index:03d}",
                "test",
                "text",
                BASE_CONDITION,
                [_turn(f"Do you have {query}?")],
                _cannot_answer_expected("lookup_inventory", query),
                "voice-retailbench-inventory-unsupported",
                "Out-of-catalog item should not be hallucinated.",
            )
        )

    for index, query in enumerate(AMBIGUOUS_QUERIES, start=1):
        tasks.append(
            _task(
                f"paper_inv_clarify_{index:03d}",
                "test",
                "text",
                BASE_CONDITION,
                [_turn(query)],
                {
                    "action": "request_info",
                    "tool_name": None,
                    "query": None,
                    "item_ids": [],
                    "source_ids": [],
                    "aisles": [],
                    "required_slots": ["clarification"],
                },
                "voice-retailbench-clarification",
                "Underspecified user turn should trigger clarification.",
            )
        )

    sampled_items = rng.sample(items, k=min(30, len(items)))
    for index, item in enumerate(sampled_items, start=1):
        alias = _item_aliases(item)[-1]
        new_item = rng.choice(items)
        new_alias = _item_aliases(new_item)[-1]
        tasks.append(
            _task(
                f"paper_inv_followup_{index:03d}",
                "test",
                "interaction",
                BASE_CONDITION,
                [
                    _turn(f"Do you have {alias}?"),
                    _turn(
                        f"{item['name']} is on aisle {item['aisle']}, bay {item['bay']}.",
                        role="assistant",
                    ),
                    _turn("Can you list what you have in that category?"),
                ],
                _inventory_expected(alias, item["sku"], item["aisle"]),
                "voice-retailbench-multiturn",
                "Follow-up should preserve the previous product/category referent.",
            )
        )
        tasks.append(
            _task(
                f"paper_inv_barge_proxy_{index:03d}",
                "stress",
                "interaction",
                {"accent": "reference_us", "language": "en-US", "noise": "clean", "snr_db": None, "barge_in": True},
                [
                    _turn(f"Where is {alias}?"),
                    _turn(
                        "Let me check that item and give you the aisle, bay, and current stock count.",
                        role="assistant",
                    ),
                    _turn(f"Actually, where is {new_alias}?", interrupt=1),
                ],
                _inventory_expected(new_alias, new_item["sku"], new_item["aisle"]),
                "voice-retailbench-barge-in-proxy",
                "Text proxy for interruption: the final user intent should replace the interrupted one.",
            )
        )

    return tasks


def rag_tasks(rng: random.Random) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    known_sources = {chunk.source for chunk in load_knowledge_chunks()}
    query_items = [(source, queries) for source, queries in RAG_QUERY_BANK.items() if source in known_sources]

    for source_index, (source, queries) in enumerate(query_items, start=1):
        for query_index, query in enumerate(queries, start=1):
            tasks.append(
                _task(
                    f"paper_rag_clean_{source_index:03d}_{query_index:02d}",
                    "test",
                    "text",
                    BASE_CONDITION,
                    [_turn(query)],
                    _rag_expected(query, [source]),
                    "voice-retailbench-rag-clean",
                    "Clean knowledge-base retrieval with source-level labels.",
                )
            )

            stress_condition = rng.choice(ASR_PROXY_CONDITIONS + NOISE_CONDITIONS)
            stress_query = _asr_proxy(query)
            tasks.append(
                _task(
                    f"paper_rag_stress_{source_index:03d}_{query_index:02d}",
                    "stress",
                    "text",
                    stress_condition,
                    [
                        _turn(
                            stress_query,
                            reference_text=query,
                            transcript_variants=[
                                _transcript_variant(
                                    stress_query,
                                    stress_condition,
                                    "Knowledge-query ASR/noise proxy generated from the clean reference text.",
                                )
                            ],
                        )
                    ],
                    _rag_expected(stress_query, [source]),
                    "voice-retailbench-rag-asr-noise-proxy",
                    "ASR/noise proxy for knowledge retrieval source accuracy.",
                )
            )

    for index, query in enumerate(UNSUPPORTED_RAG_QUERIES, start=1):
        condition = BASE_CONDITION if index % 2 else rng.choice(NOISE_CONDITIONS)
        tasks.append(
            _task(
                f"paper_rag_unsupported_{index:03d}",
                "test",
                "text",
                condition,
                [_turn(query)],
                _cannot_answer_expected("search_knowledge", query),
                "voice-retailbench-rag-unsupported",
                "Unsupported policy/service request should abstain.",
            )
        )

    return tasks


def multi_tool_tasks(rng: random.Random) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    items_by_sku = {item["sku"]: item for item in load_inventory()}
    cases = [
        {
            "slug": "shelf_empty",
            "sku": "SHOP-1001",
            "product_query": "paper towels",
            "knowledge_query": "shelf empty but system shows stock",
            "sources": ["out_of_stock#1"],
            "text": "The shelf is empty for paper towels, but your system says you have it. Where should I check?",
            "notes": "Requires inventory stock/aisle plus out-of-stock shelf-check SOP.",
        },
        {
            "slug": "zero_stock",
            "sku": "SHOP-1036",
            "product_query": "ground beef",
            "knowledge_query": "zero units available restock detail",
            "sources": ["out_of_stock#2"],
            "text": "Do you have ground beef, and if it is out, can you promise when it comes back?",
            "notes": "Requires zero-stock inventory plus no-restock-promise policy.",
        },
        {
            "slug": "category_substitute",
            "sku": "SHOP-1050",
            "product_query": "dog food",
            "knowledge_query": "similar item for category request",
            "sources": ["out_of_stock#3"],
            "text": "What dog food do you have, and if that one is gone what similar option can you suggest?",
            "notes": "Requires category inventory plus grounded substitute guidance.",
        },
        {
            "slug": "opened_return",
            "sku": "SHOP-1048",
            "product_query": "Tylenol Extra Strength Caplets 100 Count",
            "knowledge_query": "opened personal care item return",
            "sources": ["returns_and_exchanges#2"],
            "text": "Where is Tylenol, and can I return it if I opened it?",
            "notes": "Requires product aisle plus opened consumable return policy.",
        },
        {
            "slug": "shelf_price",
            "sku": "SHOP-1001",
            "product_query": "Bounty paper towels",
            "knowledge_query": "shelf tag and register price disagree",
            "sources": ["price_match#1"],
            "text": "Where are Bounty paper towels, and what happens if the shelf tag and register price disagree?",
            "notes": "Requires aisle lookup plus price-adjustment routing policy.",
        },
        {
            "slug": "app_price",
            "sku": "SHOP-1011",
            "product_query": "milk",
            "knowledge_query": "app price different from shelf",
            "sources": ["price_match#3"],
            "text": "Is milk available, and why might the app price be different from the shelf price?",
            "notes": "Requires inventory lookup plus app-price evidence.",
        },
        {
            "slug": "pickup_substitution",
            "sku": "SHOP-1015",
            "product_query": "yogurt",
            "knowledge_query": "grocery pickup substitutions",
            "sources": ["online_pickup#3"],
            "text": "Do you have yogurt, and can a pickup customer reject a substitution?",
            "notes": "Requires inventory lookup plus pickup substitution policy.",
        },
    ]

    for index, case in enumerate(cases, start=1):
        item = items_by_sku[case["sku"]]
        tasks.append(
            _task(
                f"paper_multi_clean_{index:03d}_{case['slug']}",
                "test",
                "text",
                BASE_CONDITION,
                [_turn(case["text"])],
                _multi_tool_expected(
                    inventory_query=case["product_query"],
                    knowledge_query=case["knowledge_query"],
                    sku=item["sku"],
                    aisle=item["aisle"],
                    sources=case["sources"],
                ),
                "voice-retailbench-multi-tool-clean",
                case["notes"],
            )
        )

        stress_condition = rng.choice(ASR_PROXY_CONDITIONS + NOISE_CONDITIONS)
        stress_product_query = _asr_proxy(item["name"])
        stress_knowledge_query = _asr_proxy(case["knowledge_query"])
        stress_text = _asr_proxy(case["text"])
        tasks.append(
            _task(
                f"paper_multi_stress_{index:03d}_{case['slug']}",
                "stress",
                "text",
                stress_condition,
                [
                    _turn(
                        stress_text,
                        reference_text=case["text"],
                        transcript_variants=[
                            _transcript_variant(
                                stress_text,
                                stress_condition,
                                "Multi-tool ASR/noise proxy generated from the clean reference text.",
                            )
                        ],
                    )
                ],
                _multi_tool_expected(
                    inventory_query=stress_product_query,
                    knowledge_query=stress_knowledge_query,
                    sku=item["sku"],
                    aisle=item["aisle"],
                    sources=case["sources"],
                ),
                "voice-retailbench-multi-tool-asr-noise-proxy",
                f"ASR/noise proxy. {case['notes']}",
            )
        )

    return tasks


def build_tasks() -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    tasks = inventory_tasks(rng) + rag_tasks(rng) + multi_tool_tasks(rng)
    tasks.sort(key=lambda task: task["task_id"])
    return tasks


def write_jsonl(tasks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, separators=(",", ":")) + "\n")


def write_manifest(tasks: list[dict[str, Any]], path: Path) -> None:
    by_dataset: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_modality: dict[str, int] = {}
    by_noise: dict[str, int] = {}
    by_accent: dict[str, int] = {}
    transcript_pair_count = 0
    for task in tasks:
        dataset = task.get("source", {}).get("dataset", "unknown")
        condition = task.get("condition") or {}
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
        by_split[task["split"]] = by_split.get(task["split"], 0) + 1
        by_modality[task["modality"]] = by_modality.get(task["modality"], 0) + 1
        by_noise[str(condition.get("noise"))] = by_noise.get(str(condition.get("noise")), 0) + 1
        by_accent[str(condition.get("accent"))] = by_accent.get(str(condition.get("accent")), 0) + 1
        for turn in task.get("turns", []):
            variants = turn.get("transcript_variants")
            if isinstance(variants, list):
                transcript_pair_count += len(variants)

    manifest = {
        "name": "VoiceRetailBench paper-scale synthetic suite",
        "version": "0.4.0",
        "seed": RANDOM_SEED,
        "task_count": len(tasks),
        "transcript_pair_count": transcript_pair_count,
        "by_dataset": by_dataset,
        "by_split": by_split,
        "by_modality": by_modality,
        "by_noise": by_noise,
        "by_accent": by_accent,
        "notes": [
            "Synthetic retail benchmark for speech-to-tool and speech-to-RAG evaluation.",
            "Includes multi-tool product-plus-policy tasks where both inventory and knowledge evidence are required.",
            "Includes reference_text and transcript_variants fields for WER/entity-WER transcript robustness evaluation.",
            "Audio fields are placeholders until recorded/TTS audio is generated.",
            "Stress conditions are text proxies for ASR, accent, noise, and interruption experiments.",
        ],
    }
    manifest_path = path.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the paper-scale VoiceRetailBench synthetic task suite.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tasks = build_tasks()
    write_jsonl(tasks, args.out)
    write_manifest(tasks, args.out)
    print(json.dumps({"tasks": len(tasks), "out": str(args.out), "manifest": str(args.out.with_suffix(".manifest.json"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
