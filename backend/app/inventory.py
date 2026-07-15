from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from hashlib import blake2b
from pathlib import Path
from typing import Any

from .multilingual import canonicalize_query, localize_answer


DATA_PATH = Path(__file__).parent / "data" / "inventory.json"
MAX_TOOL_MATCHES = 10
MAX_SPOKEN_MATCHES = 5
MAX_RELATED_ITEMS = 4
GENERIC_RELATION_ALIASES = {
    "adult",
    "dry food",
    "food",
    "pet",
    "pet food",
    "pets",
    "standard",
    "wet food",
}
STOP_WORDS = {
    "a",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "available",
    "availability",
    "bay",
    "brand",
    "brands",
    "can",
    "carry",
    "carries",
    "could",
    "different",
    "do",
    "exact",
    "find",
    "for",
    "give",
    "had",
    "has",
    "have",
    "i",
    "in",
    "inventory",
    "is",
    "it",
    "item",
    "items",
    "know",
    "list",
    "listed",
    "listing",
    "looking",
    "may",
    "me",
    "need",
    "not",
    "of",
    "on",
    "please",
    "product",
    "products",
    "provide",
    "show",
    "some",
    "specific",
    "stock",
    "stocked",
    "sure",
    "tell",
    "the",
    "there",
    "to",
    "type",
    "types",
    "unit",
    "units",
    "well",
    "what",
    "whether",
    "where",
    "which",
    "you",
    "your",
    "best",
    "based",
    "better",
    "customer",
    "highest",
    "one",
    "option",
    "options",
    "overall",
    "popular",
    "rating",
    "ratings",
    "recommend",
    "recommended",
    "recommendation",
    "review",
    "reviews",
    "top",
}
LIST_INTENT_RE = re.compile(
    r"\b(all|any|carry|different|list|show|what.*have|what.*inventory|items?|options?|types?)\b",
    re.IGNORECASE,
)
BEST_INTENT_RE = re.compile(
    r"\b(best|better|top\s*rated|highest\s*rated|recommend|recommended|recommendation|popular|reviews?|rating|which\s+one|which\s+is\s+best)\b",
    re.IGNORECASE,
)
NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
BROAD_SINGLE_TOKEN_ALIASES = {
    "ambient",
    "bag",
    "bottle",
    "box",
    "case",
    "count",
    "each",
    "frozen",
    "ft",
    "gallon",
    "gallons",
    "inch",
    "jug",
    "lb",
    "mm",
    "oz",
    "pack",
    "pair",
    "qt",
    "refrigerated",
    "roll",
    "rolls",
    "set",
    "sheets",
    "standard",
    "yd",
}
COMPLEMENT_QUERIES_BY_SUBCATEGORY = {
    "Baby Care": ["baby wipes", "baby lotion"],
    "Bakery": ["peanut butter", "strawberry jam"],
    "Batteries": ["flashlight", "surge protector"],
    "Breakfast": ["coffee", "milk"],
    "Cat Food": ["cat litter"],
    "Cleaning": ["trash bags", "paper towels"],
    "Cold and Flu": ["cough drops", "thermometer"],
    "Dairy": ["eggs", "butter"],
    "Dog Food": ["dog treats"],
    "Food Storage": ["aluminum foil", "plastic wrap"],
    "Hair Care": ["conditioner", "shampoo"],
    "Laundry": ["fabric softener", "laundry basket"],
    "Pain Relief": ["thermometer", "bandages"],
    "Pantry": ["pasta sauce", "olive oil"],
    "Pet Food": [],
    "Phone Accessories": ["usb c cable", "wall charger", "surge protector"],
    "Produce": ["salad", "spinach"],
    "Soup": ["crackers", "bread"],
}


@dataclass(frozen=True)
class InventoryMatch:
    item: dict[str, Any] | None
    score: float
    reason: str


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", value.lower()).strip()


def _tokens(value: str) -> set[str]:
    tokens = set()
    for token in _normalize(value).split():
        if not token or token in STOP_WORDS:
            continue
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _aliases(item: dict[str, Any]) -> list[str]:
    aliases = [
        item["name"],
        item["department"],
        item.get("brand"),
        item.get("category"),
        item.get("subcategory"),
        item.get("size"),
        item.get("unit"),
        item.get("availabilityStatus"),
    ]
    for key in ("synonyms", "attributes", "shelfTags", "queryHints"):
        values = item.get(key) or []
        aliases.extend(values if isinstance(values, list) else [values])
    return [str(alias) for alias in aliases if alias]


def _spoken_int(value: int) -> str:
    if value in NUMBER_WORDS:
        return NUMBER_WORDS[value]
    if 20 < value < 100:
        tens = (value // 10) * 10
        ones = value % 10
        return NUMBER_WORDS[tens] if ones == 0 else f"{NUMBER_WORDS[tens]} {NUMBER_WORDS[ones]}"
    return str(value)


def _spoken_digits(value: str) -> str:
    return " ".join(NUMBER_WORDS[int(digit)] for digit in value)


def _spoken_code(value: str) -> str:
    parts = re.findall(r"[A-Za-z]+|\d+", str(value))
    spoken: list[str] = []
    for part in parts:
        if part.isdigit():
            if part.startswith("0"):
                spoken.append(_spoken_digits(part))
            else:
                spoken.append(_spoken_int(int(part)))
        else:
            spoken.append(" ".join(part.upper()))
    return " ".join(spoken)


def _spoken_bay(value: str) -> str:
    parts = re.findall(r"[A-Za-z]+|\d+", str(value))
    spoken: list[str] = []
    for part in parts:
        if part.isdigit():
            spoken.append(_spoken_int(int(part)))
        else:
            spoken.append(part.lower())
    return " ".join(spoken)


def _spoken_name(value: str) -> str:
    spoken = value
    replacements = [
        (r"(\d+)/(\d+)", r"\1 \2"),
        (r"\bfl oz\b", "fluid ounce"),
        (r"\boz\b", "ounce"),
        (r"\blb\b", "pound"),
        (r"(\d+)%", r"\1 percent"),
    ]
    for pattern, replacement in replacements:
        spoken = re.sub(pattern, replacement, spoken, flags=re.IGNORECASE)
    return spoken


def _stock_phrase(stock: int) -> str:
    if stock == 0:
        return "zero units available"
    if stock == 1:
        return "one unit available"
    return f"{_spoken_int(stock)} units available"


def _availability_state(item: dict[str, Any]) -> str:
    if item.get("availabilityStatus"):
        return str(item["availabilityStatus"])
    stock = int(item.get("stock", 0))
    reorder_point = int(item.get("reorderPoint") or 0)
    if stock <= 0:
        return "out_of_stock"
    if reorder_point and stock <= reorder_point:
        return "low_stock"
    return "in_stock"


def _availability_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"in_stock": 0, "low_stock": 0, "out_of_stock": 0}
    total_units = 0
    low_stock: list[dict[str, Any]] = []
    out_of_stock: list[dict[str, Any]] = []
    for item in items:
        state = _availability_state(item)
        counts[state] = counts.get(state, 0) + 1
        total_units += int(item.get("stock", 0))
        if state == "low_stock":
            low_stock.append(_public_item(item))
        elif state == "out_of_stock":
            out_of_stock.append(_public_item(item))
    return {
        "items": len(items),
        "totalUnits": total_units,
        "inStock": counts.get("in_stock", 0),
        "lowStock": counts.get("low_stock", 0),
        "outOfStock": counts.get("out_of_stock", 0),
        "lowStockItems": low_stock[:5],
        "outOfStockItems": out_of_stock[:5],
    }


def _speech_line(item: dict[str, Any]) -> str:
    spoken = item.get("spoken") or {}
    return (
        f"{spoken.get('name', item['name'])}, aisle {spoken.get('aisle', item['aisle'])}, "
        f"bay {spoken.get('bay', item['bay'])}, with {spoken.get('stock', _stock_phrase(item['stock']))}"
    )


def _speech_answer(public_matches: list[dict[str, Any]]) -> str:
    if len(public_matches) == 1:
        return f"{_speech_line(public_matches[0])}."
    lines = [_speech_line(item) for item in public_matches[:MAX_SPOKEN_MATCHES]]
    return f"I found {len(public_matches)} matching items. " + "; ".join(lines) + "."




def _stable_number(value: str) -> int:
    digest = blake2b(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _customer_review_summary(item: dict[str, Any]) -> dict[str, Any]:
    seed = _stable_number(str(item.get("sku") or item.get("name") or "item"))
    attributes = {str(value).lower() for value in item.get("attributes", [])}
    shelf_tags = {str(value).lower() for value in item.get("shelfTags", [])}
    price = float(item.get("price") or 0)
    stock = int(item.get("stock") or 0)
    velocity = float(item.get("dailyVelocity") or 0)

    rating = 3.7 + ((seed % 13) / 10)
    if "grain free" in attributes:
        rating += 0.08
    if "puppy" in attributes:
        rating += 0.05
    if "wet food" in attributes:
        rating += 0.04
    if stock <= 0:
        rating -= 0.18
    rating = round(min(4.9, max(3.6, rating)), 1)

    review_count = int(48 + (seed % 620) + min(260, velocity * 72) + min(120, stock * 3))
    themes: list[str] = []
    concerns: list[str] = []

    if price and price < 15:
        themes.append("good value")
    elif price > 30:
        themes.append("premium formula")
        concerns.append("higher price")
    if "grain free" in attributes:
        themes.append("grain-free recipe")
    if "puppy" in attributes:
        themes.append("puppy formula")
    if "wet food" in attributes:
        themes.append("wet food texture")
    if "adult" in attributes:
        themes.append("adult daily feeding")
    if "dry food" in attributes:
        themes.append("dry food convenience")
    if any("large" in tag for tag in shelf_tags) or "lb" in str(item.get("size", "")).lower():
        themes.append("bulk bag value")
    if stock <= int(item.get("reorderPoint") or 0):
        concerns.append("limited shelf quantity")
    if not themes:
        themes.append("reliable everyday option")

    return {
        "rating": rating,
        "reviewCount": review_count,
        "positiveThemes": list(dict.fromkeys(themes))[:3],
        "concerns": list(dict.fromkeys(concerns))[:2],
        "source": "synthetic_customer_review_signal",
        "sampleData": True,
    }


def _review_recommendation_score(item: dict[str, Any], *, min_price: float, max_price: float) -> tuple[float, dict[str, Any]]:
    reviews = _customer_review_summary(item)
    rating_norm = (float(reviews["rating"]) - 3.5) / 1.4
    review_volume_norm = min(1.0, float(reviews["reviewCount"]) / 720)
    stock = int(item.get("stock") or 0)
    reorder_point = max(1, int(item.get("reorderPoint") or 6))
    availability_norm = 0.0 if stock <= 0 else min(1.0, stock / (reorder_point * 2))
    price = float(item.get("price") or 0)
    if max_price > min_price and price:
        value_norm = 1.0 - ((price - min_price) / (max_price - min_price))
    else:
        value_norm = 0.72
    confidence = float(item.get("inventoryConfidence") or 0.9)
    score = (0.48 * rating_norm) + (0.18 * review_volume_norm) + (0.18 * availability_norm) + (0.10 * value_norm) + (0.06 * confidence)
    return round(max(0.0, min(1.0, score)), 3), reviews


def _best_option(public_matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not public_matches:
        return None
    prices = [float(item.get("price") or 0) for item in public_matches if item.get("price") is not None]
    min_price = min(prices) if prices else 0.0
    max_price = max(prices) if prices else min_price
    ranked: list[dict[str, Any]] = []
    for item in public_matches:
        score, reviews = _review_recommendation_score(item, min_price=min_price, max_price=max_price)
        enriched = {**item, "customerReviewSummary": reviews, "recommendationScore": score}
        ranked.append(enriched)
    ranked.sort(
        key=lambda item: (
            item.get("recommendationScore", 0),
            item.get("customerReviewSummary", {}).get("rating", 0),
            item.get("stock", 0),
        ),
        reverse=True,
    )
    winner = ranked[0]
    reviews = winner["customerReviewSummary"]
    reason_parts = [
        f"{reviews['rating']} stars from {reviews['reviewCount']} sample reviews",
        f"{winner.get('stock', 0)} units available",
    ]
    if reviews.get("positiveThemes"):
        reason_parts.append(", ".join(reviews["positiveThemes"][:2]))
    return {
        "item": winner,
        "score": winner["recommendationScore"],
        "basis": "customer_review_signal_plus_availability_and_value",
        "reason": "; ".join(reason_parts),
        "alternativesRanked": ranked[1:4],
    }


def _best_option_speech(best_option: dict[str, Any]) -> str:
    item = best_option["item"]
    reviews = item.get("customerReviewSummary") or {}
    themes = reviews.get("positiveThemes") or []
    theme_text = f", with customers liking {themes[0]}" if themes else ""
    return (
        f"Best overall is {item['spoken']['name']}, with {reviews.get('rating')} stars from "
        f"{reviews.get('reviewCount')} sample reviews{theme_text}. "
        f"It is on aisle {item['spoken']['aisle']}, bay {item['spoken']['bay']}, with {item['spoken']['stock']}."
    )

def _score_item(query: str, item: dict[str, Any]) -> tuple[float, str]:
    query_tokens = _tokens(query)
    query_norm = _normalize(query)
    if query_norm and query_norm == _normalize(item["name"]):
        return 1.0, "exact product name match"
    aliases = _aliases(item)
    item_tokens = _tokens(" ".join(aliases))
    overlap = query_tokens & item_tokens
    query_overlap = len(overlap) / max(1, len(query_tokens))
    item_overlap = len(overlap) / max(1, len(item_tokens))
    phrase_score = max(SequenceMatcher(None, query_norm, _normalize(alias)).ratio() for alias in aliases)
    contains_score = 0.0

    for alias in aliases:
        alias_norm = _normalize(alias)
        if alias_norm and alias_norm in query_norm:
            alias_tokens = _tokens(alias_norm)
            if alias_norm == query_norm:
                contains_score = max(contains_score, 1.0)
            elif len(alias_tokens) >= 2 and not alias_tokens & BROAD_SINGLE_TOKEN_ALIASES:
                contains_score = max(contains_score, 1.0)
            elif not alias_tokens & BROAD_SINGLE_TOKEN_ALIASES:
                contains_score = max(contains_score, 0.7)

    score = max(query_overlap, item_overlap, phrase_score * 0.92, contains_score)
    if len(query_tokens) >= 3 and len(overlap) <= 1 and contains_score < 1.0:
        score = min(score, 0.38)
    if query_tokens and not overlap and contains_score == 0:
        score = min(score, 0.38)

    if contains_score >= 1:
        reason = "exact phrase match"
    elif query_overlap >= 0.8:
        reason = "category/product word match"
    elif item_overlap >= phrase_score:
        reason = "product word overlap"
    elif query_tokens and not overlap:
        reason = "no catalog token overlap"
    else:
        reason = "fuzzy product match"
    return score, reason


@lru_cache(maxsize=1)
def load_inventory() -> list[dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ranked_inventory(query: str) -> list[InventoryMatch]:
    matches: list[InventoryMatch] = []
    for item in load_inventory():
        score, reason = _score_item(query, item)
        matches.append(InventoryMatch(item=item, score=round(score, 3), reason=reason))
    return sorted(matches, key=lambda match: (match.score, match.reason == "exact product name match"), reverse=True)


def search_inventory(query: str) -> InventoryMatch:
    ranked = ranked_inventory(query)
    best = ranked[0] if ranked else InventoryMatch(item=None, score=0.0, reason="no inventory loaded")
    best_score = best.score
    if best_score < 0.42:
        return InventoryMatch(item=None, score=best_score, reason=best.reason)
    return best


def _public_item(item: dict[str, Any]) -> dict[str, Any]:
    public = {
        "sku": item["sku"],
        "name": item["name"],
        "brand": item.get("brand"),
        "department": item["department"],
        "category": item.get("category"),
        "subcategory": item.get("subcategory"),
        "aisle": item["aisle"],
        "bay": item["bay"],
        "stock": item["stock"],
        "availabilityStatus": item.get("availabilityStatus"),
        "price": item["price"],
        "size": item.get("size"),
        "unit": item.get("unit"),
        "notes": item["notes"],
        "locationHint": item.get("locationHint"),
        "shelfLevel": item.get("shelfLevel"),
        "temperature": item.get("temperature"),
        "reorderPoint": item.get("reorderPoint"),
        "dailyVelocity": item.get("dailyVelocity"),
        "restockEta": item.get("restockEta"),
        "fulfillment": item.get("fulfillment", []),
        "attributes": item.get("attributes", []),
        "allergens": item.get("allergens", []),
        "substitutes": item.get("substitutes", []),
        "ageRestricted": item.get("ageRestricted", False),
        "inventoryConfidence": item.get("inventoryConfidence"),
        "lastUpdated": item.get("lastUpdated"),
        "shelfTags": item.get("shelfTags", []),
        "customerReviewSummary": _customer_review_summary(item),
        "spoken": {
            "name": _spoken_name(item["name"]),
            "aisle": _spoken_code(item["aisle"]),
            "bay": _spoken_bay(item["bay"]),
            "stock": _stock_phrase(int(item["stock"])),
        },
    }
    return {key: value for key, value in public.items() if value is not None}


def _retrieval_evidence(matches: list[InventoryMatch]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for rank, match in enumerate(matches, start=1):
        if not match.item:
            continue
        item = match.item
        evidence.append(
            {
                "rank": rank,
                "sku": item["sku"],
                "name": item["name"],
                "score": match.score,
                "reason": match.reason,
                "department": item.get("department"),
                "category": item.get("category"),
                "subcategory": item.get("subcategory"),
                "aisle": item.get("aisle"),
                "bay": item.get("bay"),
                "availabilityStatus": _availability_state(item),
                "stock": item.get("stock"),
                "customerRating": _customer_review_summary(item)["rating"],
                "reviewCount": _customer_review_summary(item)["reviewCount"],
                "shelfTags": item.get("shelfTags", [])[:6],
            }
        )
    return evidence


def _shared_values(left: dict[str, Any], right: dict[str, Any], key: str) -> set[str]:
    left_values = {str(value).lower() for value in left.get(key, []) if value}
    right_values = {str(value).lower() for value in right.get(key, []) if value}
    return left_values & right_values


def _relation_aliases(item: dict[str, Any]) -> set[str]:
    aliases = [item.get("subcategory")]
    aliases.extend(item.get("synonyms") or [])
    aliases.extend(item.get("attributes") or [])
    aliases.extend(item.get("shelfTags") or [])
    values: set[str] = set()
    for alias in aliases:
        normalized = _normalize(str(alias))
        if not normalized:
            continue
        tokens = _tokens(normalized) - BROAD_SINGLE_TOKEN_ALIASES
        if tokens and normalized not in GENERIC_RELATION_ALIASES and not all(token.isdigit() for token in tokens):
            values.add(normalized)
    return values


def _relation_terms(item: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for alias in _relation_aliases(item):
        terms.update(_tokens(alias))
    return terms - BROAD_SINGLE_TOKEN_ALIASES


def _same_product_family(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if anchor.get("subcategory") and anchor.get("subcategory") == candidate.get("subcategory"):
        return True
    if _relation_aliases(anchor) & _relation_aliases(candidate):
        return True
    anchor_terms = _relation_terms(anchor)
    candidate_terms = _relation_terms(candidate)
    overlap = anchor_terms & candidate_terms
    specific_overlap = overlap - {"food", "pet", "pets"}
    return ("food" in overlap and bool(specific_overlap)) or len(specific_overlap) >= 2


def _substitute_score(anchor: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if not _same_product_family(anchor, candidate):
        return 0.0, ["different product family"]
    score += 3.5
    reasons.append("same product family")
    if anchor.get("subcategory") and anchor.get("subcategory") == candidate.get("subcategory"):
        score += 6
        reasons.append("same subcategory")
    if anchor.get("category") and anchor.get("category") == candidate.get("category"):
        score += 2
        reasons.append("same category")
    if anchor.get("department") and anchor.get("department") == candidate.get("department"):
        score += 1.5
        reasons.append("same department")
    shared_attributes = _shared_values(anchor, candidate, "attributes")
    if shared_attributes:
        score += min(2.5, len(shared_attributes))
        reasons.append("shared attributes")
    if anchor.get("aisle") == candidate.get("aisle"):
        score += 0.75
        reasons.append("nearby shelf")
    if int(candidate.get("stock", 0)) > 0:
        score += 1
        reasons.append("available now")
    else:
        score -= 3
        reasons.append("currently unavailable")
    return score, reasons


def _relation_item(candidate: dict[str, Any], score: float, reasons: list[str], relation_type: str) -> dict[str, Any]:
    public = _public_item(candidate)
    public["relationType"] = relation_type
    public["relationScore"] = round(score, 3)
    public["relationReason"] = ", ".join(dict.fromkeys(reasons)) or "catalog proximity"
    return public


def _substitutes_for(anchor: dict[str, Any], *, limit: int = MAX_RELATED_ITEMS) -> list[dict[str, Any]]:
    ranked: list[tuple[float, list[str], dict[str, Any]]] = []
    for candidate in load_inventory():
        if candidate["sku"] == anchor["sku"]:
            continue
        if int(candidate.get("stock", 0)) <= 0:
            continue
        score, reasons = _substitute_score(anchor, candidate)
        if score >= 4:
            ranked.append((score, reasons, candidate))
    ranked.sort(key=lambda entry: (entry[0], int(entry[2].get("stock", 0))), reverse=True)
    return [_relation_item(candidate, score, reasons, "substitute") for score, reasons, candidate in ranked[:limit]]


def _complement_queries(anchor: dict[str, Any]) -> list[str]:
    queries = []
    relation_terms = _relation_terms(anchor)
    keys: list[str] = []
    if {"dog", "food"} <= relation_terms:
        keys.append("Dog Food")
    if {"cat", "food"} <= relation_terms:
        keys.append("Cat Food")
    keys.extend(str(anchor.get(key)) for key in ("subcategory", "category", "department") if anchor.get(key))
    for value in keys:
        if value in COMPLEMENT_QUERIES_BY_SUBCATEGORY:
            queries.extend(COMPLEMENT_QUERIES_BY_SUBCATEGORY[value])
    for synonym in anchor.get("synonyms", []):
        if synonym in COMPLEMENT_QUERIES_BY_SUBCATEGORY:
            queries.extend(COMPLEMENT_QUERIES_BY_SUBCATEGORY[synonym])
    return list(dict.fromkeys(queries))


def _complements_for(anchor: dict[str, Any], *, limit: int = MAX_RELATED_ITEMS) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for query in _complement_queries(anchor):
        for match in ranked_inventory(query)[:3]:
            if not match.item or match.item["sku"] == anchor["sku"] or match.item["sku"] in selected:
                continue
            if match.score < 0.42:
                continue
            if int(match.item.get("stock", 0)) <= 0:
                continue
            selected[match.item["sku"]] = _relation_item(
                match.item,
                max(0.1, match.score),
                [f"complements {anchor.get('subcategory') or anchor.get('department')}", match.reason],
                "complement",
            )
            break

    if len(selected) < limit:
        for candidate in load_inventory():
            if candidate["sku"] == anchor["sku"] or candidate["sku"] in selected:
                continue
            if candidate.get("aisle") != anchor.get("aisle"):
                continue
            if candidate.get("subcategory") == anchor.get("subcategory"):
                continue
            if _same_product_family(anchor, candidate):
                continue
            if int(candidate.get("stock", 0)) <= 0:
                continue
            selected[candidate["sku"]] = _relation_item(candidate, 0.55, ["same aisle", "different category"], "complement")
            if len(selected) >= limit:
                break
    return list(selected.values())[:limit]


def catalog_summary() -> dict[str, Any]:
    items = load_inventory()
    departments: dict[str, int] = {}
    categories: dict[str, int] = {}
    aisles: dict[str, int] = {}
    for item in items:
        departments[str(item.get("department") or "Unknown")] = departments.get(str(item.get("department") or "Unknown"), 0) + 1
        categories[str(item.get("category") or "Unknown")] = categories.get(str(item.get("category") or "Unknown"), 0) + 1
        aisles[str(item.get("aisle") or "Unknown")] = aisles.get(str(item.get("aisle") or "Unknown"), 0) + 1

    return {
        "totalProducts": len(items),
        "departments": dict(sorted(departments.items())),
        "categories": dict(sorted(categories.items())),
        "topDepartments": sorted(departments.items(), key=lambda entry: entry[1], reverse=True)[:8],
        "topAisles": sorted(aisles.items(), key=lambda entry: entry[1], reverse=True)[:8],
        "availability": _availability_summary(items),
    }


def product_relations_payload(query: str, *, limit: int = MAX_RELATED_ITEMS) -> dict[str, Any]:
    canonical = canonicalize_query(query)
    search_query = str(canonical.get("canonicalText") or query)
    matches = _top_matches(search_query)
    if not matches:
        best = search_inventory(search_query)
        return {
            "tool": "catalog_relations",
            "query": query,
            "canonicalQuery": search_query,
            "multilingual": canonical,
            "found": False,
            "score": best.score,
            "reason": best.reason,
            "alternatives": [],
            "complements": [],
            "retrievalEvidence": [],
        }
    anchor = matches[0].item
    assert anchor is not None
    return {
        "tool": "catalog_relations",
        "query": query,
        "canonicalQuery": search_query,
        "multilingual": canonical,
        "found": True,
        "score": matches[0].score,
        "reason": matches[0].reason,
        "anchor": _public_item(anchor),
        "alternatives": _substitutes_for(anchor, limit=limit),
        "complements": _complements_for(anchor, limit=limit),
        "availabilitySummary": _availability_summary([match.item for match in matches if match.item]),
        "retrievalEvidence": _retrieval_evidence(matches),
    }


def _top_matches(query: str) -> list[InventoryMatch]:
    ranked = ranked_inventory(query)
    if not ranked:
        return []
    if ranked[0].reason == "exact product name match":
        return [ranked[0]]

    top_score = ranked[0].score
    minimum_score = max(0.42, min(0.72, top_score - 0.18))
    return [match for match in ranked if match.item and match.score >= minimum_score][:MAX_TOOL_MATCHES]


def tool_payload(query: str) -> dict[str, Any]:
    canonical = canonicalize_query(query)
    search_query = str(canonical.get("canonicalText") or query)
    language = str(canonical.get("language") or "unknown")
    matches = _top_matches(search_query)
    if not matches:
        best = search_inventory(search_query)
        speech_answer = "I do not see that item in the current shopping catalog. Try another product or give me the exact name and I can check again."
        return {
            "tool": "lookup_inventory",
            "query": query,
            "canonicalQuery": search_query,
            "multilingual": canonical,
            "found": False,
            "score": best.score,
            "reason": best.reason,
            "message": "No matching item in the shopping catalog.",
            "speechAnswer": localize_answer(speech_answer, language),
            "guidance": "Say that the item is not in the current shopping catalog, then ask for another product or an exact name.",
        }

    list_intent = bool(LIST_INTENT_RE.search(search_query))
    best_intent = bool(BEST_INTENT_RE.search(search_query))
    match_type = "recommendation" if best_intent and len(matches) > 1 else "category" if list_intent or len(matches) > 1 else "single"
    item = matches[0].item
    assert item is not None
    public_matches = [_public_item(match.item) for match in matches if match.item]
    aisles = sorted({entry["aisle"] for entry in public_matches})
    raw_matches = [match.item for match in matches if match.item]
    alternatives = _substitutes_for(item, limit=3)
    complements = _complements_for(item, limit=3)
    best_option = _best_option(public_matches) if len(public_matches) > 1 else None
    speech_answer = (
        _best_option_speech(best_option)
        if best_intent and best_option
        else _speech_answer(public_matches if match_type == "category" else public_matches[:1])
    )
    speech_answer = localize_answer(speech_answer, language)

    return {
        "tool": "lookup_inventory",
        "query": query,
        "canonicalQuery": search_query,
        "multilingual": canonical,
        "found": True,
        "score": matches[0].score,
        "reason": matches[0].reason,
        "matchType": match_type,
        "itemCount": len(public_matches),
        "aisles": aisles,
        "item": _public_item(item),
        "matches": public_matches,
        "availabilitySummary": _availability_summary(raw_matches),
        "alternatives": alternatives,
        "complements": complements,
        "bestOption": best_option,
        "retrievalEvidence": _retrieval_evidence(matches),
        "grounding": {
            "source": "synthetic_enriched_catalog",
            "evidenceCount": len(matches),
            "usedFields": ["name", "brand", "department", "category", "subcategory", "synonyms", "attributes", "shelfTags", "customerReviewSummary"],
        },
        "speechAnswer": speech_answer,
        "guidance": (
            "Use speechAnswer as the spoken response whenever possible. "
            "It expands aisle codes, bay codes, package units, and stock counts for clearer TTS. "
            "Never say 'in stock' as shorthand for the number; say 'eighteen units available' or 'zero units available'."
            " If the user asks for substitutes, alternatives, or related products, use the alternatives and complements arrays."
            " If the user asks for the best, top-rated, or recommended option, use bestOption and mention rating, review count, location, and stock."
        ),
    }


def template_answer(payload: dict[str, Any]) -> str:
    if not payload["found"]:
        return payload.get("speechAnswer") or "I do not see that item in this shopping catalog. Try the exact product name and I can check again."

    return payload.get("speechAnswer") or f"{payload['item']['name']} is on aisle {payload['item']['aisle']}."




