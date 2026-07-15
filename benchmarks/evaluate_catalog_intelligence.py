from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.inventory import catalog_summary, load_inventory, product_relations_payload  # noqa: E402


DEFAULT_OUT = ROOT / "artifacts" / "catalog_intelligence_eval.json"


def relation_checks(anchor: dict[str, Any], item: dict[str, Any], *, relation_type: str) -> dict[str, bool]:
    same_sku = item.get("sku") == anchor.get("sku")
    available = int(item.get("stock") or 0) > 0 and item.get("availabilityStatus") != "out_of_stock"
    same_category = item.get("category") == anchor.get("category")
    same_subcategory = item.get("subcategory") == anchor.get("subcategory")
    different_subcategory = item.get("subcategory") != anchor.get("subcategory")
    if relation_type == "substitute":
        semantically_valid = same_category or same_subcategory
    else:
        semantically_valid = same_category or different_subcategory
    return {
        "notSelf": not same_sku,
        "available": available,
        "semanticallyValid": semantically_valid,
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate() -> dict[str, Any]:
    items = load_inventory()
    summary = catalog_summary()
    relation_results: list[dict[str, Any]] = []
    alternative_checks: list[dict[str, bool]] = []
    complement_checks: list[dict[str, bool]] = []

    for item in items:
        payload = product_relations_payload(item["name"], limit=4)
        anchor = payload.get("anchor") or {}
        alternatives = payload.get("alternatives") or []
        complements = payload.get("complements") or []
        for alternative in alternatives:
            alternative_checks.append(relation_checks(anchor, alternative, relation_type="substitute"))
        for complement in complements:
            complement_checks.append(relation_checks(anchor, complement, relation_type="complement"))
        relation_results.append(
            {
                "sku": item["sku"],
                "name": item["name"],
                "found": payload.get("found", False),
                "alternatives": [entry.get("sku") for entry in alternatives],
                "complements": [entry.get("sku") for entry in complements],
                "evidenceCount": len(payload.get("retrievalEvidence") or []),
            }
        )

    availability = summary.get("availability") or {}
    availability_count = sum(int(availability.get(key) or 0) for key in ("inStock", "lowStock", "outOfStock"))
    relation_covered = sum(1 for row in relation_results if row["alternatives"] or row["complements"])
    found_count = sum(1 for row in relation_results if row["found"])
    alternative_available = sum(1 for check in alternative_checks if check["available"])
    alternative_not_self = sum(1 for check in alternative_checks if check["notSelf"])
    alternative_valid = sum(1 for check in alternative_checks if check["semanticallyValid"])
    complement_available = sum(1 for check in complement_checks if check["available"])
    complement_not_self = sum(1 for check in complement_checks if check["notSelf"])
    complement_valid = sum(1 for check in complement_checks if check["semanticallyValid"])

    return {
        "summary": {
            "totalProducts": len(items),
            "catalogSummaryProducts": summary.get("totalProducts"),
            "summaryConsistent": summary.get("totalProducts") == len(items) and availability_count == len(items),
            "relationsEvaluated": len(relation_results),
            "lookupFoundRate": safe_rate(found_count, len(relation_results)),
            "relationCoverage": safe_rate(relation_covered, len(relation_results)),
            "alternativeCount": len(alternative_checks),
            "alternativeAvailabilityRate": safe_rate(alternative_available, len(alternative_checks)),
            "alternativeNotSelfRate": safe_rate(alternative_not_self, len(alternative_checks)),
            "alternativeSemanticValidityRate": safe_rate(alternative_valid, len(alternative_checks)),
            "complementCount": len(complement_checks),
            "complementAvailabilityRate": safe_rate(complement_available, len(complement_checks)),
            "complementNotSelfRate": safe_rate(complement_not_self, len(complement_checks)),
            "complementSemanticValidityRate": safe_rate(complement_valid, len(complement_checks)),
        },
        "catalogSummary": summary,
        "results": relation_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate catalog relation and evidence quality.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
