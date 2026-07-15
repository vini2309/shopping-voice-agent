from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.trace_eval import evaluate_saved_traces, evaluate_trace_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved AislePilot traces for trust, grounding, privacy, and replay stability.")
    parser.add_argument("--trace-id", default=None, help="Evaluate one trace ID.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    output = evaluate_trace_id(args.trace_id) if args.trace_id else evaluate_saved_traces(limit=args.limit)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
