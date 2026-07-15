from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.traces import TRACE_DIR, replay_trace  # noqa: E402


def trace_ids(path: Path | None) -> list[str]:
    directory = path or TRACE_DIR
    if directory.is_file():
        return [directory.stem]
    if not directory.exists():
        return []
    return [item.stem for item in sorted(directory.glob("*.json"))]


def summarize(replays: list[dict[str, Any]]) -> dict[str, Any]:
    tool_calls = sum(item["replay"]["toolCalls"] for item in replays)
    matches = sum(item["replay"]["deterministicMatches"] for item in replays)
    return {
        "traces": len(replays),
        "toolCalls": tool_calls,
        "deterministicMatches": matches,
        "deterministicMatchRate": matches / tool_calls if tool_calls else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay saved AislePilot trace tool calls against the current backend.")
    parser.add_argument("--trace-dir", type=Path, default=None, help="Directory of trace JSON files. Defaults to artifacts/traces.")
    parser.add_argument("--trace-id", default=None, help="Replay one trace ID.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ids = [args.trace_id] if args.trace_id else trace_ids(args.trace_dir)
    replays = [replay_trace(trace_id) for trace_id in ids]
    output = {
        "summary": summarize(replays),
        "results": replays,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
