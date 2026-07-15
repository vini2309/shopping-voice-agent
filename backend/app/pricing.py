from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PriceBook:
    vapi_platform_per_minute: float = 0.05
    deepgram_nova3_streaming_per_minute: float = 0.0048
    deepgram_keyterm_per_minute: float = 0.0013
    gpt4o_mini_input_per_million: float = 0.15
    gpt4o_mini_output_per_million: float = 0.60
    elevenlabs_turbo_per_1k_chars: float = 0.05
    openai_realtime_audio_input_per_million: float = 32.00
    openai_realtime_audio_output_per_million: float = 64.00
    gemini_live_audio_input_per_million: float = 3.00
    gemini_live_audio_output_per_million: float = 12.00


PRICE_BOOK = PriceBook()

PRICING_SOURCES = {
    "vapi": "https://vapi.ai/pricing",
    "deepgram": "https://deepgram.com/pricing",
    "openai": "https://platform.openai.com/docs/pricing",
    "elevenlabs": "https://elevenlabs.io/pricing/api",
    "gemini": "https://ai.google.dev/gemini-api/docs/pricing",
}


def _tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def _spoken_seconds(text: str) -> float:
    return max(0.8, len(text) / 14.5)


def _audio_tokens(seconds: float) -> int:
    return max(1, round(seconds * 50))


def estimate_costs(
    *,
    call_ms: int,
    user_text: str,
    answer: str,
    total_latency_ms: int | None = None,
) -> dict[str, Any]:
    call_minutes = max(call_ms / 60_000, 1 / 6000)
    input_tokens = _tokens(user_text) + 220
    output_tokens = _tokens(answer)
    output_seconds = _spoken_seconds(answer)
    input_audio_tokens = _audio_tokens(max(call_ms / 1000, 0.1))
    output_audio_tokens = _audio_tokens(output_seconds)

    parts = {
        "vapi": call_minutes * PRICE_BOOK.vapi_platform_per_minute,
        "deepgram": call_minutes
        * (PRICE_BOOK.deepgram_nova3_streaming_per_minute + PRICE_BOOK.deepgram_keyterm_per_minute),
        "gpt4oMini": (
            input_tokens / 1_000_000 * PRICE_BOOK.gpt4o_mini_input_per_million
            + output_tokens / 1_000_000 * PRICE_BOOK.gpt4o_mini_output_per_million
        ),
        "elevenlabs": len(answer) / 1000 * PRICE_BOOK.elevenlabs_turbo_per_1k_chars,
    }
    vapi_stack = sum(parts.values())
    openai_realtime = (
        input_audio_tokens / 1_000_000 * PRICE_BOOK.openai_realtime_audio_input_per_million
        + output_audio_tokens / 1_000_000 * PRICE_BOOK.openai_realtime_audio_output_per_million
    )
    gemini_live = (
        input_audio_tokens / 1_000_000 * PRICE_BOOK.gemini_live_audio_input_per_million
        + output_audio_tokens / 1_000_000 * PRICE_BOOK.gemini_live_audio_output_per_million
    )

    rows = [
        {
            "id": "vapi-stack",
            "label": "Composed voice agent",
            "mode": "live estimate",
            "cost": vapi_stack,
            "per1000": vapi_stack * 1000,
            "latencyMs": total_latency_ms,
            "parts": parts,
        },
        {
            "id": "openai",
            "label": "Native realtime baseline",
            "mode": "published pricing baseline",
            "cost": openai_realtime,
            "per1000": openai_realtime * 1000,
            "latencyMs": 950,
        },
        {
            "id": "gemini",
            "label": "Native audio baseline",
            "mode": "published pricing baseline",
            "cost": gemini_live,
            "per1000": gemini_live * 1000,
            "latencyMs": 900,
        },
    ]

    return {
        "rows": rows,
        "assumptions": {
            "callSeconds": round(call_ms / 1000, 2),
            "audioOutputSeconds": round(output_seconds, 2),
            "promptTokens": input_tokens,
            "completionTokens": output_tokens,
        },
        "priceBook": PRICE_BOOK.__dict__,
        "sources": PRICING_SOURCES,
    }
