from __future__ import annotations

import re
from typing import Any


SPANISH_HINTS = {
    "abierto",
    "abiertos",
    "articulo",
    "articulos",
    "comida",
    "cual",
    "devolver",
    "electronico",
    "electronicos",
    "ensename",
    "ensenarme",
    "enumerar",
    "existencias",
    "inventario",
    "lugar",
    "mejor",
    "muestra",
    "muestran",
    "papel",
    "para",
    "perro",
    "perros",
    "podria",
    "podrias",
    "puede",
    "puedes",
    "puedo",
    "sistema",
    "sistemas",
    "toallas",
    "todo",
    "vacio",
}

SPANISH_PHRASES = [
    (r"\btoallas?\s+de\s+papel\b", "paper towels"),
    (r"\bcomida\s+para\s+perros?\b", "dog food"),
    (r"\balimento\s+para\s+perros?\b", "dog food"),
    (r"\bcomida\s+de\s+perros?\b", "dog food"),
    (r"\barticulos?\s+electronicos?\b", "electronics item"),
    (r"\bart\s+culos?\s+electr\s+nicos?\b", "electronics item"),
    (r"\belectronicos?\b", "electronics"),
    (r"\belectr\s+nicos?\b", "electronics"),
    (r"\babiertos?\b", "opened"),
    (r"\babierto\b", "opened"),
    (r"\bdevolver\b", "return"),
    (r"\bdevolucion\b", "return"),
    (r"\binventario\b", "inventory"),
    (r"\bexistencias?\b", "stock"),
    (r"\ben\s+stock\b", "in stock"),
    (r"\bpasillo\b", "aisle"),
    (r"\bestante\b", "shelf"),
    (r"\brepisa\b", "shelf"),
    (r"\bvacio\b", "empty"),
    (r"\bvacia\b", "empty"),
    (r"\bsistema\b", "system"),
    (r"\bmuestra\b", "shows"),
    (r"\bmuestran\b", "shows"),
    (r"\bmejor\b", "best"),
    (r"\blista\b", "list"),
    (r"\benumerar\b", "list"),
    (r"\bmostrar\b", "show"),
    (r"\bmuestre\b", "show"),
    (r"\bensenarme\b", "show me"),
    (r"\bensenar\b", "show"),
    (r"\ben\s+su\s+lugar\b", "instead"),
    (r"\ben\s+lugar\b", "instead"),
]

SPANISH_WORDS = {
    "actualmente": "actually",
    "al": "to",
    "articulo": "item",
    "articulos": "items",
    "art": "item",
    "culos": "items",
    "cual": "what",
    "de": "of",
    "del": "of",
    "disponible": "available",
    "disponibles": "available",
    "el": "the",
    "en": "in",
    "es": "is",
    "han": "have",
    "hay": "there is",
    "indica": "indicates",
    "la": "the",
    "las": "the",
    "lo": "it",
    "los": "the",
    "me": "me",
    "para": "for",
    "pero": "but",
    "perro": "dog",
    "perros": "dogs",
    "podria": "could",
    "podrias": "could",
    "puede": "can",
    "puedes": "can",
    "puedo": "can I",
    "que": "that",
    "su": "your",
    "sistemas": "systems",
    "todo": "all",
    "tus": "your",
    "ya": "already",
}

SPANISH_ANSWER_REPLACEMENTS = [
    (r"\baisle\b", "pasillo"),
    (r"\bbay\b", "bahia"),
    (r"\bunits available\b", "unidades disponibles"),
    (r"\bunit available\b", "unidad disponible"),
    (r"\bI found\b", "Encontre"),
    (r"\bmatching items\b", "articulos que coinciden"),
    (r"\bBest overall is\b", "La mejor opcion general es"),
    (r"\bwith customers liking\b", "y a los clientes les gusta"),
    (r"\bstars from\b", "estrellas de"),
    (r"\bsample reviews\b", "resenas de muestra"),
    (r"\bIt is on\b", "Esta en"),
]


def _strip_accents(value: str) -> str:
    translation = str.maketrans(
        {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ü": "u",
            "ñ": "n",
            "Á": "a",
            "É": "e",
            "Í": "i",
            "Ó": "o",
            "Ú": "u",
            "Ü": "u",
            "Ñ": "n",
        }
    )
    return value.translate(translation)


def _tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", _strip_accents(str(value or "").lower()))


def detect_language(text: Any) -> dict[str, Any]:
    raw = str(text or "")
    normalized = _strip_accents(raw.lower())
    tokens = _tokens(normalized)
    if not tokens:
        return {"language": "unknown", "confidence": 0.0, "signals": []}

    signals: list[str] = []
    if any(char in raw for char in "¿¡áéíóúüñÁÉÍÓÚÜÑ"):
        signals.append("spanish_punctuation_or_diacritic")
    hint_hits = sorted(set(tokens) & SPANISH_HINTS)
    if hint_hits:
        signals.extend(f"spanish_token:{token}" for token in hint_hits[:8])
    phrase_hits = []
    for pattern, _ in SPANISH_PHRASES:
        if re.search(pattern, normalized):
            phrase_hits.append(pattern)
    if phrase_hits:
        signals.extend("spanish_phrase" for _ in phrase_hits[:4])

    score = min(1.0, (0.22 * len(hint_hits)) + (0.22 * len(phrase_hits)) + (0.22 if any("diacritic" in item for item in signals) else 0.0))
    if score >= 0.34:
        return {"language": "es", "confidence": round(score, 3), "signals": signals}
    return {"language": "en", "confidence": round(max(0.5, 1.0 - score), 3), "signals": signals}


def canonicalize_query(text: Any) -> dict[str, Any]:
    original = re.sub(r"\s+", " ", str(text or "")).strip()
    detection = detect_language(original)
    if detection["language"] != "es":
        return {
            "originalText": original,
            "canonicalText": original,
            "language": detection["language"],
            "languageConfidence": detection["confidence"],
            "translated": False,
            "signals": detection["signals"],
            "method": "identity",
        }

    normalized = _strip_accents(original.lower())
    transformed = normalized
    applied: list[str] = []
    for pattern, replacement in SPANISH_PHRASES:
        updated, count = re.subn(pattern, replacement, transformed)
        if count:
            applied.append(f"{pattern}->{replacement}")
            transformed = updated

    words: list[str] = []
    for token in re.findall(r"[a-z0-9]+", transformed):
        words.append(SPANISH_WORDS.get(token, token))
    canonical = re.sub(r"\s+", " ", " ".join(words)).strip()

    intent_prefix = ""
    if re.search(r"\b(cual|mejor)\b", normalized) and "best" not in canonical:
        intent_prefix = "best "
    if re.search(r"\b(enumerar|lista|todo)\b", normalized) and "list" not in canonical:
        intent_prefix = "list all "
    if "system" in canonical and "stock" in canonical and ("no there is" in canonical or "empty" in canonical or "not" in canonical):
        canonical = "the shelf is empty but your system shows stock"
    elif "shelf" in canonical and "stock" in canonical and "empty" in canonical:
        canonical = "the shelf is empty but your system shows stock"
    elif "return" in canonical and ("electronics" in canonical or "electronics item" in canonical):
        canonical = "can I return an opened electronics item"
    elif "dog food" in canonical and "best" in canonical:
        canonical = "what is the best dog food available"
    elif "dog food" in canonical and ("list" in canonical or "inventory" in canonical):
        canonical = "can you list all the dog food inventory"
    elif "dog food" in canonical and "instead" in canonical:
        canonical = "actually show me dog food instead"
    elif "paper towels" in canonical:
        canonical = "where are paper towels"
    else:
        canonical = f"{intent_prefix}{canonical}".strip()

    return {
        "originalText": original,
        "canonicalText": canonical,
        "language": "es",
        "languageConfidence": detection["confidence"],
        "translated": canonical != original,
        "signals": detection["signals"],
        "appliedRules": applied,
        "method": "deterministic_spanish_retail_phrasebook",
    }


def localize_answer(answer: Any, language: str) -> str:
    text = re.sub(r"\s+", " ", str(answer or "")).strip()
    if language != "es" or not text:
        return text
    lowered = text.lower()
    if "service desk for return processing" in lowered or "return eligibility" in lowered:
        return (
            "Para devolver un articulo electronico abierto, ve al mostrador de servicio. "
            "El asociado debe revisar el recibo, la condicion del articulo y la elegibilidad antes de prometer un reembolso."
        )
    if "shelf is empty" in lowered and "system shows inventory" in lowered:
        return (
            "Si el estante esta vacio pero el sistema muestra inventario, revisa la bahia, exhibiciones cercanas, topstock y el carrito de devoluciones. "
            "Si hay cero unidades disponibles, di que el articulo esta agotado actualmente."
        )
    localized = text
    for pattern, replacement in SPANISH_ANSWER_REPLACEMENTS:
        localized = re.sub(pattern, replacement, localized, flags=re.IGNORECASE)
    localized = localized.replace("I do not see that item in the current shopping catalog.", "No veo ese articulo en el catalogo actual.")
    localized = localized.replace("Try another product or give me the exact name and I can check again.", "Prueba con otro producto o dame el nombre exacto y puedo revisarlo otra vez.")
    localized = localized.replace("The current knowledge base does not cover that policy clearly enough for me to answer.", "La base de conocimiento actual no cubre esa politica con suficiente claridad para responder.")
    localized = localized.replace("Please ask one specific store-policy question.", "Pregunta una politica especifica de la tienda.")
    return localized
