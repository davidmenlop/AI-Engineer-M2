"""Extracción y validación de las citas inline [chunk_id] de una respuesta."""

from __future__ import annotations

import re
from collections.abc import Iterable

BRACKETS = re.compile(r"\[([^\[\]\n]{1,300})\]")
CHUNK_ID = re.compile(r"^[\w.-]+\.md_\d+$")


def extract_citations(answer: str) -> list[str]:
    """Ids citados, sin repetir, en orden de aparición. Acepta [a] y [a, b]."""
    found: list[str] = []
    for group in BRACKETS.findall(answer):
        for token in re.split(r"[,;]\s*|\s+", group.strip()):
            token = token.strip()
            if CHUNK_ID.match(token) and token not in found:
                found.append(token)
    return found


def unknown_citations(cited: Iterable[str], allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    return [chunk_id for chunk_id in cited if chunk_id not in allowed_set]


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def is_refusal(answer: str, refusal_text: str) -> bool:
    """Rechazo = la respuesta empieza con el texto de rechazo fijo (ADR-044)."""
    return normalize_text(answer).startswith(normalize_text(refusal_text)[:60])
