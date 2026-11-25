"""Métricas de retrieval y de generación. Funciones puras, sin red.

Retrieval, por pregunta dentro de alcance, sobre los k chunks recuperados:

- precision@k        relevantes recuperados / k
- recall@k           relevantes recuperados / relevantes esperados (a nivel chunk)
- evidence_recall@k  evidencias cubiertas / evidencias. Una evidencia está cubierta si
                     los chunks recuperados cubren >= COVERAGE de sus caracteres. No
                     castiga a una estrategia por partir un dato en dos chunks solapados
                     cuando uno solo ya lo contiene (ADR-042).
- complete@k         1 si todas las evidencias de la pregunta quedaron cubiertas
- hit@k              1 si al menos un chunk recuperado es relevante
- mrr                1 / posición del primer chunk relevante (0 si no hay)
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

COVERAGE = 0.8


@dataclass(frozen=True)
class Span:
    source: str
    start: int
    end: int


def covered_fraction(evidence: Span, spans: Iterable[Span]) -> float:
    """Fracción de los caracteres de la evidencia cubierta por la unión de los tramos."""
    length = evidence.end - evidence.start
    if length <= 0:
        return 0.0
    covered = [False] * length
    for span in spans:
        if span.source != evidence.source:
            continue
        lo = max(evidence.start, span.start)
        hi = min(evidence.end, span.end)
        for position in range(lo, hi):
            covered[position - evidence.start] = True
    return sum(covered) / length


def retrieval_scores(
    retrieved: Sequence[str],
    expected: Sequence[str],
    evidence: Sequence[Span],
    retrieved_spans: Sequence[Span],
    k: int,
) -> dict[str, float]:
    expected_set = set(expected)
    relevant_flags = [chunk_id in expected_set for chunk_id in retrieved[:k]]
    hits = sum(relevant_flags)
    first = next((rank for rank, flag in enumerate(relevant_flags, start=1) if flag), None)
    covered = [covered_fraction(e, retrieved_spans[:k]) >= COVERAGE for e in evidence]
    return {
        "precision": hits / k,
        "recall": hits / len(expected_set) if expected_set else 0.0,
        "evidence_recall": sum(covered) / len(covered) if covered else 0.0,
        "complete": float(all(covered)) if covered else 0.0,
        "hit": float(hits > 0),
        "mrr": 1.0 / first if first else 0.0,
    }


def mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.fmean(clean), 4) if clean else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Percentil por el método del rango más cercano (q en [0, 100])."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return round(ordered[rank - 1], 4)


def groundedness(claims: Sequence[dict]) -> float | None:
    """Afirmaciones sostenidas / total. None si la respuesta no tiene afirmaciones."""
    if not claims:
        return None
    return sum(1 for claim in claims if claim.get("supported") is True) / len(claims)
