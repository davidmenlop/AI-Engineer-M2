"""Búsqueda léxica BM25 con normalización para español.

La ingesta tokeniza los chunks y guarda los tokens en ``data/index/bm25_<estrategia>.json``.
La consulta lee ese archivo: no vuelve a procesar el corpus (ADR-038).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import snowballstemmer
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[a-z0-9]+")

# Palabras vacías frecuentes en español. Lista corta a propósito: se sacan las que no
# aportan al ranking; los números y siglas (USD, ARS, VPN, MFA) se conservan.
STOPWORDS = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes como con contra cual cuales
    cuando de del desde donde dos e el ella ellas ellos en entre era es esa esas ese eso esos
    esta estan estas este esto estos fue ha hay hasta la las le les lo los mas me mi mis mucho
    muy nada ni no nos o otra otras otro otros para pero poco por porque que quien se sea segun
    ser si sin sobre su sus tambien te tengo tiene tienen tu tus un una uno unos y ya yo
    puedo puede pueden tengo tenemos hacer hago debo debe deben cuanto cuanta cuantos cuantas
    me mi
    """.split()
)


@lru_cache(maxsize=1)
def _stemmer():
    return snowballstemmer.stemmer("spanish")


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """minúsculas, sin tildes, sin palabras vacías, con stemming (licencias -> licenci)."""
    words = [w for w in TOKEN.findall(normalize(text)) if w not in STOPWORDS]
    return _stemmer().stemWords(words)


class BM25Index:
    def __init__(self, ids: Sequence[str], tokens: Sequence[Sequence[str]]) -> None:
        if len(ids) != len(tokens) or not ids:
            raise ValueError("BM25Index necesita la misma cantidad (>0) de ids y documentos")
        self.ids = list(ids)
        self._tokens = [list(t) for t in tokens]
        self._bm25 = BM25Okapi(self._tokens)

    @classmethod
    def from_texts(cls, ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        return cls(ids, [tokenize(text) for text in texts])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ids": self.ids, "tokens": self._tokens}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        if not path.exists():
            raise FileNotFoundError(f"no existe {path}: corré la ingesta primero")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["ids"], payload["tokens"])

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        """Los n chunks con mayor puntaje BM25. Los de puntaje <= 0 no se devuelven:
        no comparten ningún término informativo con la consulta."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.ids, scores, strict=True), key=lambda item: item[1], reverse=True)
        return [(chunk_id, float(score)) for chunk_id, score in ranked[:n] if score > 0]
