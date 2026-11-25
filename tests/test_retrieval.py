"""Tests de BM25, RRF y del retriever híbrido con dobles (sin red ni modelo).

Ejecutar: python tests/test_retrieval.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _runner import run

from src.config import load_rag_config
from src.lexical import BM25Index, normalize, tokenize
from src.retriever import Retriever, rrf_fuse
from src.vector_store import StoredChunk

TEXTS = {
    "a.md_0": "Las vacaciones se piden en el portal de People con 15 días de anticipación.",
    "a.md_1": "Avisá en el canal de Slack #people-avisos antes de las 10 AM si estás enfermo.",
    "b.md_0": "El estipendio de conectividad para home office es de $45,000 ARS por mes.",
}


def test_normalize_removes_accents_and_case() -> None:
    assert normalize("Días Hábiles ÑANDÚ") == "dias habiles nandu"


def test_tokenize_stems_and_drops_stopwords() -> None:
    assert tokenize("las licencias") == tokenize("licencia")
    assert "de" not in tokenize("crédito de capacitación")
    assert "usd" in tokenize("USD 500")


def test_bm25_finds_exact_terms() -> None:
    index = BM25Index.from_texts(list(TEXTS), list(TEXTS.values()))
    assert index.search("¿qué canal es people-avisos?", 3)[0][0] == "a.md_1"
    assert index.search("estipendio", 3)[0][0] == "b.md_0"


def test_bm25_omits_zero_scores() -> None:
    index = BM25Index.from_texts(list(TEXTS), list(TEXTS.values()))
    assert index.search("cochera estacionamiento", 3) == []


def test_bm25_roundtrip() -> None:
    index = BM25Index.from_texts(list(TEXTS), list(TEXTS.values()))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bm25.json"
        index.save(path)
        loaded = BM25Index.load(path)
    assert loaded.search("vacaciones", 1) == index.search("vacaciones", 1)


def test_rrf_math_and_order() -> None:
    fused = dict(rrf_fuse([["x", "y"], ["y", "z"]], k=60))
    assert abs(fused["y"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(fused["x"] - 1 / 61) < 1e-12
    assert list(dict(rrf_fuse([["x", "y"], ["y", "z"]], k=60))) == ["y", "x", "z"]


class FakeStore:
    def __init__(self, ranking: list[tuple[str, float]]) -> None:
        self.ranking = ranking

    def query(self, strategy: str, embedding, n: int):
        return self.ranking[:n]


class FakeEmbedder:
    model_name = "fake"

    def embed_query(self, text: str):
        return [0.0]


class FakeReranker:
    """Puntúa más alto los chunks que contienen la palabra clave."""

    model_name = "fake-reranker"

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    def scores(self, query, documents):
        return [0.9 if self.keyword in doc else 0.1 for doc in documents]


def _retriever(vector_ranking: list[tuple[str, float]], reranker=None) -> Retriever:
    chunks = {
        cid: StoredChunk(cid, text, {"source": cid.split("_")[0], "section": "S"}) for cid, text in TEXTS.items()
    }
    bm25 = BM25Index.from_texts(list(TEXTS), list(TEXTS.values()))
    return Retriever("structural", FakeStore(vector_ranking), FakeEmbedder(), bm25, chunks, load_rag_config(), reranker)


def test_vector_mode_returns_cosine_scores() -> None:
    retriever = _retriever([("b.md_0", 0.8), ("a.md_0", 0.4), ("a.md_1", 0.1)])
    results = retriever.retrieve("internet en casa", top_k=2, mode="vector")
    assert [r.chunk_id for r in results] == ["b.md_0", "a.md_0"]
    assert results[0].score == 0.8 and results[0].bm25_rank is None


def test_hybrid_rescues_lexical_match_missed_by_vector() -> None:
    # La vectorial ubica último al chunk del canal; BM25 lo encuentra primero.
    retriever = _retriever([("b.md_0", 0.8), ("a.md_0", 0.4), ("a.md_1", 0.1)])
    vector_top = [r.chunk_id for r in retriever.retrieve("#people-avisos", top_k=1, mode="vector")]
    hybrid_top = [r.chunk_id for r in retriever.retrieve("#people-avisos", top_k=2, mode="hybrid")]
    assert vector_top == ["b.md_0"]
    assert "a.md_1" in hybrid_top


def test_hybrid_score_is_normalized() -> None:
    retriever = _retriever([("a.md_1", 0.9), ("b.md_0", 0.5), ("a.md_0", 0.1)])
    top = retriever.retrieve("canal people-avisos", top_k=1, mode="hybrid")[0]
    assert top.chunk_id == "a.md_1" and abs(top.score - 1.0) < 1e-9
    assert top.vector_rank == 1 and top.bm25_rank == 1


def test_contract_fields() -> None:
    retriever = _retriever([("a.md_0", 0.9)])
    payload = retriever.retrieve("vacaciones", top_k=1, mode="vector")[0].to_contract()
    assert set(payload) == {"chunk_id", "source", "section", "content", "score"}


def test_rerank_reorders_fused_candidates() -> None:
    ranking = [("b.md_0", 0.8), ("a.md_0", 0.4), ("a.md_1", 0.1)]
    plain = _retriever(ranking).retrieve("home office", top_k=1, mode="vector")
    reranked = _retriever(ranking, FakeReranker("Slack")).retrieve("home office", top_k=1, mode="vector", rerank=True)
    assert plain[0].chunk_id == "b.md_0"
    assert reranked[0].chunk_id == "a.md_1" and reranked[0].score == 0.9


def test_rerank_without_reranker_fails_loudly() -> None:
    try:
        _retriever([("a.md_0", 0.9)]).retrieve("x", top_k=1, mode="vector", rerank=True)
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError")


if __name__ == "__main__":
    raise SystemExit(run(globals()))
