"""Recuperación híbrida: vectorial (Chroma) + léxica (BM25), fusionadas con RRF.

Reciprocal Rank Fusion (Cormack, Clarke y Büttcher, 2009): cada buscador aporta su
ranking de ``candidates`` chunks y cada chunk suma ``1 / (rrf_k + rango)`` por cada
ranking en el que aparece. Usa posiciones, no puntajes, así que no hace falta
calibrar la similitud coseno contra BM25, que viven en escalas distintas (ADR-039).

El ``score`` que se devuelve depende del modo:
- hybrid: RRF normalizado a [0, 1] (1 = primero en ambos rankings);
- vector: similitud coseno;
- bm25: puntaje BM25 crudo;
- con --rerank: probabilidad de relevancia del cross-encoder, en [0, 1] (ADR-050).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.config import MODES, RagConfig, Settings
from src.embeddings import Embedder
from src.lexical import BM25Index
from src.reranker import CrossEncoderReranker
from src.vector_store import IndexNotReady, StoredChunk, VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source: str
    section: str
    content: str
    score: float
    vector_rank: int | None = None   # posición (1 = primero) en el ranking vectorial
    bm25_rank: int | None = None     # posición en el ranking BM25

    def to_contract(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "section": self.section,
            "content": self.content,
            "score": round(self.score, 4),
        }


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int) -> list[tuple[str, float]]:
    """Fusiona rankings (listas de ids, mejor primero). Empates: orden de primera aparición."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def bm25_path(settings: Settings, strategy: str):
    return settings.index_dir / f"bm25_{strategy}.json"


class Retriever:
    def __init__(
        self,
        strategy: str,
        store: VectorStore,
        embedder: Embedder,
        bm25: BM25Index,
        chunks: dict[str, StoredChunk],
        rag: RagConfig,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.strategy = strategy
        self._reranker = reranker
        self._store = store
        self._embedder = embedder
        self._bm25 = bm25
        self._chunks = chunks
        self._rag = rag
        self.last_latency_ms: float = 0.0
        self.last_rerank_ms: float = 0.0

    @classmethod
    def open(
        cls,
        strategy: str,
        settings: Settings,
        rag: RagConfig,
        embedder: Embedder | None = None,
        reranker: CrossEncoderReranker | None = None,
        with_reranker: bool = False,
    ) -> Retriever:
        """Abre un índice ya construido. Nunca indexa: si falta o está desactualizado, falla."""
        store = VectorStore(settings.index_dir, rag.collection_prefix)
        info = store.info(strategy)
        if info.get("embedding_model") != rag.embedding_model:
            raise IndexNotReady(
                f"el índice '{strategy}' se construyó con {info.get('embedding_model')!r} y la "
                f"configuración pide {rag.embedding_model!r}: reindexá con "
                f"python -m ingest.ingest --strategy {strategy}"
            )
        if info.get("chunking_sha256") != rag.chunking_sha256(strategy):
            raise IndexNotReady(
                f"los parámetros de chunking de '{strategy}' cambiaron desde la ingesta: "
                f"reindexá con python -m ingest.ingest --strategy {strategy}"
            )
        if embedder is None:
            embedder = Embedder(rag.embedding_model, settings.models_dir)
        elif embedder.model_name != rag.embedding_model:
            raise IndexNotReady("el embedder recibido no es el modelo con el que se indexó")
        bm25 = BM25Index.load(bm25_path(settings, strategy))
        chunks = store.all_chunks(strategy)
        if set(bm25.ids) != set(chunks):
            raise IndexNotReady(f"el índice BM25 y Chroma de '{strategy}' no coinciden: reindexá")
        if reranker is None and (with_reranker or rag.rerank.enabled):
            reranker = CrossEncoderReranker(rag.rerank.model, settings.models_dir)
        return cls(strategy, store, embedder, bm25, chunks, rag, reranker)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def chunk(self, chunk_id: str) -> StoredChunk:
        return self._chunks[chunk_id]

    def retrieve(
        self, query: str, top_k: int | None = None, mode: str | None = None, rerank: bool | None = None
    ) -> list[RetrievedChunk]:
        params = self._rag.retrieval
        rerank = self._rag.rerank.enabled if rerank is None else rerank
        if rerank and self._reranker is None:
            raise ValueError("se pidió re-ranking pero el retriever no tiene reranker (Retriever.open(..., with_reranker=True))")
        top_k = top_k or params.top_k
        mode = mode or params.default_mode
        if mode not in MODES:
            raise ValueError(f"modo desconocido '{mode}'. Opciones: {MODES}")
        started = time.perf_counter()
        candidates = max(params.candidates, top_k)

        vector_hits: list[tuple[str, float]] = []
        bm25_hits: list[tuple[str, float]] = []
        if mode in ("hybrid", "vector"):
            embedding = self._embedder.embed_query(query)
            vector_hits = self._store.query(self.strategy, embedding, candidates)
        if mode in ("hybrid", "bm25"):
            bm25_hits = self._bm25.search(query, candidates)

        vector_rank = {cid: rank for rank, (cid, _) in enumerate(vector_hits, start=1)}
        bm25_rank = {cid: rank for rank, (cid, _) in enumerate(bm25_hits, start=1)}

        if mode == "vector":
            ranked = vector_hits
        elif mode == "bm25":
            ranked = bm25_hits
        else:
            fused = rrf_fuse([[c for c, _ in vector_hits], [c for c, _ in bm25_hits]], params.rrf_k)
            best_possible = 2.0 / (params.rrf_k + 1)
            ranked = [(cid, score / best_possible) for cid, score in fused]

        self.last_rerank_ms = 0.0
        if rerank:
            ranked = self._rerank(query, ranked[: max(self._rag.rerank.candidates, top_k)])
        results = [self._build(cid, score, vector_rank.get(cid), bm25_rank.get(cid)) for cid, score in ranked[:top_k]]
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        return results

    def _rerank(self, query: str, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Reordena los candidatos por el puntaje del cross-encoder (empates: orden previo)."""
        started = time.perf_counter()
        ids = [cid for cid, _ in ranked]
        scores = self._reranker.scores(query, [self._chunks[cid].text for cid in ids])
        self.last_rerank_ms = (time.perf_counter() - started) * 1000
        return sorted(zip(ids, scores, strict=True), key=lambda item: item[1], reverse=True)

    def _build(self, chunk_id: str, score: float, v_rank: int | None, b_rank: int | None) -> RetrievedChunk:
        stored = self._chunks[chunk_id]
        return RetrievedChunk(
            chunk_id=chunk_id,
            source=str(stored.metadata.get("source", "")),
            section=str(stored.metadata.get("section", "")),
            content=stored.text,
            score=score,
            vector_rank=v_rank,
            bm25_rank=b_rank,
        )
