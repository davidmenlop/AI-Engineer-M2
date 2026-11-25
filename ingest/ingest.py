"""Ingesta de cero al índice completo, por estrategia de chunking.

    python -m ingest.ingest --strategy structural
    python -m ingest.ingest --strategy fixed
    python -m ingest.ingest --strategy all

Pasos: cargar el corpus (verificando SHA256SUMS) -> chunking -> embeddings ->
colección de Chroma recreada desde cero -> tokens BM25 -> export legible de los chunks
en data/chunks/<estrategia>.jsonl (versionado, para revisarlos sin correr nada).

La consulta nunca indexa: lee lo que deja este comando.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ingest.chunking import Chunk, chunk_corpus
from ingest.loader import CorpusError, corpus_sha256, load_corpus
from src.config import REPO_ROOT, STRATEGIES, ConfigError, load_rag_config, load_settings
from src.embeddings import Embedder
from src.lexical import BM25Index
from src.retriever import bm25_path
from src.vector_store import VectorStore

CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
logger = logging.getLogger("ingest")


def export_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            record = {"chunk_id": chunk.chunk_id, **chunk.metadata(), "chars": len(chunk.text), "text": chunk.text}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def ingest_strategy(strategy: str, embedder: Embedder | None = None) -> dict:
    settings = load_settings()
    rag = load_rag_config()
    started = time.perf_counter()

    documents = load_corpus(rag.corpus_dir)
    params = rag.chunking_params(strategy)
    chunks = chunk_corpus(documents, strategy, params)
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise RuntimeError("chunk_id duplicados: revisar la estrategia de chunking")

    if embedder is None:
        embedder = Embedder(rag.embedding_model, settings.models_dir)
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    if embeddings and len(embeddings[0]) != rag.embedding_dimensions:
        raise ConfigError(
            f"el modelo devolvió {len(embeddings[0])} dimensiones y rag.yaml declara "
            f"{rag.embedding_dimensions}"
        )

    store = VectorStore(settings.index_dir, rag.collection_prefix)
    collection_meta = {
        "strategy": strategy,
        "embedding_model": rag.embedding_model,
        "dimensions": rag.embedding_dimensions,
        "distance": rag.distance,
        "chunking_sha256": rag.chunking_sha256(strategy),
        "chunking_params": json.dumps(params, sort_keys=True),
        "corpus_sha256": corpus_sha256(documents),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    store.recreate(strategy, rag.distance, collection_meta)
    store.add(strategy, ids, texts, embeddings, [c.metadata() for c in chunks])

    BM25Index.from_texts(ids, texts).save(bm25_path(settings, strategy))
    export_chunks(chunks, CHUNKS_DIR / f"{strategy}.jsonl")

    sizes = [len(t) for t in texts]
    return {
        "strategy": strategy,
        "documents": len(documents),
        "chunks": len(chunks),
        "chars_min": min(sizes),
        "chars_avg": round(sum(sizes) / len(sizes)),
        "chars_max": max(sizes),
        "seconds": round(time.perf_counter() - started, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Construye el índice de una estrategia de chunking.")
    parser.add_argument("--strategy", required=True, choices=[*STRATEGIES, "all"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    try:
        rag = load_rag_config()
        embedder = Embedder(rag.embedding_model, load_settings().models_dir)
        for strategy in strategies:
            summary = ingest_strategy(strategy, embedder)
            logger.info(
                "%s: %d documentos -> %d chunks (caracteres min/prom/max %d/%d/%d) en %.2f s",
                summary["strategy"], summary["documents"], summary["chunks"],
                summary["chars_min"], summary["chars_avg"], summary["chars_max"], summary["seconds"],
            )
    except (ConfigError, CorpusError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    logger.info("índice persistido en %s", load_settings().index_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
