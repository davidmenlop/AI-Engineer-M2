"""Índice vectorial en Chroma local persistente: una colección por estrategia."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class IndexNotReady(RuntimeError):
    """El índice no existe o no corresponde a la configuración actual."""


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


class VectorStore:
    def __init__(self, index_dir: Path, collection_prefix: str = "rag") -> None:
        import chromadb  # import diferido: tarda ~1 s
        from chromadb.config import Settings as ChromaSettings

        index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(index_dir / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._prefix = collection_prefix

    def collection_name(self, strategy: str) -> str:
        return f"{self._prefix}_{strategy}"

    def recreate(self, strategy: str, distance: str, metadata: dict[str, Any]):
        """Borra y crea la colección de la estrategia: la ingesta siempre parte de cero."""
        name = self.collection_name(strategy)
        if name in {c.name for c in self._client.list_collections()}:
            self._client.delete_collection(name)
        return self._client.create_collection(
            name=name,
            configuration={"hnsw": {"space": distance}},
            metadata=metadata,
        )

    def _collection(self, strategy: str):
        name = self.collection_name(strategy)
        try:
            return self._client.get_collection(name)
        except Exception as exc:  # chromadb lanza NotFoundError / ValueError según versión
            raise IndexNotReady(
                f"no existe la colección '{name}'. Corré: python -m ingest.ingest --strategy {strategy}"
            ) from exc

    def add(
        self,
        strategy: str,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        self._collection(strategy).add(
            ids=list(ids), documents=list(texts), embeddings=[list(e) for e in embeddings],
            metadatas=list(metadatas),
        )

    def info(self, strategy: str) -> dict[str, Any]:
        collection = self._collection(strategy)
        return {"count": collection.count(), **(collection.metadata or {})}

    def all_chunks(self, strategy: str) -> dict[str, StoredChunk]:
        result = self._collection(strategy).get(include=["documents", "metadatas"])
        return {
            chunk_id: StoredChunk(chunk_id, text, dict(metadata or {}))
            for chunk_id, text, metadata in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        }

    def query(self, strategy: str, embedding: Sequence[float], n: int) -> list[tuple[str, float]]:
        """(chunk_id, similitud coseno = 1 - distancia), de mayor a menor."""
        collection = self._collection(strategy)
        n = min(n, collection.count())
        result = collection.query(query_embeddings=[list(embedding)], n_results=n, include=["distances"])
        return [
            (chunk_id, 1.0 - float(distance))
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]
