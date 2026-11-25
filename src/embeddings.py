"""Embeddings locales con fastembed (ONNX, sin claves ni red después de la descarga)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class Embedder:
    """Envuelve el modelo de embeddings. El mismo objeto sirve para indexar y para consultar.

    El modelo se descarga de Hugging Face en la primera ejecución (~220 MB) y queda
    cacheado en ``cache_dir`` (``data/models/``, fuera de Git).
    """

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        from fastembed import TextEmbedding  # import diferido: tarda ~1 s

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
