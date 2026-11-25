"""Re-ranking con cross-encoder multilingüe (fastembed/ONNX, local).

Un cross-encoder lee la pregunta y el chunk *juntos* y devuelve un puntaje de
relevancia. Es más preciso que comparar embeddings calculados por separado, pero más
lento: por eso solo reordena los candidatos que ya trajo la búsqueda híbrida (ADR-050).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: Path) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # import diferido

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name, cache_dir=str(cache_dir))

    def scores(self, query: str, documents: Sequence[str]) -> list[float]:
        """Probabilidad de relevancia en [0, 1] (sigmoide del logit del modelo)."""
        if not documents:
            return []
        return [sigmoid(float(s)) for s in self._model.rerank(query, list(documents))]
