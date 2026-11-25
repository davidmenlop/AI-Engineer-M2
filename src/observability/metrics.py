"""Registro append-only de métricas por consulta (una línea JSON por consulta).

La pregunta no se guarda en texto plano: solo su sha256 y su longitud (heredado de M1).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.assistant import AssistResult


def build_record(result: AssistResult) -> dict[str, Any]:
    question = result.response.user_question
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "request_id": result.request_id,
        "strategy": result.strategy,
        "mode": result.mode,
        "top_k": result.top_k,
        "provider": result.provider,
        "model": result.model,
        "prompt_version": result.prompt_version,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "question_chars": len(question),
        "retrieved_ids": [c.chunk_id for c in result.retrieved],
        "cited_ids": result.cited_ids,
        "refused": result.refused,
        "success": result.success,
        "attempts": result.attempts,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "token_source": result.usage.token_source,
        "cost_usd": result.cost_usd,
        "retrieval_ms": round(result.retrieval_ms, 2),
        "latency_s": round(result.latency_s, 4),
        "error": result.error,
    }


def append_metric(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
