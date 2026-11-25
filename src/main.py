"""CLI de Nubbix Assist: una pregunta -> JSON {user_question, system_answer, chunks_related}.

    python -m src.main --query "¿Con cuánta anticipación pido las vacaciones?" --pretty
    python -m src.main --query "..." --provider anthropic
    python -m src.main --query "..." --strategy fixed --mode vector --top-k 5
    python -m src.main --query "..." --retrieve-only     # solo recuperación, sin LLM
    python -m src.main --query "..." --rerank            # + cross-encoder (extra credit)
    echo "pregunta" | python -m src.main --query -

stdout: solo el JSON del contrato. stderr: logs y una línea de resumen.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.assistant import Assistant
from src.config import (
    DEFAULT_MODEL_KEY,
    MODES,
    STRATEGIES,
    ConfigError,
    load_model_config,
    load_rag_config,
    load_settings,
)
from src.observability.metrics import append_metric, build_record
from src.prompts import PromptError, load_prompt
from src.providers.base import ProviderFatalError, ProviderUnavailable
from src.providers.factory import build_provider
from src.retriever import Retriever
from src.vector_store import IndexNotReady

logger = logging.getLogger("assist")
DEFAULT_PROMPT = "generation_v1"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nubbix Assist: preguntas sobre la documentación interna.")
    parser.add_argument("--query", required=True, help="la pregunta; '-' la lee de stdin")
    parser.add_argument("--strategy", choices=STRATEGIES, help="colección a consultar (default: rag.yaml)")
    parser.add_argument("--mode", choices=MODES, help="búsqueda: hybrid, vector o bm25 (default: rag.yaml)")
    parser.add_argument("--top-k", type=int, help="chunks que se pasan al modelo (default: rag.yaml)")
    parser.add_argument("--provider", help="mock, anthropic u openai (default: DEFAULT_PROVIDER)")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, help="clave de modelo en providers.yaml")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt de generación en prompts/")
    parser.add_argument("--rerank", action="store_true", help="reordenar con el cross-encoder (extra credit)")
    parser.add_argument("--retrieve-only", action="store_true", help="devolver los chunks sin llamar al LLM")
    parser.add_argument("--pretty", action="store_true", help="JSON indentado")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        settings = load_settings()
        rag = load_rag_config()
    except ConfigError as exc:
        print(f"error de configuración: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("httpx", "httpx2", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    question = sys.stdin.read() if args.query == "-" else args.query
    if not question.strip():
        print("error: la pregunta está vacía", file=sys.stderr)
        return 2
    strategy = args.strategy or rag.retrieval.default_strategy
    mode = args.mode or rag.retrieval.default_mode
    top_k = args.top_k or rag.retrieval.top_k
    indent = 2 if args.pretty else None

    try:
        rerank = args.rerank or rag.rerank.enabled
        retriever = Retriever.open(strategy, settings, rag, with_reranker=rerank)
        if args.retrieve_only:
            chunks = retriever.retrieve(question, top_k=top_k, mode=mode, rerank=rerank)
            payload = {"user_question": question.strip(), "chunks_related": [c.to_contract() for c in chunks]}
            print(json.dumps(payload, ensure_ascii=False, indent=indent))
            return 0
        provider_name = args.provider or settings.default_provider
        provider = build_provider(provider_name)
        model = load_model_config(provider_name, args.model)
        prompt = load_prompt(args.prompt, kind="generation")
        assistant = Assistant(
            retriever, provider, model, prompt,
            max_retries=settings.max_retries, timeout_s=settings.request_timeout_s,
            mode=mode, top_k=top_k, rerank=rerank,
        )
        result = assistant.answer(question)
    except (ConfigError, PromptError, IndexNotReady, ProviderUnavailable, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ProviderFatalError as exc:
        print(f"error: el proveedor rechazó la solicitud: {exc}", file=sys.stderr)
        return 2

    append_metric(settings.metrics_path, build_record(result))
    logger.info(
        "request=%s %s/%s k=%d provider=%s model=%s attempts=%d refused=%s cited=%s cost_usd=%.8f "
        "retrieval_ms=%.1f latency_s=%.3f",
        result.request_id[:8], result.strategy, result.mode, result.top_k, result.provider, result.model,
        result.attempts, result.refused, result.cited_ids, result.cost_usd, result.retrieval_ms, result.latency_s,
    )
    print(json.dumps(result.response.model_dump(), ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
