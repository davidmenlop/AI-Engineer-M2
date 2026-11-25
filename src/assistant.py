"""Orquestación de una consulta: recuperar -> armar contexto -> generar -> validar citas.

Validación de la respuesta del modelo (si falla, se reintenta con el motivo):
- no vacía;
- toda cita [id] corresponde a un chunk recuperado en esta consulta;
- si no es el rechazo fijo, tiene al menos una cita.

``chunks_related`` contiene solo los chunks citados, en el orden en que se
recuperaron: los recuperados que el modelo no usó no se presentan como fuente.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from src.citations import extract_citations, is_refusal, normalize_text, unknown_citations
from src.config import ModelConfig
from src.llm import OutputRejected, Usage, call_validated
from src.prompts import PromptSpec
from src.providers.base import LLMProvider
from src.retriever import RetrievedChunk, Retriever
from src.schemas import AssistResponse, ChunkRelated

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = (
    "No pude generar una respuesta verificable con la documentación disponible. "
    "Probá reformular la pregunta o consultá con People, IT o Finance."
)


@dataclass(frozen=True)
class _Generated:
    text: str
    cited: list[str]
    refused: bool


@dataclass(frozen=True)
class AssistResult:
    request_id: str
    response: AssistResponse
    retrieved: list[RetrievedChunk]
    refused: bool
    success: bool
    strategy: str
    mode: str
    top_k: int
    provider: str
    model: str
    prompt_version: str
    attempts: int
    usage: Usage
    cost_usd: float
    retrieval_ms: float
    latency_s: float
    error: str | None = field(default=None)

    @property
    def cited_ids(self) -> list[str]:
        return [chunk.chunk_id for chunk in self.response.chunks_related]


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f'<fragmento id="{c.chunk_id}" documento="{c.source}" seccion="{c.section}">\n{c.content}\n</fragmento>'
        for c in chunks
    ]
    return "\n\n".join(blocks)


class Assistant:
    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider,
        model: ModelConfig,
        prompt: PromptSpec,
        *,
        max_retries: int,
        timeout_s: float,
        mode: str,
        top_k: int,
        rerank: bool = False,
    ) -> None:
        if prompt.kind != "generation":
            raise ValueError("Assistant necesita un prompt de tipo generation")
        self._retriever = retriever
        self._provider = provider
        self._model = model
        self._prompt = prompt
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._mode = mode
        self._top_k = top_k
        self._rerank = rerank

    def answer(self, question: str) -> AssistResult:
        question = question.strip()
        if not question:
            raise ValueError("la pregunta está vacía")
        request_id = uuid.uuid4().hex
        started = time.perf_counter()

        retrieved = self._retriever.retrieve(question, top_k=self._top_k, mode=self._mode, rerank=self._rerank)
        retrieval_ms = self._retriever.last_latency_ms
        allowed = [c.chunk_id for c in retrieved]
        system, user = self._prompt.render(question=question, context=format_context(retrieved))

        def validate(text: str) -> _Generated:
            answer = normalize_text(text)
            if not answer:
                raise OutputRejected("la respuesta está vacía")
            if is_refusal(answer, self._prompt.refusal_text):
                return _Generated(self._prompt.refusal_text, [], True)
            cited = extract_citations(answer)
            unknown = unknown_citations(cited, allowed)
            if unknown:
                raise OutputRejected(
                    f"citaste ids que no están entre los fragmentos: {unknown}. Ids válidos: {allowed}"
                )
            if not cited:
                raise OutputRejected(
                    "la respuesta no tiene citas. Citá el id de cada fragmento usado entre corchetes, "
                    "o devolvé el texto de rechazo si los fragmentos no contienen la respuesta"
                )
            return _Generated(answer, cited, False)

        call = call_validated(
            self._provider, self._model, system, user, validate,
            max_retries=self._max_retries, timeout_s=self._timeout_s,
        )
        if call.value is None:
            logger.error("sin respuesta válida tras %d intentos: %s", call.attempts, call.error)
            generated = _Generated(FALLBACK_ANSWER, [], False)
        else:
            generated = call.value

        cited = set(generated.cited)
        related = [ChunkRelated(**c.to_contract()) for c in retrieved if c.chunk_id in cited]
        response = AssistResponse(
            user_question=question, system_answer=generated.text, chunks_related=related
        )
        return AssistResult(
            request_id=request_id,
            response=response,
            retrieved=retrieved,
            refused=generated.refused,
            success=call.success,
            strategy=self._retriever.strategy,
            mode=self._mode + ("+rerank" if self._rerank else ""),
            top_k=len(retrieved),
            provider=self._provider.name,
            model=self._model.id,
            prompt_version=self._prompt.version,
            attempts=call.attempts,
            usage=call.usage,
            cost_usd=call.usage.cost(self._model),
            retrieval_ms=retrieval_ms,
            latency_s=time.perf_counter() - started,
            error=call.error,
        )
