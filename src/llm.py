"""Llamada a un LLM con validación de la salida y reintento con el error (heredado de M1).

La política de reintentos vive en un solo lugar: los SDK se crean con max_retries=0
(ADR-015 de M1). Un error transitorio del proveedor o una salida que no pasa la
validación se reintentan hasta ``max_retries`` veces; en el segundo caso, el modelo
recibe el motivo del rechazo para corregirse.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from src.config import ModelConfig
from src.observability.pricing import compute_cost
from src.providers.base import LLMProvider, LLMResponse, ProviderError

logger = logging.getLogger(__name__)
T = TypeVar("T")
MAX_FEEDBACK_CHARS = 500


class OutputRejected(ValueError):
    """La salida del modelo no cumple lo pedido. El mensaje se le devuelve al modelo."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    estimated: bool = False

    def add(self, response: LLMResponse) -> None:
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.calls += 1
        self.estimated = self.estimated or response.token_source == "estimated"

    @property
    def token_source(self) -> str:
        if self.calls == 0:
            return "none"
        return "estimated" if self.estimated else "api"

    def cost(self, model: ModelConfig) -> float:
        return compute_cost(self.prompt_tokens, self.completion_tokens, model.input_per_1m_usd, model.output_per_1m_usd)


@dataclass
class CallResult(Generic[T]):
    value: T | None
    attempts: int
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.value is not None


def _repair_message(base_user: str, error: str) -> str:
    return (
        f"{base_user}\n\n"
        f"Tu respuesta anterior no cumplió las reglas: {error}\n"
        "Respondé de nuevo corrigiendo eso."
    )


def call_validated(
    provider: LLMProvider,
    model: ModelConfig,
    system: str,
    user: str,
    validate: Callable[[str], T],
    *,
    max_retries: int,
    timeout_s: float,
) -> CallResult[T]:
    usage = Usage()
    last_error: str | None = None
    attempts = 0
    for _ in range(max_retries + 1):
        attempts += 1
        message = user if last_error is None else _repair_message(user, last_error)
        try:
            response = provider.complete(
                system,
                message,
                model=model.id,
                temperature=model.temperature,
                max_tokens=model.max_tokens,
                timeout_s=timeout_s,
                effort=model.effort,
            )
        except ProviderError as exc:
            last_error = f"error del proveedor: {exc}"[:MAX_FEEDBACK_CHARS]
            logger.warning("intento %d falló: %s", attempts, last_error)
            continue
        usage.add(response)
        try:
            return CallResult(value=validate(response.text), attempts=attempts, usage=usage)
        except OutputRejected as exc:
            last_error = str(exc)[:MAX_FEEDBACK_CHARS]
            logger.warning("intento %d rechazado: %s", attempts, last_error)
    return CallResult(value=None, attempts=attempts, usage=usage, error=last_error)


def extract_json(text: str) -> dict:
    """Primer objeto JSON del texto, tolerando cercos ``` y texto alrededor."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
    start, end = stripped.find("{"), stripped.rfind("}")
    for candidate in (stripped, stripped[start : end + 1] if start != -1 and end > start else ""):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise OutputRejected("no se encontró un objeto JSON válido; devolvé solo el JSON pedido")
