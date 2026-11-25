"""Proveedor determinista para probar el pipeline sin claves ni red.

No es un LLM: arma una respuesta extractiva con la oración del primer fragmento que
más palabras comparte con la pregunta, y la cita. Si ningún fragmento comparte
palabras informativas, devuelve el texto de rechazo del prompt. Sirve para verificar
la instalación y el contrato; no se usa para medir calidad de generación.
"""

from __future__ import annotations

import math
import re

from src.lexical import tokenize
from src.providers.base import LLMProvider, LLMResponse

QUESTION = re.compile(r"<pregunta>\s*(.*?)\s*</pregunta>", re.DOTALL)
FRAGMENT = re.compile(r'<fragmento id="([^"]+)"[^>]*>\s*(.*?)\s*</fragmento>', re.DOTALL)
REFUSAL = re.compile(r"respondé exactamente con este texto y\s+nada más, sin citas:\s*\n\s*(.+?)\n", re.DOTALL)
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class MockProvider(LLMProvider):
    name = "mock"

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float | None,
        max_tokens: int,
        timeout_s: float,
        effort: str | None = None,
    ) -> LLMResponse:
        text = self._answer(system, user)
        return LLMResponse(
            text=text,
            prompt_tokens=math.ceil(len(system + user) / 4),
            completion_tokens=math.ceil(len(text) / 4),
            token_source="estimated",
        )

    @staticmethod
    def _answer(system: str, user: str) -> str:
        match = QUESTION.search(user)
        question = set(tokenize(match.group(1) if match else user))
        refusal = REFUSAL.search(system)
        refusal_text = refusal.group(1).strip() if refusal else "No encontré esa información."

        best: tuple[int, str, str] | None = None
        for chunk_id, content in FRAGMENT.findall(user):
            for sentence in SENTENCE.split(content):
                sentence = sentence.strip(" -")
                if not sentence or sentence.startswith("#"):
                    continue
                overlap = len(question & set(tokenize(sentence)))
                if overlap and (best is None or overlap > best[0]):
                    best = (overlap, sentence, chunk_id)
        if best is None:
            return refusal_text
        _, sentence, chunk_id = best
        sentence = sentence.rstrip(".")
        return f"Según la documentación, {sentence[0].lower()}{sentence[1:]} [{chunk_id}]."
