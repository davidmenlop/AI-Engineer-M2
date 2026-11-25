"""Juez de groundedness (LLM-as-judge) sobre los chunks que la respuesta citó."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evals.metrics import groundedness
from src.config import ModelConfig
from src.llm import OutputRejected, Usage, call_validated, extract_json
from src.prompts import PromptSpec
from src.providers.base import LLMProvider
from src.schemas import ChunkRelated


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim: str = Field(min_length=1)
    supported: bool
    chunk_id: str | None = None
    reason: str = ""


class Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claims: list[Claim]


@dataclass
class JudgeResult:
    claims: list[dict]
    groundedness: float | None
    attempts: int
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    error: str | None = None


def format_cited(chunks: list[ChunkRelated]) -> str:
    return "\n\n".join(f'<fragmento id="{c.chunk_id}">\n{c.content}\n</fragmento>' for c in chunks)


class GroundednessJudge:
    def __init__(
        self, provider: LLMProvider, model: ModelConfig, prompt: PromptSpec, *, max_retries: int, timeout_s: float
    ) -> None:
        if prompt.kind != "judge":
            raise ValueError("GroundednessJudge necesita un prompt de tipo judge")
        self._provider = provider
        self._model = model
        self._prompt = prompt
        self._max_retries = max_retries
        self._timeout_s = timeout_s

    @property
    def model_id(self) -> str:
        return self._model.id

    def judge(self, question: str, answer: str, cited: list[ChunkRelated]) -> JudgeResult:
        system, user = self._prompt.render(question=question, answer=answer, context=format_cited(cited))

        def validate(text: str) -> Verdict:
            try:
                return Verdict.model_validate(extract_json(text))
            except ValidationError as exc:
                raise OutputRejected(f"el JSON no tiene la forma pedida: {exc.errors()[0]['msg']}") from exc

        call = call_validated(
            self._provider, self._model, system, user, validate,
            max_retries=self._max_retries, timeout_s=self._timeout_s,
        )
        claims = [c.model_dump() for c in call.value.claims] if call.value else []
        return JudgeResult(
            claims=claims,
            groundedness=groundedness(claims) if call.value else None,
            attempts=call.attempts,
            usage=call.usage,
            cost_usd=call.usage.cost(self._model),
            error=call.error,
        )
