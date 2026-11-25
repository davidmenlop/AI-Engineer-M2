"""Tests de la generación con citas, el rechazo y el reintento (sin red).

Ejecutar: python tests/test_assistant.py
"""

from __future__ import annotations

from _runner import run

from src.assistant import FALLBACK_ANSWER, Assistant
from src.citations import extract_citations, is_refusal, unknown_citations
from src.config import ModelConfig
from src.prompts import PromptError, load_prompt
from src.providers.base import LLMProvider, LLMResponse
from src.providers.mock_provider import MockProvider
from src.retriever import RetrievedChunk

MODEL = ModelConfig(id="fake", temperature=None, max_tokens=500, input_per_1m_usd=1.0, output_per_1m_usd=2.0)
PROMPT = load_prompt("generation_v1", kind="generation")
CHUNKS = [
    RetrievedChunk("hr_policies.md_0", "hr_policies.md", "Licencias", "Matrimonio: 10 días corridos.", 0.9),
    RetrievedChunk("hr_policies.md_1", "hr_policies.md", "Home Office", "Estipendio de $45,000 ARS.", 0.5),
]


class FakeRetriever:
    strategy = "structural"
    last_latency_ms = 1.0

    def retrieve(self, question, top_k=None, mode=None):
        return CHUNKS


class Scripted(LLMProvider):
    name = "scripted"

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.users: list[str] = []

    def complete(self, system, user, *, model, temperature, max_tokens, timeout_s, effort=None):
        self.users.append(user)
        return LLMResponse(self.answers.pop(0), 100, 20, "api")


def _assistant(provider: LLMProvider, max_retries: int = 2) -> Assistant:
    return Assistant(FakeRetriever(), provider, MODEL, PROMPT, max_retries=max_retries, timeout_s=5,
                     mode="hybrid", top_k=2)


def test_extract_citations_variants() -> None:
    text = "A [hr_policies.md_0]. B [it_manual.md_3, finance_processes.md_1]. C [nota] [hr_policies.md_0]"
    assert extract_citations(text) == ["hr_policies.md_0", "it_manual.md_3", "finance_processes.md_1"]
    assert unknown_citations(["a.md_1", "b.md_2"], ["a.md_1"]) == ["b.md_2"]


def test_is_refusal_tolerates_whitespace() -> None:
    assert is_refusal("  " + PROMPT.refusal_text.replace(" ", "\n  "), PROMPT.refusal_text)
    assert not is_refusal("Son 10 días [hr_policies.md_0].", PROMPT.refusal_text)


def test_valid_answer_keeps_only_cited_chunks() -> None:
    result = _assistant(Scripted(["Son 10 días corridos [hr_policies.md_0]."])).answer("¿días por casamiento?")
    assert result.success and not result.refused and result.attempts == 1
    assert result.cited_ids == ["hr_policies.md_0"]
    assert result.response.chunks_related[0].source == "hr_policies.md"
    assert abs(result.cost_usd - (100 * 1.0 + 20 * 2.0) / 1_000_000) < 1e-12


def test_unknown_citation_is_retried_with_feedback() -> None:
    provider = Scripted(["Son 10 días [it_manual.md_9].", "Son 10 días [hr_policies.md_0]."])
    result = _assistant(provider).answer("¿días por casamiento?")
    assert result.attempts == 2 and result.cited_ids == ["hr_policies.md_0"]
    assert "it_manual.md_9" in provider.users[1]


def test_answer_without_citations_is_retried() -> None:
    provider = Scripted(["Son 10 días.", "Son 10 días [hr_policies.md_0]."])
    result = _assistant(provider).answer("¿días por casamiento?")
    assert result.attempts == 2 and result.success


def test_refusal_has_no_chunks() -> None:
    result = _assistant(Scripted([PROMPT.refusal_text])).answer("¿hay cochera?")
    assert result.refused and result.response.chunks_related == []
    assert result.response.system_answer == PROMPT.refusal_text


def test_fallback_after_exhausting_retries() -> None:
    result = _assistant(Scripted(["sin citas"] * 3), max_retries=2).answer("¿días?")
    assert not result.success and result.attempts == 3
    assert result.response.system_answer == FALLBACK_ANSWER and result.response.chunks_related == []


def test_contract_shape() -> None:
    result = _assistant(Scripted(["Son 10 días [hr_policies.md_0]."])).answer("  ¿días?  ")
    payload = result.response.model_dump()
    assert set(payload) == {"user_question", "system_answer", "chunks_related"}
    assert payload["user_question"] == "¿días?"
    assert set(payload["chunks_related"][0]) == {"chunk_id", "source", "section", "content", "score"}


def test_empty_question_is_rejected() -> None:
    try:
        _assistant(Scripted([])).answer("   ")
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError")


def test_mock_provider_cites_and_refuses() -> None:
    answered = _assistant(MockProvider()).answer("¿Cuántos días de matrimonio?")
    assert answered.cited_ids == ["hr_policies.md_0"] and not answered.refused
    refused = _assistant(MockProvider()).answer("¿cochera?")
    assert refused.refused


def test_prompts_load_and_validate() -> None:
    judge = load_prompt("judge_groundedness_v1", kind="judge")
    assert judge.kind == "judge" and len(PROMPT.sha256) == 64
    try:
        load_prompt("judge_groundedness_v1", kind="generation")
    except PromptError:
        return
    raise AssertionError("se esperaba PromptError por tipo incorrecto")


if __name__ == "__main__":
    raise SystemExit(run(globals()))
