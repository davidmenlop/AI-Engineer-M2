"""Tests del golden dataset, el etiquetado por solapamiento y las métricas (sin red).

Ejecutar: python tests/test_evals.py
"""

from __future__ import annotations

from _runner import run

from evals.dataset import DatasetError, Evidence, load_dataset, locate, read_rows, stale_cases, parse_rows
from evals.judge import Verdict
from evals.metrics import Span, covered_fraction, groundedness, percentile, retrieval_scores
from evals.runner import diagnose
from ingest.chunking import Chunk
from ingest.loader import load_corpus
from src.config import load_rag_config


def _chunk(chunk_id: str, start: int, end: int, source: str = "a.md") -> Chunk:
    return Chunk(chunk_id, source, "S", "fixed", "x" * (end - start), start, end)


def test_dataset_is_valid_and_labels_are_fresh() -> None:
    cases = load_dataset()
    assert len(cases) >= 25, "la consigna pide al menos 25 preguntas"
    assert sum(not c.in_scope for c in cases) >= 1, "el bonus pide preguntas fuera de alcance"
    assert {"sinonimos", "cruza_documentos", "distractor"} <= {c.type for c in cases}
    rows = read_rows()
    assert stale_cases(rows, parse_rows(rows)) == []


def test_every_in_scope_case_has_expected_chunks_in_both_strategies() -> None:
    for case in load_dataset():
        if case.in_scope:
            assert case.expected_chunks["fixed"] and case.expected_chunks["structural"], case.id


def test_cross_document_cases_cite_two_documents() -> None:
    for case in load_dataset():
        if case.type == "cruza_documentos":
            assert len({e.source for e in case.evidence}) >= 2, case.id


def test_locate_rejects_missing_quote() -> None:
    documents = {d.source: d for d in load_corpus(load_rag_config().corpus_dir)}
    try:
        locate(documents, "hr_policies.md", "una frase que no existe")
    except DatasetError:
        return
    raise AssertionError("se esperaba DatasetError")


def test_relevance_rule_uses_overlap() -> None:
    evidence = Evidence("a.md", "q" * 100, 100, 200)
    assert evidence.is_relevant(_chunk("a.md_0", 0, 130))        # 30 caracteres de solapamiento
    assert not evidence.is_relevant(_chunk("a.md_1", 0, 129))    # 29
    assert not evidence.is_relevant(_chunk("b.md_0", 100, 200, "b.md"))
    short = Evidence("a.md", "q" * 10, 100, 110)
    assert short.is_relevant(_chunk("a.md_2", 95, 115))          # evidencia corta: basta cubrirla


def test_covered_fraction_uses_union_of_spans() -> None:
    evidence = Span("a.md", 100, 200)
    assert covered_fraction(evidence, [Span("a.md", 100, 150), Span("a.md", 140, 190)]) == 0.9
    assert covered_fraction(evidence, [Span("b.md", 100, 200)]) == 0.0


def test_retrieval_scores() -> None:
    evidence = [Span("a.md", 0, 50), Span("b.md", 0, 50)]
    scores = retrieval_scores(
        retrieved=["x", "a1", "y"],
        expected=["a1", "b1"],
        evidence=evidence,
        retrieved_spans=[Span("c.md", 0, 50), Span("a.md", 0, 60), Span("c.md", 0, 9)],
        k=3,
    )
    assert abs(scores["precision"] - 1 / 3) < 1e-9
    assert scores["recall"] == 0.5 and scores["evidence_recall"] == 0.5
    assert scores["complete"] == 0.0 and scores["hit"] == 1.0 and scores["mrr"] == 0.5


def test_groundedness_and_percentile() -> None:
    assert groundedness([{"supported": True}, {"supported": False}]) == 0.5
    assert groundedness([]) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0 and percentile([], 95) is None


def test_judge_verdict_schema() -> None:
    verdict = Verdict.model_validate({"claims": [{"claim": "x", "supported": True, "chunk_id": None, "extra": 1}]})
    assert verdict.claims[0].supported is True


def test_diagnose_separates_stages() -> None:
    case = load_dataset()[0]
    assert diagnose(case, 1.0, False, 1.0, True) == "ok"
    assert diagnose(case, 0.0, False, 1.0, True) == "retrieval"
    assert diagnose(case, 1.0, True, None, True) == "generacion"
    assert diagnose(case, 0.0, False, 0.5, True) == "ambas"


if __name__ == "__main__":
    raise SystemExit(run(globals()))
