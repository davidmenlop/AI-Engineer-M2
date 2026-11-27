"""Tests de las estrategias de chunking (sin red). Ejecutar: python tests/test_chunking.py"""

from __future__ import annotations

from _runner import run

from ingest.chunking import chunk_corpus, chunk_fixed, chunk_structural
from ingest.loader import CorpusError, Document, load_corpus
from src.config import load_rag_config

DOC = Document(
    source="demo.md",
    sha256="x",
    text=(
        "# Título Demo\n\n"
        "## Primera sección\n"
        "Uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce.\n"
        "- Ítem de lista con un dato: 15 días.\n\n"
        "## Segunda sección\n"
        "Quince dieciséis diecisiete dieciocho diecinueve veinte veintiuno veintidós."
    ),
)


def test_fixed_respects_size_and_does_not_cut_words() -> None:
    chunks = chunk_fixed(DOC, size_chars=60, overlap_chars=15)
    assert len(chunks) > 3
    words = set(DOC.text.split())
    for chunk in chunks:
        assert len(chunk.text) <= 60
        assert chunk.text.split()[0] in words and chunk.text.split()[-1] in words, chunk.text


def test_fixed_spans_match_original_text_and_overlap() -> None:
    chunks = chunk_fixed(DOC, size_chars=60, overlap_chars=15)
    for chunk in chunks:
        assert DOC.text[chunk.char_start : chunk.char_end] == chunk.text
    for previous, current in zip(chunks, chunks[1:]):
        assert current.char_start < previous.char_end, "debe haber solapamiento"
    assert chunks[-1].char_end == len(DOC.text.rstrip())


def test_fixed_section_is_where_chunk_starts() -> None:
    chunks = chunk_fixed(DOC, size_chars=60, overlap_chars=15)
    assert chunks[0].section == "Título Demo"
    assert chunks[-1].section == "Segunda sección"


def test_fixed_rejects_overlap_not_smaller_than_size() -> None:
    try:
        chunk_fixed(DOC, size_chars=50, overlap_chars=50)
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError")


def test_structural_one_chunk_per_section_with_headers() -> None:
    chunks = chunk_structural(DOC, max_chars=1000)
    assert [c.section for c in chunks] == ["Primera sección", "Segunda sección"]
    assert chunks[0].text.startswith("# Título Demo\n## Primera sección\n")
    assert "15 días" in chunks[0].text and "Quince" not in chunks[0].text
    assert DOC.text[chunks[0].char_start :].startswith("## Primera sección")


def test_structural_splits_long_sections_repeating_header() -> None:
    chunks = chunk_structural(DOC, max_chars=110)
    first = [c for c in chunks if c.section == "Primera sección"]
    assert len(first) == 2
    assert all(c.text.startswith("# Título Demo\n## Primera sección\n") for c in first)
    assert "Ítem de lista" in first[1].text


def test_chunk_ids_are_traceable_and_unique() -> None:
    chunks = chunk_fixed(DOC, 60, 15)
    assert [c.chunk_id for c in chunks] == [f"demo.md_{i}" for i in range(len(chunks))]


def test_real_corpus_counts() -> None:
    rag = load_rag_config()
    documents = load_corpus(rag.corpus_dir)
    fixed = chunk_corpus(documents, "fixed", rag.chunking_params("fixed"))
    structural = chunk_corpus(documents, "structural", rag.chunking_params("structural"))
    # La consigna pide al menos 20 chunks; se exige a ambas estrategias, y en especial
    # a la que usa el asistente (ADR-052).
    assert len(fixed) >= 20 and len(structural) >= 20
    by_strategy = {"fixed": fixed, "structural": structural}
    assert len(by_strategy[rag.retrieval.default_strategy]) >= 20
    assert len({c.section for c in structural}) == 18, "18 secciones ##, algunas subdivididas"
    assert len({c.chunk_id for c in fixed}) == len(fixed)
    assert len({c.chunk_id for c in structural}) == len(structural)


def test_modified_corpus_is_rejected() -> None:
    import shutil
    import tempfile
    from pathlib import Path

    rag = load_rag_config()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        for path in rag.corpus_dir.iterdir():
            shutil.copy(path, target / path.name)
        (target / "hr_policies.md").write_text("alterado", encoding="utf-8")
        try:
            load_corpus(target)
        except CorpusError:
            return
    raise AssertionError("se esperaba CorpusError")


if __name__ == "__main__":
    raise SystemExit(run(globals()))
