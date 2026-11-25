"""Golden dataset: carga, validación y etiquetado de chunks esperados por estrategia.

Cada pregunta trae su evidencia como fragmentos *literales* del corpus
(``{source, quote}``), no como ids de chunk: la evidencia no depende de cómo se corte
el corpus. A partir de ella se calculan los chunks esperados de cada estrategia con
una misma regla (ADR-041):

    un chunk es relevante si su tramo se solapa con el de alguna evidencia en al menos
    min(MIN_OVERLAP_CHARS, largo de la evidencia) caracteres.

Uso:
    python evals/dataset.py --write   # recalcula expected_chunks en el dataset
    python evals/dataset.py --check   # falla si expected_chunks está desactualizado
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.chunking import Chunk, chunk_corpus  # noqa: E402
from ingest.loader import Document, load_corpus  # noqa: E402
from src.config import REPO_ROOT, STRATEGIES, load_rag_config  # noqa: E402

DATASET_PATH = REPO_ROOT / "evals" / "dataset.jsonl"
MIN_OVERLAP_CHARS = 30
TYPES = ("directa", "sinonimos", "termino_exacto", "distractor", "cruza_documentos", "fuera_de_alcance")


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Evidence:
    source: str
    quote: str
    start: int
    end: int

    def overlap(self, chunk: Chunk) -> int:
        if chunk.source != self.source:
            return 0
        return max(0, min(self.end, chunk.char_end) - max(self.start, chunk.char_start))

    def is_relevant(self, chunk: Chunk) -> bool:
        return self.overlap(chunk) >= min(MIN_OVERLAP_CHARS, self.end - self.start)


@dataclass(frozen=True)
class Case:
    id: str
    type: str
    in_scope: bool
    question: str
    evidence: tuple[Evidence, ...]
    reference_answer: str
    expected_chunks: dict[str, list[str]]


def dataset_sha256(path: Path = DATASET_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(documents: dict[str, Document], source: str, quote: str) -> tuple[int, int]:
    if source not in documents:
        raise DatasetError(f"documento desconocido '{source}'")
    text = documents[source].text
    start = text.find(quote)
    if start == -1:
        raise DatasetError(f"la evidencia no aparece literal en {source}: {quote[:60]!r}")
    if text.find(quote, start + 1) != -1:
        raise DatasetError(f"la evidencia aparece más de una vez en {source}: {quote[:60]!r}")
    return start, start + len(quote)


def expected_for(evidence: tuple[Evidence, ...], chunks: list[Chunk]) -> list[str]:
    return [c.chunk_id for c in chunks if any(e.is_relevant(c) for e in evidence)]


def corpus_chunks() -> dict[str, list[Chunk]]:
    rag = load_rag_config()
    documents = load_corpus(rag.corpus_dir)
    return {s: chunk_corpus(documents, s, rag.chunking_params(s)) for s in STRATEGIES}


def read_rows(path: Path = DATASET_PATH) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"línea {number}: JSON inválido ({exc})") from exc
    return rows


def parse_rows(rows: list[dict]) -> list[Case]:
    rag = load_rag_config()
    documents = {doc.source: doc for doc in load_corpus(rag.corpus_dir)}
    cases: list[Case] = []
    seen: set[str] = set()
    for row in rows:
        case_id = str(row.get("id", ""))
        if not case_id or case_id in seen:
            raise DatasetError(f"id vacío o repetido: {case_id!r}")
        seen.add(case_id)
        if row.get("type") not in TYPES:
            raise DatasetError(f"{case_id}: type debe ser uno de {TYPES}")
        if not str(row.get("question", "")).strip():
            raise DatasetError(f"{case_id}: pregunta vacía")
        evidence = tuple(
            Evidence(e["source"], e["quote"], *locate(documents, e["source"], e["quote"]))
            for e in row.get("evidence", [])
        )
        in_scope = bool(row.get("in_scope"))
        if in_scope != bool(evidence):
            raise DatasetError(f"{case_id}: in_scope=true exige evidencia y in_scope=false no la admite")
        if (row["type"] == "fuera_de_alcance") == in_scope:
            raise DatasetError(f"{case_id}: el tipo fuera_de_alcance corresponde a in_scope=false")
        cases.append(
            Case(
                id=case_id,
                type=row["type"],
                in_scope=in_scope,
                question=row["question"].strip(),
                evidence=evidence,
                reference_answer=str(row.get("reference_answer", "")),
                expected_chunks={k: list(v) for k, v in row.get("expected_chunks", {}).items()},
            )
        )
    return cases


def label(rows: list[dict], cases: list[Case]) -> list[dict]:
    chunks = corpus_chunks()
    labeled = []
    for row, case in zip(rows, cases, strict=True):
        row = dict(row)
        row["expected_chunks"] = {s: expected_for(case.evidence, chunks[s]) for s in STRATEGIES}
        labeled.append(row)
    return labeled


def stale_cases(rows: list[dict], cases: list[Case]) -> list[str]:
    fresh = label(rows, cases)
    return [r["id"] for r, f in zip(rows, fresh, strict=True) if r.get("expected_chunks") != f["expected_chunks"]]


def load_dataset(path: Path = DATASET_PATH, *, require_labels: bool = True) -> list[Case]:
    rows = read_rows(path)
    cases = parse_rows(rows)
    if require_labels:
        stale = stale_cases(rows, cases)
        if stale:
            raise DatasetError(
                f"expected_chunks desactualizado en {len(stale)} casos ({', '.join(stale[:5])}...): "
                "corré python evals/dataset.py --write"
            )
    return cases


def write_dataset(rows: list[dict], path: Path = DATASET_PATH) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summary(cases: list[Case]) -> str:
    by_type = Counter(c.type for c in cases)
    lines = [f"{len(cases)} preguntas ({sum(c.in_scope for c in cases)} dentro de alcance)"]
    lines += [f"  {t:<18} {by_type.get(t, 0)}" for t in TYPES]
    for strategy in STRATEGIES:
        sizes = [len(c.expected_chunks.get(strategy, [])) for c in cases if c.in_scope]
        lines.append(
            f"  chunks esperados por pregunta ({strategy}): "
            f"min {min(sizes)} / prom {sum(sizes) / len(sizes):.2f} / max {max(sizes)}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida y etiqueta el golden dataset.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="recalcular expected_chunks y guardar")
    group.add_argument("--check", action="store_true", help="validar sin modificar")
    args = parser.parse_args(argv)
    try:
        rows = read_rows()
        cases = parse_rows(rows)
        if args.write:
            write_dataset(label(rows, cases))
            cases = load_dataset()
            print("expected_chunks actualizado")
        else:
            stale = stale_cases(rows, cases)
            if stale:
                print(f"desactualizado: {', '.join(stale)}. Corré --write", file=sys.stderr)
                return 1
            cases = load_dataset()
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(summary(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
