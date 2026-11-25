"""Carga del corpus Markdown y verificación de que no fue modificado."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

CHECKSUMS_FILE = "SHA256SUMS"


@dataclass(frozen=True)
class Document:
    source: str   # nombre del archivo, p. ej. "hr_policies.md": es lo que se cita
    text: str
    sha256: str


class CorpusError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_checksums(corpus_dir: Path) -> dict[str, str]:
    path = corpus_dir / CHECKSUMS_FILE
    if not path.exists():
        return {}
    checksums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            checksums[name.strip()] = digest
    return checksums


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Lee los .md en orden alfabético. Falla si alguno difiere de SHA256SUMS.

    El corpus debe quedar idéntico al provisto: si cambia, los fragmentos de evidencia
    del golden dataset dejan de apuntar a las posiciones correctas.
    """
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise CorpusError(f"no hay archivos .md en {corpus_dir}")
    expected = read_checksums(corpus_dir)
    documents = []
    for path in paths:
        raw = path.read_bytes()
        digest = _sha256(raw)
        if expected and expected.get(path.name) != digest:
            raise CorpusError(
                f"{path.name} no coincide con {CHECKSUMS_FILE}: el corpus debe usarse sin modificar"
            )
        documents.append(Document(source=path.name, text=raw.decode("utf-8"), sha256=digest))
    return documents


def corpus_sha256(documents: list[Document]) -> str:
    """Huella de todo el corpus (nombre + hash de cada documento)."""
    joined = "\n".join(f"{doc.source}:{doc.sha256}" for doc in documents)
    return _sha256(joined.encode("utf-8"))
