"""Las dos estrategias de chunking que se comparan.

- ``fixed``: ventana de tamaño fijo con solapamiento, sobre el texto crudo del documento.
  Ignora la estructura a propósito: es la línea base "ingenua" (corta listas y puede
  mezclar el final de una sección con el inicio de la siguiente).
- ``structural``: un chunk por sección ``##``, con el título del documento y el de la
  sección antepuestos, para que cada chunk se entienda solo. Si una sección supera
  ``max_chars`` se subdivide por línea, repitiendo el encabezado en cada parte.

Cada chunk guarda ``char_start``/``char_end``: el tramo del documento original que
cubre. El golden dataset etiqueta la evidencia como un fragmento literal del documento,
y el solapamiento de tramos decide qué chunks son relevantes en cada estrategia (ADR-041).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from ingest.loader import Document

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # "<source>_<n>", n correlativo por documento y estrategia
    source: str        # documento de origen: lo que permite rastrear la cita
    section: str       # sección donde empieza el chunk
    strategy: str
    text: str          # lo que se embebe, se indexa y se muestra como contenido
    char_start: int    # tramo [char_start, char_end) del documento original
    char_end: int

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text")
        data.pop("chunk_id")
        return data


def _headings(text: str) -> list[tuple[int, int, str]]:
    """(posición, nivel, título) de cada encabezado Markdown."""
    return [(m.start(), len(m.group(1)), m.group(2)) for m in HEADING.finditer(text)]


def _document_title(doc: Document) -> str:
    for _, level, title in _headings(doc.text):
        if level == 1:
            return title
    return doc.source


def _section_at(headings: list[tuple[int, int, str]], position: int, title: str) -> str:
    """Último encabezado ## que empieza en o antes de ``position``."""
    current = title
    for start, level, heading in headings:
        if start > position:
            break
        if level >= 2:
            current = heading
    return current


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_fixed(doc: Document, size_chars: int, overlap_chars: int) -> list[Chunk]:
    if not 0 <= overlap_chars < size_chars:
        raise ValueError("se requiere 0 <= overlap_chars < size_chars")
    text = doc.text
    length = len(text)
    title = _document_title(doc)
    headings = _headings(text)
    chunks: list[Chunk] = []
    position = 0
    while position < length:
        end = min(position + size_chars, length)
        if end < length:
            # Retroceder hasta un espacio para no cortar palabras (sin achicar más de la mitad).
            cut = text.rfind(" ", position + size_chars // 2, end)
            cut_nl = text.rfind("\n", position + size_chars // 2, end)
            cut = max(cut, cut_nl)
            if cut > position:
                end = cut
        start, stop = _strip_span(text, position, end)
        if stop > start:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}_{len(chunks)}",
                    source=doc.source,
                    section=_section_at(headings, start, title),
                    strategy="fixed",
                    text=text[start:stop],
                    char_start=start,
                    char_end=stop,
                )
            )
        if end >= length:
            break
        # El siguiente chunk empieza overlap_chars antes, en un límite de palabra.
        next_start = max(end - overlap_chars, position + 1)
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1
        position = next_start
    return chunks


def chunk_structural(doc: Document, max_chars: int) -> list[Chunk]:
    text = doc.text
    title = _document_title(doc)
    sections = [(start, heading) for start, level, heading in _headings(text) if level == 2]
    chunks: list[Chunk] = []
    for index, (start, heading) in enumerate(sections):
        end = sections[index + 1][0] if index + 1 < len(sections) else len(text)
        body_start = text.index("\n", start) + 1 if "\n" in text[start:end] else end
        header = f"# {title}\n## {heading}\n"
        for part_start, part_end in _split_body(text, body_start, end, max_chars - len(header)):
            # El tramo de la primera parte incluye la línea "## ..." del original.
            span_start = start if part_start == body_start else part_start
            span_start, span_end = _strip_span(text, span_start, part_end)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}_{len(chunks)}",
                    source=doc.source,
                    section=heading,
                    strategy="structural",
                    text=header + text[part_start:part_end].strip(),
                    char_start=span_start,
                    char_end=span_end,
                )
            )
    return chunks


def _split_body(text: str, start: int, end: int, budget: int) -> list[tuple[int, int]]:
    """Parte el cuerpo de una sección en tramos de líneas completas de hasta ``budget``."""
    if end - start <= budget:
        return [(start, end)]
    parts: list[tuple[int, int]] = []
    part_start = start
    cursor = start
    while cursor < end:
        line_end = text.find("\n", cursor, end)
        line_end = end if line_end == -1 else line_end + 1
        if line_end - part_start > budget and cursor > part_start:
            parts.append((part_start, cursor))
            part_start = cursor
        cursor = line_end
    parts.append((part_start, end))
    return parts


def chunk_document(doc: Document, strategy: str, params: dict[str, Any]) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(doc, int(params["size_chars"]), int(params["overlap_chars"]))
    if strategy == "structural":
        return chunk_structural(doc, int(params["max_chars"]))
    raise ValueError(f"estrategia desconocida '{strategy}'")


def chunk_corpus(documents: list[Document], strategy: str, params: dict[str, Any]) -> list[Chunk]:
    return [chunk for doc in documents for chunk in chunk_document(doc, strategy, params)]
