"""Revisión humana del golden dataset, pregunta por pregunta.

Uso (en tu terminal: es interactivo):
    python evals/review_labels.py --by "Nombre Apellido"   # revisa las pendientes
    python evals/review_labels.py --all                    # vuelve a revisar todas
    python evals/review_labels.py --id q29                 # una pregunta puntual
    python evals/review_labels.py --summary                # estado, sin interacción

Por cada pregunta se muestra la evidencia resaltada dentro de su sección del corpus,
la respuesta de referencia y los chunks esperados de cada estrategia.

Teclas:
    Enter   aprobar (la evidencia responde la pregunta y alcanza)
    d       marcar en disputa (pide una nota: qué falta o sobra)
    s       saltar (queda como estaba)
    q       salir

Cada decisión se guarda al instante en evals/dataset_review.jsonl, separado del
dataset para no cambiar su sha256. La revisión queda atada al contenido de la
pregunta: si después se edita la pregunta o su evidencia, vuelve a quedar pendiente.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.dataset import Case, corpus_chunks, load_dataset  # noqa: E402
from src.config import REPO_ROOT  # noqa: E402

REVIEW_PATH = REPO_ROOT / "evals" / "dataset_review.jsonl"
WIDTH = 92
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


class Quit(Exception):
    pass


def case_sha256(case: Case) -> str:
    payload = {
        "question": case.question,
        "in_scope": case.in_scope,
        "evidence": [[e.source, e.quote] for e in case.evidence],
        "reference_answer": case.reference_answer,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_reviews(path: Path = REVIEW_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    reviews = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            reviews[record["id"]] = record  # la última decisión de cada id gana
    return reviews


def append_review(record: dict, path: Path = REVIEW_PATH) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def status(case: Case, reviews: dict[str, dict]) -> str:
    review = reviews.get(case.id)
    if review is None or review.get("case_sha256") != case_sha256(case):
        return "pendiente"
    return review["status"]


def summary(cases: list[Case], reviews: dict[str, dict]) -> str:
    counts: dict[str, int] = {}
    for case in cases:
        counts[status(case, reviews)] = counts.get(status(case, reviews), 0) + 1
    disputed = [c.id for c in cases if status(c, reviews) == "en_disputa"]
    lines = [f"{len(cases)} preguntas: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    for case_id in disputed:
        lines.append(f"  en disputa {case_id}: {reviews[case_id].get('note', '')}")
    return "\n".join(lines)


def section_context(case: Case, sections: dict[str, list]) -> list[str]:
    """Cada sección que contiene evidencia, una sola vez, con sus evidencias resaltadas."""
    grouped: dict[str, tuple[object, list[str]]] = {}
    loose: list[str] = []
    for evidence in case.evidence:
        container = next(
            (c for c in sections[evidence.source] if c.char_start <= evidence.start and evidence.end <= c.char_end),
            None,
        )
        if container is None:
            loose.append(f"{evidence.source}: {BOLD}{evidence.quote}{RESET}")
        else:
            grouped.setdefault(container.chunk_id, (container, []))[1].append(evidence.quote)
    blocks = []
    for chunk_id, (chunk, quotes) in grouped.items():
        text = chunk.text
        for quote in quotes:
            text = text.replace(quote, f"{BOLD}{quote}{RESET}")
        blocks.append(f"{DIM}[{chunk_id}]{RESET}\n{text}")
    return blocks + loose


def show(case: Case, index: int, total: int, sections: dict[str, list], current: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f"{BOLD}{case.id}{RESET}  ({index}/{total})  tipo: {case.type}  estado: {current}")
    print("-" * WIDTH)
    print(textwrap.fill(f"Pregunta: {case.question}", WIDTH))
    print(textwrap.fill(f"Respuesta de referencia: {case.reference_answer}", WIDTH))
    if not case.in_scope:
        print(f"\n{BOLD}Fuera de alcance{RESET}: verificá que la respuesta NO esté en ningún documento.")
        return
    print(f"\nEvidencia ({len(case.evidence)}), resaltada en su sección:")
    for block in section_context(case, sections):
        print(textwrap.indent(block, "  "))
    print(f"\n{DIM}Chunks esperados  structural: {case.expected_chunks.get('structural')}")
    print(f"                   fixed:      {case.expected_chunks.get('fixed')}{RESET}")


def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise Quit from exc


def review(cases: list[Case], reviews: dict[str, dict], reviewer: str) -> None:
    sections = {}
    for chunk in corpus_chunks()["structural"]:
        sections.setdefault(chunk.source, []).append(chunk)
    for index, case in enumerate(cases, start=1):
        current = status(case, reviews)
        show(case, index, len(cases), sections, current)
        while True:
            key = ask("\n[Enter] aprobar  [d] disputa  [s] saltar  [q] salir > ").strip().lower()
            if key == "q":
                raise Quit
            if key == "s":
                break
            if key in ("", "d"):
                note = ask("Nota (qué falta o sobra): ").strip() if key == "d" else ""
                record = {
                    "id": case.id,
                    "status": "aprobada" if key == "" else "en_disputa",
                    "note": note,
                    "by": reviewer,
                    "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                    "case_sha256": case_sha256(case),
                }
                append_review(record)
                reviews[case.id] = record
                break
            print("Tecla no reconocida.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Revisión humana del golden dataset.")
    parser.add_argument("--all", action="store_true", help="revisar también las ya revisadas")
    parser.add_argument("--id", action="append", help="revisar solo estas preguntas (repetible)")
    parser.add_argument("--by", default="", help="nombre de quien revisa (queda registrado)")
    parser.add_argument("--summary", action="store_true", help="mostrar el estado y salir")
    args = parser.parse_args(argv)

    cases = load_dataset()
    reviews = load_reviews()
    if args.summary:
        print(summary(cases, reviews))
        return 0
    if not sys.stdin.isatty():
        print("error: la revisión es interactiva; corré el comando en tu terminal", file=sys.stderr)
        return 2
    if not args.by:
        args.by = ask("¿Quién revisa? ").strip() or "sin nombre"

    selected = [c for c in cases if not args.id or c.id in args.id]
    if not args.all and not args.id:
        selected = [c for c in selected if status(c, reviews) == "pendiente"]
    if not selected:
        print("No hay preguntas pendientes. Usá --all para revisar de nuevo.")
        print(summary(cases, reviews))
        return 0
    try:
        review(selected, reviews, args.by)
    except (Quit, KeyboardInterrupt):
        print("\nRevisión interrumpida; lo decidido quedó guardado.")
    print("\n" + summary(cases, load_reviews()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
