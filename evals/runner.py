"""Suite de evals: mide retrieval y generación por separado.

Retrieval (sin LLM ni claves): todas las estrategias x modos con el mismo dataset y k.
    python evals/runner.py
    python evals/runner.py --k 1 3 5 --modes hybrid

Generación (LLM + juez de groundedness), sobre una estrategia y modo:
    python evals/runner.py --generation --strategy structural --provider anthropic
    python evals/runner.py --generation --strategy fixed --provider anthropic --limit 5 --no-save

Tablas markdown a partir de resultados guardados:
    python evals/runner.py --compare evals/results/*.json

Cada corrida guardada en evals/results/ incluye el sha256 del dataset, del corpus y de
los prompts, los parámetros de recuperación, el commit y si el árbol estaba limpio.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.dataset import Case, DatasetError, dataset_sha256, load_dataset  # noqa: E402
from evals.judge import GroundednessJudge  # noqa: E402
from evals.metrics import Span, mean, percentile, retrieval_scores  # noqa: E402
from ingest.loader import corpus_sha256, load_corpus  # noqa: E402
from src.assistant import Assistant  # noqa: E402
from src.config import (  # noqa: E402
    MODES,
    REPO_ROOT,
    STRATEGIES,
    ConfigError,
    load_model_config,
    load_rag_config,
    load_settings,
    role_defaults,
)
from src.embeddings import Embedder  # noqa: E402
from src.prompts import PromptError, load_prompt  # noqa: E402
from src.providers.base import ProviderFatalError, ProviderUnavailable  # noqa: E402
from src.providers.factory import build_provider  # noqa: E402
from src.retriever import Retriever  # noqa: E402
from src.vector_store import IndexNotReady  # noqa: E402

RESULTS_DIR = REPO_ROOT / "evals" / "results"
RETRIEVAL_KEYS = ("precision", "recall", "evidence_recall", "complete", "hit", "mrr")
logger = logging.getLogger("evals")


# ---------------------------------------------------------------- utilidades


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"commit": run("rev-parse", "--short", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def spans_of(retriever: Retriever, chunk_ids: list[str]) -> list[Span]:
    spans = []
    for chunk_id in chunk_ids:
        meta = retriever.chunk(chunk_id).metadata
        spans.append(Span(str(meta["source"]), int(meta["char_start"]), int(meta["char_end"])))
    return spans


def evidence_spans(case: Case) -> list[Span]:
    return [Span(e.source, e.start, e.end) for e in case.evidence]


def check_labels(cases: list[Case], retriever: Retriever) -> None:
    """Los chunks esperados del dataset tienen que existir en el índice que se mide."""
    for case in cases:
        missing = [c for c in case.expected_chunks.get(retriever.strategy, []) if not _exists(retriever, c)]
        if missing:
            raise DatasetError(
                f"{case.id}: chunks esperados {missing} no están en el índice '{retriever.strategy}'. "
                "Reindexá y corré python evals/dataset.py --write"
            )


def _exists(retriever: Retriever, chunk_id: str) -> bool:
    try:
        retriever.chunk(chunk_id)
    except KeyError:
        return False
    return True


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> dict[str, float | None]:
    return {key: mean(row[key] for row in rows) for key in keys}


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def save(payload: dict[str, Any], prefix: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{prefix}_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def header(rag) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "dataset_sha256": dataset_sha256(),
        "corpus_sha256": corpus_sha256(load_corpus(rag.corpus_dir)),
        "retrieval_params": {
            "embedding_model": rag.embedding_model,
            "candidates": rag.retrieval.candidates,
            "rrf_k": rag.retrieval.rrf_k,
            "chunking": {s: rag.chunking_params(s) for s in STRATEGIES},
        },
    }


# ---------------------------------------------------------------- retrieval


def eval_retrieval(cases: list[Case], retriever: Retriever, mode: str, k: int) -> dict[str, Any]:
    rows = []
    latencies = []
    for case in cases:
        if not case.in_scope:
            continue
        results = retriever.retrieve(case.question, top_k=k, mode=mode)
        latencies.append(retriever.last_latency_ms)
        ids = [r.chunk_id for r in results]
        expected = case.expected_chunks[retriever.strategy]
        scores = retrieval_scores(ids, expected, evidence_spans(case), spans_of(retriever, ids), k)
        rows.append({"id": case.id, "type": case.type, "retrieved": ids, "expected": expected, **scores})
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    return {
        "strategy": retriever.strategy,
        "mode": mode,
        "k": k,
        "n": len(rows),
        "chunks_in_index": retriever.chunk_count,
        "aggregate": {**aggregate(rows, RETRIEVAL_KEYS), "latency_ms_p50": percentile(latencies, 50)},
        "by_type": {t: {"n": len(r), **aggregate(r, RETRIEVAL_KEYS)} for t, r in sorted(by_type.items())},
        "cases": rows,
    }


def retrieval_table(runs: list[dict]) -> str:
    lines = [
        "| estrategia | modo | k | chunks | precision@k | recall@k | evidence_recall@k | complete@k | hit@k | MRR |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        a = run["aggregate"]
        lines.append(
            f"| {run['strategy']} | {run['mode']} | {run['k']} | {run['chunks_in_index']} | {fmt(a['precision'])} | "
            f"{fmt(a['recall'])} | {fmt(a['evidence_recall'])} | {fmt(a['complete'])} | {fmt(a['hit'])} | {fmt(a['mrr'])} |"
        )
    return "\n".join(lines)


def by_type_table(runs: list[dict], metric: str = "evidence_recall") -> str:
    types = sorted({t for run in runs for t in run["by_type"]})
    lines = [
        f"| estrategia | modo | k | " + " | ".join(types) + " |",
        "|---|---|---|" + "---|" * len(types),
    ]
    for run in runs:
        cells = [fmt(run["by_type"].get(t, {}).get(metric)) for t in types]
        lines.append(f"| {run['strategy']} | {run['mode']} | {run['k']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_retrieval(args, cases, settings, rag) -> int:
    embedder = Embedder(rag.embedding_model, settings.models_dir)
    runs = []
    for strategy in args.strategies:
        retriever = Retriever.open(strategy, settings, rag, embedder)
        check_labels(cases, retriever)
        for mode in args.modes:
            for k in args.k:
                runs.append(eval_retrieval(cases, retriever, mode, k))
    print(retrieval_table(runs))
    print()
    print("evidence_recall@k por tipo de pregunta:")
    print(by_type_table(runs))
    if not args.no_save:
        path = save({"kind": "retrieval", **header(rag), "runs": runs}, "retrieval")
        print(f"\nresultados: {path.relative_to(REPO_ROOT)}")
    return 0


# ---------------------------------------------------------------- generación


def diagnose(case: Case, complete: float, refused: bool, grounded: float | None, success: bool) -> str:
    """En qué etapa falla una pregunta dentro de alcance."""
    retrieval_ok = complete == 1.0
    generation_ok = success and not refused and grounded == 1.0
    if retrieval_ok and generation_ok:
        return "ok"
    if not retrieval_ok and not generation_ok:
        return "ambas"
    return "generacion" if retrieval_ok else "retrieval"


def run_generation(args, cases, settings, rag) -> int:
    gen_provider_name, gen_model_key = role_defaults("generation")
    judge_provider_name, judge_model_key = role_defaults("judge")
    gen_provider_name = args.provider or gen_provider_name
    gen_model_key = args.model or gen_model_key
    judge_provider_name = args.judge_provider or judge_provider_name
    judge_model_key = args.judge_model or judge_model_key

    strategy = args.strategy or rag.retrieval.default_strategy
    mode = args.mode or rag.retrieval.default_mode
    k = args.k[0]
    retriever = Retriever.open(strategy, settings, rag)
    check_labels(cases, retriever)

    gen_prompt = load_prompt(args.prompt, kind="generation")
    judge_prompt = load_prompt(args.judge_prompt, kind="judge")
    assistant = Assistant(
        retriever, build_provider(gen_provider_name), load_model_config(gen_provider_name, gen_model_key), gen_prompt,
        max_retries=settings.max_retries, timeout_s=settings.request_timeout_s, mode=mode, top_k=k,
    )
    use_judge = judge_provider_name != "mock"
    judge = None
    if use_judge:
        judge = GroundednessJudge(
            build_provider(judge_provider_name), load_model_config(judge_provider_name, judge_model_key), judge_prompt,
            max_retries=settings.max_retries, timeout_s=settings.request_timeout_s,
        )
    else:
        logger.warning("juez mock: no se mide groundedness (solo citas y rechazo)")

    selected = cases[: args.limit] if args.limit else cases
    rows = []
    for index, case in enumerate(selected, start=1):
        result = assistant.answer(case.question)
        ids = [c.chunk_id for c in result.retrieved]
        expected = case.expected_chunks[strategy]
        row: dict[str, Any] = {
            "id": case.id,
            "type": case.type,
            "in_scope": case.in_scope,
            "answer": result.response.system_answer,
            "retrieved": ids,
            "cited": result.cited_ids,
            "refused": result.refused,
            "success": result.success,
            "attempts": result.attempts,
            "gen_cost_usd": result.cost_usd,
            "gen_latency_s": round(result.latency_s, 3),
        }
        if case.in_scope:
            scores = retrieval_scores(ids, expected, evidence_spans(case), spans_of(retriever, ids), k)
            row.update({f"retrieval_{key}": scores[key] for key in ("evidence_recall", "complete", "hit")})
            row["cited_relevant"] = (
                sum(c in expected for c in result.cited_ids) / len(result.cited_ids) if result.cited_ids else None
            )
            verdict = None
            if judge is not None and result.success and not result.refused:
                verdict = judge.judge(case.question, result.response.system_answer, result.response.chunks_related)
                row.update({
                    "groundedness": verdict.groundedness,
                    "claims": verdict.claims,
                    "judge_cost_usd": verdict.cost_usd,
                    "judge_error": verdict.error,
                })
            row["diagnosis"] = diagnose(case, scores["complete"], result.refused, row.get("groundedness"), result.success)
        else:
            row["diagnosis"] = "ok" if result.refused else "respondio_fuera_de_alcance"
        rows.append(row)
        print(
            f"[{index:>2}/{len(selected)}] {case.id} {case.type:<16} refused={str(result.refused):<5} "
            f"ground={fmt(row.get('groundedness'), 2)} diag={row['diagnosis']}",
            file=sys.stderr,
        )

    in_scope = [r for r in rows if r["in_scope"]]
    out_scope = [r for r in rows if not r["in_scope"]]
    answered = [r for r in in_scope if not r["refused"] and r["success"]]
    judged = [r for r in answered if r.get("groundedness") is not None]
    diagnosis_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        diagnosis_counts[r["diagnosis"]] += 1
    summary = {
        "n": len(rows),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_scope),
        "groundedness": mean(r.get("groundedness") for r in judged),
        "fully_grounded_rate": mean(float(r["groundedness"] == 1.0) for r in judged),
        "cited_relevant": mean(r.get("cited_relevant") for r in answered),
        "answered_rate": round(len(answered) / len(in_scope), 4) if in_scope else None,
        "false_refusal_rate": mean(float(r["refused"]) for r in in_scope),
        "correct_refusal_rate": mean(float(r["refused"]) for r in out_scope),
        "fallback_rate": mean(float(not r["success"]) for r in rows),
        "retry_rate": mean(float(r["attempts"] > 1) for r in rows),
        "retrieval_complete": mean(r.get("retrieval_complete") for r in in_scope),
        "gen_cost_usd_total": round(sum(r["gen_cost_usd"] for r in rows), 6),
        "judge_cost_usd_total": round(sum(r.get("judge_cost_usd", 0.0) for r in rows), 6),
        "gen_latency_s_p50": percentile([r["gen_latency_s"] for r in rows], 50),
        "gen_latency_s_p95": percentile([r["gen_latency_s"] for r in rows], 95),
        "diagnosis": dict(sorted(diagnosis_counts.items())),
    }
    run = {
        "strategy": strategy,
        "mode": mode,
        "k": k,
        "generation": {"provider": gen_provider_name, "model": assistant_model_id(gen_provider_name, gen_model_key),
                       "prompt": gen_prompt.version, "prompt_sha256": gen_prompt.sha256},
        "judge": {"provider": judge_provider_name, "model": judge.model_id if judge else None,
                  "prompt": judge_prompt.version, "prompt_sha256": judge_prompt.sha256},
        "summary": summary,
        "cases": rows,
    }
    print(generation_table([run]))
    if not args.no_save:
        path = save({"kind": "generation", **header(rag), "limit": args.limit, "runs": [run]}, "generation")
        print(f"\nresultados: {path.relative_to(REPO_ROOT)}")
    return 0


def assistant_model_id(provider: str, key: str) -> str:
    return load_model_config(provider, key).id


def generation_table(runs: list[dict]) -> str:
    lines = [
        "| estrategia | modo | k | generador | groundedness | 100 % sostenidas | citas relevantes | "
        "rechazo correcto | falso rechazo | retrieval completo | costo gen + juez (USD) | p95 s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        s = run["summary"]
        cost = (s["gen_cost_usd_total"] or 0) + (s["judge_cost_usd_total"] or 0)
        lines.append(
            f"| {run['strategy']} | {run['mode']} | {run['k']} | {run['generation']['model']} | {fmt(s['groundedness'])} | "
            f"{fmt(s['fully_grounded_rate'])} | {fmt(s['cited_relevant'])} | {fmt(s['correct_refusal_rate'])} | "
            f"{fmt(s['false_refusal_rate'])} | {fmt(s['retrieval_complete'])} | {cost:.4f} | {fmt(s['gen_latency_s_p95'], 2)} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- compare


def run_compare(paths: list[str]) -> int:
    retrieval_runs, generation_runs = [], []
    current = dataset_sha256()
    for raw in paths:
        payload = json.loads(Path(raw).read_text(encoding="utf-8"))
        if payload.get("dataset_sha256") != current:
            print(f"aviso: {raw} usa otra versión del dataset; se omite", file=sys.stderr)
            continue
        target = retrieval_runs if payload["kind"] == "retrieval" else generation_runs
        target.extend(payload["runs"])
    if retrieval_runs:
        print(retrieval_table(retrieval_runs))
        print()
    if generation_runs:
        print(generation_table(generation_runs))
    return 0


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evals de retrieval y generación de Nubbix Assist.")
    parser.add_argument("--generation", action="store_true", help="evaluar generación (necesita un LLM)")
    parser.add_argument("--compare", nargs="+", metavar="JSON", help="tablas a partir de resultados guardados")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--k", nargs="+", type=int, help="top_k (default: rag.yaml)")
    parser.add_argument("--strategy", choices=STRATEGIES, help="estrategia para --generation")
    parser.add_argument("--mode", choices=MODES, help="modo para --generation")
    parser.add_argument("--provider", help="proveedor del generador (default: rol generation)")
    parser.add_argument("--model", help="clave de modelo del generador")
    parser.add_argument("--judge-provider", help="proveedor del juez (default: rol judge)")
    parser.add_argument("--judge-model", help="clave de modelo del juez")
    parser.add_argument("--prompt", default="generation_v1")
    parser.add_argument("--judge-prompt", default="judge_groundedness_v1")
    parser.add_argument("--limit", type=int, help="evaluar solo las primeras N preguntas (prueba rápida)")
    parser.add_argument("--no-save", action="store_true", help="no guardar en evals/results/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.compare:
        return run_compare(args.compare)
    try:
        settings = load_settings()
        rag = load_rag_config()
        logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
        for noisy in ("httpx", "httpx2", "chromadb", "src.llm"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        args.k = args.k or [rag.retrieval.top_k]
        cases = load_dataset()
        started = time.perf_counter()
        code = run_generation(args, cases, settings, rag) if args.generation else run_retrieval(args, cases, settings, rag)
        print(f"\n{time.perf_counter() - started:.1f} s", file=sys.stderr)
        return code
    except (ConfigError, DatasetError, IndexNotReady, PromptError, ProviderUnavailable, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ProviderFatalError as exc:
        print(f"error fatal del proveedor (no se guardan resultados): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
