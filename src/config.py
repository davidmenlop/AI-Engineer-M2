"""Carga de configuración: .env, config/providers.yaml y config/rag.yaml."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
PROVIDERS_FILE = REPO_ROOT / "config" / "providers.yaml"
RAG_FILE = REPO_ROOT / "config" / "rag.yaml"

DEFAULT_MODEL_KEY = "default"
EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}
STRATEGIES = ("fixed", "structural")
MODES = ("hybrid", "vector", "bm25")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    id: str
    temperature: float | None
    max_tokens: int
    input_per_1m_usd: float
    output_per_1m_usd: float
    effort: str | None = None


@dataclass(frozen=True)
class Settings:
    default_provider: str
    max_retries: int
    request_timeout_s: float
    index_dir: Path
    models_dir: Path
    metrics_path: Path
    log_level: str


@dataclass(frozen=True)
class FixedParams:
    size_chars: int
    overlap_chars: int


@dataclass(frozen=True)
class StructuralParams:
    max_chars: int


@dataclass(frozen=True)
class RetrievalParams:
    top_k: int
    candidates: int
    rrf_k: int
    default_mode: str
    default_strategy: str


@dataclass(frozen=True)
class RerankParams:
    enabled: bool
    model: str
    candidates: int


@dataclass(frozen=True)
class RagConfig:
    corpus_dir: Path
    embedding_model: str
    embedding_dimensions: int
    collection_prefix: str
    distance: str
    fixed: FixedParams
    structural: StructuralParams
    retrieval: RetrievalParams
    rerank: RerankParams
    raw: dict[str, Any]

    def chunking_params(self, strategy: str) -> dict[str, Any]:
        return dict(self.raw["chunking"][strategy])

    def chunking_sha256(self, strategy: str) -> str:
        """Huella de lo que determina los chunks: estrategia y parámetros."""
        payload = json.dumps({"strategy": strategy, **self.chunking_params(strategy)}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"no existe {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} debe ser un mapa YAML")
    return data


def _read_providers_file() -> dict[str, Any]:
    data = _read_yaml(PROVIDERS_FILE)
    if "providers" not in data:
        raise ConfigError("providers.yaml debe tener la clave 'providers'")
    return data


def load_settings() -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    providers = _read_providers_file()

    try:
        max_retries = int(_env("MAX_RETRIES", "2"))
        timeout_s = float(_env("REQUEST_TIMEOUT_S", "60"))
    except ValueError as exc:
        raise ConfigError("MAX_RETRIES y REQUEST_TIMEOUT_S deben ser numéricos") from exc
    if max_retries < 0 or timeout_s <= 0:
        raise ConfigError("se requiere MAX_RETRIES >= 0 y REQUEST_TIMEOUT_S > 0")

    log_level = _env("LOG_LEVEL", "INFO").upper()
    if not isinstance(logging.getLevelName(log_level), int):
        raise ConfigError(f"LOG_LEVEL inválido: {log_level}")

    return Settings(
        default_provider=_env("DEFAULT_PROVIDER", str(providers["default_provider"])),
        max_retries=max_retries,
        request_timeout_s=timeout_s,
        index_dir=_repo_path(_env("INDEX_DIR", "data/index")),
        models_dir=_repo_path(_env("MODELS_DIR", "data/models")),
        metrics_path=_repo_path(_env("METRICS_PATH", "data/metrics.jsonl")),
        log_level=log_level,
    )


@lru_cache(maxsize=1)
def load_rag_config() -> RagConfig:
    raw = _read_yaml(RAG_FILE)
    try:
        chunking = raw["chunking"]
        retrieval = raw["retrieval"]
        config = RagConfig(
            corpus_dir=_repo_path(str(raw["corpus_dir"])),
            embedding_model=str(raw["embeddings"]["model"]),
            embedding_dimensions=int(raw["embeddings"]["dimensions"]),
            collection_prefix=str(raw["vector_store"]["collection_prefix"]),
            distance=str(raw["vector_store"]["distance"]),
            fixed=FixedParams(
                size_chars=int(chunking["fixed"]["size_chars"]),
                overlap_chars=int(chunking["fixed"]["overlap_chars"]),
            ),
            structural=StructuralParams(max_chars=int(chunking["structural"]["max_chars"])),
            retrieval=RetrievalParams(
                top_k=int(retrieval["top_k"]),
                candidates=int(retrieval["candidates"]),
                rrf_k=int(retrieval["rrf_k"]),
                default_mode=str(retrieval["default_mode"]),
                default_strategy=str(retrieval["default_strategy"]),
            ),
            rerank=RerankParams(
                enabled=bool(raw.get("rerank", {}).get("enabled", False)),
                model=str(raw.get("rerank", {}).get("model", "BAAI/bge-reranker-v2-m3-int8")),
                candidates=int(raw.get("rerank", {}).get("candidates", 10)),
            ),
            raw=raw,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"rag.yaml incompleto o inválido: {exc}") from exc

    if not 0 <= config.fixed.overlap_chars < config.fixed.size_chars:
        raise ConfigError("chunking.fixed: se requiere 0 <= overlap_chars < size_chars")
    if config.retrieval.top_k < 1 or config.retrieval.candidates < config.retrieval.top_k:
        raise ConfigError("retrieval: se requiere 1 <= top_k <= candidates")
    if config.rerank.candidates < config.retrieval.top_k:
        raise ConfigError("rerank.candidates debe ser >= retrieval.top_k")
    if config.retrieval.default_mode not in MODES:
        raise ConfigError(f"retrieval.default_mode debe ser uno de {MODES}")
    if config.retrieval.default_strategy not in STRATEGIES:
        raise ConfigError(f"retrieval.default_strategy debe ser una de {STRATEGIES}")
    return config


def role_defaults(role: str) -> tuple[str, str]:
    """(proveedor, clave de modelo) configurados para un rol: generation o judge."""
    roles = _read_providers_file().get("roles", {})
    try:
        entry = roles[role]
        return str(entry["provider"]), str(entry.get("model", DEFAULT_MODEL_KEY))
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"providers.yaml no define el rol '{role}'") from exc


def available_models(provider: str) -> list[str]:
    providers = _read_providers_file()["providers"]
    try:
        return sorted(providers[provider]["models"])
    except (KeyError, TypeError):
        return []


def load_model_config(provider: str, model_key: str = DEFAULT_MODEL_KEY) -> ModelConfig:
    providers = _read_providers_file()["providers"]
    try:
        models = providers[provider]["models"]
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"providers.yaml no define modelos para '{provider}'") from exc
    if model_key not in models:
        options = ", ".join(sorted(models))
        raise ConfigError(f"'{provider}' no tiene el modelo '{model_key}'. Opciones: {options}")
    raw = models[model_key]
    try:
        pricing = raw["pricing"]
        temperature = raw.get("temperature")
        effort = raw.get("effort")
        config = ModelConfig(
            id=str(raw["id"]),
            temperature=None if temperature is None else float(temperature),
            max_tokens=int(raw["max_tokens"]),
            input_per_1m_usd=float(pricing["input_per_1m_usd"]),
            output_per_1m_usd=float(pricing["output_per_1m_usd"]),
            effort=None if effort is None else str(effort),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(
            f"providers.yaml: el modelo '{model_key}' de '{provider}' necesita id, "
            "max_tokens y pricing válidos"
        ) from exc
    if config.effort is not None and config.effort not in EFFORT_LEVELS:
        raise ConfigError(f"effort inválido '{config.effort}'. Opciones: {sorted(EFFORT_LEVELS)}")
    return config
