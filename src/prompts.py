"""Carga de prompts versionados desde prompts/<nombre>.yaml.

Un prompt es inmutable una vez evaluado: cambiarlo significa crear una versión nueva
(heredado de M1, ADR-020). Hay dos tipos, cada uno con sus marcadores obligatorios:

- generation: user_template con {question} y {context}; clave refusal_text.
- judge:      user_template con {question}, {answer} y {context}.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "prompts"
REQUIRED_KEYS = ("version", "kind", "description", "system", "user_template")
PLACEHOLDERS = {
    "generation": ("{question}", "{context}"),
    "judge": ("{question}", "{answer}", "{context}"),
}


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class PromptSpec:
    name: str
    version: str
    kind: str
    description: str
    system: str
    user_template: str
    sha256: str
    refusal_text: str = ""

    def render(self, **values: str) -> tuple[str, str]:
        user = self.user_template
        for key, value in values.items():
            user = user.replace("{" + key + "}", value)
        system = self.system.replace("{refusal_text}", self.refusal_text)
        return system, user


def list_prompts(prompts_dir: Path = PROMPTS_DIR) -> list[str]:
    return sorted(path.stem for path in prompts_dir.glob("*.yaml"))


def load_prompt(name: str, prompts_dir: Path = PROMPTS_DIR, kind: str | None = None) -> PromptSpec:
    path = prompts_dir / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(list_prompts(prompts_dir)) or "ninguno"
        raise PromptError(f"no existe {path.name}. Disponibles: {available}")
    content = path.read_bytes()
    raw = yaml.safe_load(content.decode("utf-8")) or {}
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise PromptError(f"{path.name}: faltan claves {missing}")
    spec = PromptSpec(
        name=name,
        version=str(raw["version"]),
        kind=str(raw["kind"]),
        description=" ".join(str(raw["description"]).split()),
        system=str(raw["system"]),
        user_template=str(raw["user_template"]),
        sha256=hashlib.sha256(content).hexdigest(),
        refusal_text=" ".join(str(raw.get("refusal_text", "")).split()),
    )
    if spec.kind not in PLACEHOLDERS:
        raise PromptError(f"{path.name}: kind debe ser uno de {sorted(PLACEHOLDERS)}")
    if kind is not None and spec.kind != kind:
        raise PromptError(f"{path.name} es de tipo '{spec.kind}', se esperaba '{kind}'")
    absent = [p for p in PLACEHOLDERS[spec.kind] if p not in spec.user_template]
    if absent:
        raise PromptError(f"{path.name}: el user_template debe contener {absent}")
    if spec.kind == "generation":
        if not spec.refusal_text:
            raise PromptError(f"{path.name}: un prompt de generación necesita refusal_text")
        if "{refusal_text}" not in spec.system:
            raise PromptError(f"{path.name}: el system debe contener {{refusal_text}}")
    return spec
