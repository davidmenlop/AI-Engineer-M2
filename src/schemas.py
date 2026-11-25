"""Contrato de salida del asistente: {user_question, system_answer, chunks_related}."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChunkRelated(BaseModel):
    """Un chunk usado para responder. ``source`` identifica el documento de origen."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, description="'<documento>_<n>', p. ej. 'hr_policies.md_0'")
    source: str = Field(min_length=1, description="archivo de data/corpus/ del que sale el chunk")
    section: str = Field(description="sección del documento donde empieza el chunk")
    content: str = Field(min_length=1, description="texto completo del chunk")
    score: float = Field(description="puntaje de recuperación; su escala depende del modo")


class AssistResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_question: str = Field(min_length=1)
    system_answer: str = Field(min_length=1)
    chunks_related: list[ChunkRelated] = Field(
        description="solo los chunks citados en la respuesta, en orden de recuperación; vacío si rechaza"
    )
