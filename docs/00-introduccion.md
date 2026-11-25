# Capítulo 0: Introducción

> Un RAG que "parece funcionar" en una demo puede estar recuperando los chunks
> equivocados o inventando datos con buen tono. Este proyecto se construye para poder
> decir cuál de las dos cosas pasa, con números.

---

## 1. El problema

Nubbix, el SaaS de gestión para pymes para el que construimos el triage de M1, llegó a 350 empleados. Las políticas de HR, el manual de IT y los procesos de Finance están dispersos, y People e IT responden todas las semanas las mismas preguntas, no siempre con la política vigente.

**Nubbix Assist** responde esas preguntas a partir de la documentación oficial y **cita siempre los fragmentos que usó**. Si la respuesta no está en la documentación, lo dice.

El corpus son tres documentos Markdown (`data/corpus/`): `hr_policies.md`, `it_manual.md` y `finance_processes.md`, de unos 2,8 KB cada uno, con 6 secciones `##` cada uno.

## 2. El contrato

Cada consulta devuelve un JSON con esta forma (`src/schemas.py`):

```json
{
  "user_question": "¿Hasta qué día del mes puedo cargar una rendición de gastos?",
  "system_answer": "Podés cargar tu rendición de gastos en la plataforma ExpenseHub hasta el día 25 de cada mes [finance_processes.md_0]. ...",
  "chunks_related": [
    {
      "chunk_id": "finance_processes.md_0",
      "source": "finance_processes.md",
      "section": "Reintegro de Gastos y Viáticos",
      "content": "# Procesos Financieros y Reintegros - Nubbix\n## Reintegro de Gastos y Viáticos\n...",
      "score": 0.9919
    }
  ]
}
```

- `chunk_id` es `<documento>_<n>`: la cita se rastrea hasta su archivo de origen sin consultar nada más.
- `chunks_related` contiene **solo los chunks citados** en la respuesta, no todos los recuperados (ADR-045).
- Si el asistente rechaza, `system_answer` es un texto fijo y `chunks_related` queda vacío (ADR-044).
- Los modelos Pydantic usan `extra="forbid"`: un campo de más es un error, no se ignora.

## 3. Principios

1. **Ingesta y consulta separadas.** El índice se construye una vez por comando; consultar nunca indexa. Si el índice no corresponde a la configuración actual, la consulta falla con un mensaje que dice cómo reindexar.
2. **Decisiones con datos y criterio previo.** La estrategia de chunking se elige con un criterio escrito y commiteado antes de la primera medición (ADR-040), sobre un dataset congelado (ADR-048).
3. **Medir por etapas.** Retrieval (¿llegó la evidencia al contexto?) y generación (¿la respuesta se sostiene en lo que citó?) se miden por separado, y cada pregunta fallida se atribuye a una etapa.
4. **Clonar y correr sin claves.** Embeddings locales y proveedor `mock` por defecto: la ingesta, la búsqueda y los evals de retrieval no necesitan ninguna API (heredado de M1, ADR-012).
5. **Todo resultado es trazable.** Cada corrida de evals guarda el sha256 del dataset, del corpus y de los prompts, el commit y si el árbol estaba limpio.

## 4. Arquitectura

Tres procesos independientes que comparten el índice en disco:

```
INGESTA (python -m ingest.ingest --strategy fixed|structural|all)
  data/corpus/*.md ──▶ verificar SHA256SUMS ──▶ chunking ──▶ embeddings (fastembed, local)
                                                   │                 │
                                                   ▼                 ▼
                                      data/chunks/<estrategia>.jsonl   Chroma: rag_fixed | rag_structural
                                      (legible, versionado)            + data/index/bm25_<estrategia>.json

CONSULTA (python -m src.main --query "...")
  pregunta ──▶ vectorial (Chroma, 10 candidatos) ─┐
           └─▶ BM25 (tokens persistidos, 10) ─────┴─▶ RRF ─▶ top 3 ─▶ prompt generation_v1 ─▶ LLM
                                                                                              │
                     JSON {user_question, system_answer, chunks_related} ◀── validar citas ◀──┘
                     data/metrics.jsonl (una línea por consulta, sin el texto de la pregunta)    │ reintento con el motivo

EVALS (python evals/runner.py)
  evals/dataset.jsonl ──▶ retrieval: precision, recall, evidence_recall, hit, MRR (sin LLM)
                     └──▶ generación: respuesta ─▶ juez Sonnet 5.5 por afirmación ─▶ groundedness
                                                                   └─▶ diagnóstico: ok | retrieval | generacion | ambas
```

Estructura del repositorio:

```
config/          providers.yaml (modelos, roles, precios), rag.yaml (chunking, retrieval)
data/corpus/     los tres documentos sin modificar + SHA256SUMS
data/chunks/     export de los chunks de cada estrategia (para leerlos sin correr nada)
data/index/      Chroma y tokens BM25 (fuera de Git, lo genera la ingesta)
data/models/     modelo de embeddings descargado (fuera de Git)
ingest/          loader.py, chunking.py, ingest.py
src/             config, embeddings, vector_store, lexical, retriever, assistant, citations,
                 llm, prompts, schemas, main (CLI), providers/, observability/
prompts/         generation_v1.yaml, judge_groundedness_v1.yaml
evals/           dataset.jsonl, dataset.py, metrics.py, judge.py, runner.py, results/
tests/           64 tests sin red
docs/            este manual
```

## 5. Qué se reutilizó de M1

La abstracción de proveedores (`LLMProvider` con OpenAI, Anthropic y mock), el cálculo de costo por tokens, la carga de configuración desde `.env` y YAML, el reintento con el error de validación en un solo lugar y la trazabilidad de los resultados. El detalle está en el [Capítulo 1](01-fase-0-repositorio.md).
