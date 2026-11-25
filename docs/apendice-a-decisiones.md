# Apéndice A — Registro de decisiones (ADR)

> Cada decisión técnica con su contexto, alternativas y consecuencias.
> Formato: **ADR-NNN**. Una decisión no se borra: se reemplaza por otra ADR.
> La numeración continúa la del Proyecto Integrador M1 (ADR-001 a ADR-033).

---

## Decisiones heredadas de M1

Siguen vigentes en este repositorio. El texto completo está en el
[Apéndice A de M1](https://github.com/davidmenlop/M1/blob/master/docs/apendice-a-decisiones.md).

| ADR | Decisión | Dónde se ve en M2 |
|---|---|---|
| 001–003 | Python 3.14, `uv`, `requirements.txt` con pins exactos | `.python-version`, `requirements.txt` |
| 009, 012 | Proveedor `mock` por defecto: clonar y correr sin claves | `config/providers.yaml`, `src/providers/mock_provider.py` |
| 013 | La consulta del usuario no se guarda en texto plano, solo su sha256 | `src/observability/metrics.py` |
| 014, 015 | Reintento con el error de validación, en un solo lugar; SDK con `max_retries=0` | `src/llm.py` |
| 016 | Anthropic sin `temperature` (la API la rechaza desde 2026-10-21) | `config/providers.yaml` |
| 020 | Prompts en YAML, inmutables: un cambio es una versión nueva | `prompts/` |
| 022 | Criterio de selección declarado antes de medir | ADR-040 |
| 026 | Resultados de evals con hashes y commit | `evals/results/` |

---

## ADR-034 — Repositorio nuevo, con componentes de M1 copiados

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** la consigna de M2 pide rutas (`evals/dataset.jsonl`, `evals/runner.py`, `src/`, `prompts/`) que en M1 ya existen con otro contenido (triage). M2 se entrega como repositorio propio.
- **Decisión:** repositorio nuevo con historial propio. Se copian de M1 la abstracción de proveedores, el cálculo de costo, la carga de configuración y el patrón de reintento con el error. Se descarta lo propio del triage (contrato, guardrails, caché, dataset).
- **Alternativas:** M2 como subcarpeta de M1 (choca con las rutas pedidas); M1 como dependencia instalable (acopla dos entregas que se corrigen por separado).
- **Consecuencias:** el repositorio es autocontenido. Una mejora en los proveedores de M1 no llega sola a M2.

## ADR-035 — Corpus sin modificar, verificado por hash

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** la consigna exige los tres documentos sin modificar. Además, el golden dataset ubica su evidencia por posición de caracteres: un cambio mínimo la invalida.
- **Decisión:** `data/corpus/SHA256SUMS` guarda el hash de cada documento; la ingesta falla si alguno no coincide. `.gitattributes` normaliza a LF y los archivos ya vienen en LF, así que Git no los altera.
- **Consecuencias:** un corpus alterado se detecta antes de indexar. Actualizar el corpus exige regenerar los hashes y reetiquetar el dataset a propósito.

## ADR-036 — Embeddings locales multilingües (fastembed)

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** el corpus y las preguntas están en español rioplatense, con sinónimos coloquiales ("plata", "compu"). Ingesta y evaluación de retrieval tendrían que correr sin claves (ADR-012 de M1).
- **Decisión:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` vía `fastembed` (ONNX), 384 dimensiones, ~220 MB, descargado una vez a `data/models/`. El mismo modelo indexa y consulta; su nombre queda en los metadatos de la colección y la consulta se niega a usar otro.
- **Alternativas:** `multilingual-e5-small` (la opción recomendada en el plan; `fastembed` no la ofrece); `multilingual-e5-large` (2,2 GB); `text-embedding-3-small` de OpenAI (mejor calidad probable, pero exige clave y red para indexar); el modelo por defecto de Chroma, `all-MiniLM-L6-v2` (entrenado en inglés).
- **Consecuencias:** ingesta y retrieval sin claves ni costo. La primera ingesta necesita red para descargar el modelo. Es un modelo chico: la búsqueda léxica compensa sus fallos con términos exactos (ADR-038).

## ADR-037 — Chroma local persistente, una colección por estrategia

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** hay que comparar dos estrategias sobre el mismo corpus y consultar sin reindexar.
- **Decisión:** `chromadb.PersistentClient` en `data/index/chroma/` (fuera de Git), distancia coseno, colecciones `rag_fixed` y `rag_structural`. La ingesta borra y recrea la colección de la estrategia; nunca agrega sobre un índice viejo. La colección guarda modelo, parámetros de chunking (con su sha256) y hash del corpus; la consulta falla si no coinciden con la configuración actual.
- **Alternativas:** una sola colección con la estrategia como metadato (un filtro olvidado mezcla estrategias); base gestionada (Pinecone, Qdrant Cloud: credenciales y red para 53 chunks).
- **Consecuencias:** consultar no indexa. Un cambio de parámetros obliga a reindexar, y el sistema lo avisa en vez de responder con un índice viejo.

## ADR-038 — BM25 con normalización para español, tokens persistidos

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** la búsqueda vectorial falla con términos exactos (canales de Slack, montos, nombres de plataformas). Chroma no trae BM25 en modo local.
- **Decisión:** `rank_bm25.BM25Okapi`. Los textos se pasan a minúsculas y sin tildes, se quitan palabras vacías de una lista corta y se aplica stemming Snowball en español (`licencias` → `licenci`). Los tokens se guardan en `data/index/bm25_<estrategia>.json` en la ingesta; la consulta solo los lee.
- **Alternativas:** BM25 sobre texto crudo (sin stemming, "licencia" no encuentra "licencias"); búsqueda full-text de SQLite FTS5 (no trae stemming en español).
- **Consecuencias:** la consulta no reprocesa el corpus. La lista de palabras vacías es manual.

## ADR-039 — Fusión híbrida con Reciprocal Rank Fusion (k = 60)

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** la similitud coseno (0–1) y BM25 (0–∞) están en escalas distintas; sumarlas exige normalizar y calibrar un peso.
- **Decisión:** RRF: cada buscador aporta 10 candidatos y cada chunk suma `1 / (60 + posición)` por ranking. El `score` del contrato es el RRF dividido por su máximo posible (`2 / 61`), en [0, 1]. `k = 60` es el valor del paper original (Cormack et al., 2009); no se ajustó.
- **Alternativas:** suma ponderada de puntajes normalizados (exige elegir un peso α, y con 49 preguntas se sobreajusta); solo vectorial o solo léxica (medidas como línea base).
- **Consecuencias:** sin hiperparámetros que calibrar. Con un corpus de 18–35 chunks casi todos los chunks aparecen en ambos rankings, así que el `score` híbrido varía poco (0,9–1,0): sirve para ordenar, no como medida de confianza.

## ADR-040 — Criterio de decisión de chunking, fijado antes de medir

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** la elección de estrategia tiene que salir de métricas de retrieval y no acomodarse a los números después de verlos (ADR-022 de M1).
- **Decisión:** se comparan `fixed` y `structural` con el mismo dataset, el mismo modo (`hybrid`, el de producción) y el mismo `top_k = 3`. Orden de criterios, con diferencias menores a 0,03 tratadas como empate:
  1. **evidence_recall@3**: ¿la evidencia llega al contexto del modelo? Si no llega, la generación no puede responder bien.
  2. **MRR**: ¿la evidencia aparece primero?
  3. **precision@3**: cuánto ruido acompaña a la evidencia.
  4. **Tamaño del contexto** (caracteres que recibe el modelo): menos es más barato.

  `top_k = 3` se eligió antes de medir: con 18 chunks en `structural`, k = 5 ya trae más de un cuarto del corpus y las estrategias dejan de diferenciarse. Se reportan también k = 1 y k = 5 y los modos `vector` y `bm25`, como contexto.
- **Consecuencias:** la decisión es reproducible con un comando. La generación (groundedness) no decide la estrategia: se reporta para confirmar que el retrieval elegido sostiene buenas respuestas.

## ADR-041 — Evidencia literal y relevancia por solapamiento

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** los chunks esperados dependen de la estrategia: el mismo dato cae en chunks distintos según cómo se corte. Etiquetar ids a mano para cada estrategia duplica el trabajo y abre la puerta a etiquetar distinto.
- **Decisión:** cada pregunta guarda su evidencia como fragmentos literales del corpus (`source`, `quote`), validados como únicos en su documento. `evals/dataset.py --write` calcula los `expected_chunks` de cada estrategia con una sola regla: un chunk es relevante si su tramo se solapa con el de alguna evidencia en al menos `min(30, largo de la evidencia)` caracteres. `--check` (y el runner) fallan si las etiquetas quedaron desactualizadas.
- **Alternativas:** ids de chunk etiquetados a mano por estrategia; relevancia por similitud semántica (circular: usa el mismo tipo de modelo que se evalúa).
- **Consecuencias:** la misma fórmula para ambas estrategias; el dataset declara qué chunks espera, como pide la consigna. Un cambio de parámetros de chunking reetiqueta solo.

## ADR-042 — evidence_recall además del recall a nivel chunk

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** `fixed` usa 60 caracteres de solapamiento: un dato corto puede quedar completo en dos chunks vecinos. El recall a nivel chunk exige recuperar ambos aunque uno ya alcance, y castiga a `fixed` por un artefacto de la regla, no del retrieval.
- **Decisión:** se reportan ambos. **recall@k** = chunks relevantes recuperados / esperados (definición clásica). **evidence_recall@k** = evidencias cubiertas / evidencias, donde una evidencia está cubierta si los chunks recuperados cubren al menos el 80 % de sus caracteres. Para decidir se usa evidence_recall (ADR-040).
- **Consecuencias:** la métrica principal mide lo que importa (si el dato llega al modelo) y es comparable entre estrategias. Hay que explicar dos recalls en el README.

## ADR-043 — Juez de groundedness: otro modelo, afirmación por afirmación

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** un modelo que se califica a sí mismo tiende a aprobarse. Un puntaje global 1–5 del juez es difícil de auditar.
- **Decisión:** el juez es Claude Sonnet 5.5 (`effort: medium`) y el generador, Claude Haiku 5.5. El juez descompone la respuesta en afirmaciones atómicas y marca cada una como sostenida o no por los chunks **citados** (no por todos los recuperados). El puntaje lo calcula el código: sostenidas / total. Las respuestas de rechazo no se juzgan.
- **Alternativas:** el mismo Haiku 5.5 como juez (20 veces más barato, sesgo de autoevaluación); un puntaje global del juez; métricas léxicas de solapamiento (no detectan un número cambiado).
- **Consecuencias:** cada afirmación no sostenida queda guardada con su motivo y se puede revisar. Costo del juez: unos USD 0,005 por respuesta. El juez no se calibró contra etiquetas humanas: es una limitación conocida.

## ADR-044 — Rechazo honesto con un texto fijo

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** el bonus pide medir automáticamente si el asistente reconoce que no sabe.
- **Decisión:** el prompt define un texto de rechazo exacto (`refusal_text` en el YAML). Se detecta por prefijo normalizado; en ese caso `chunks_related` queda vacío. Una respuesta parcial ("esto sí, esto no figura") no es un rechazo: lleva citas.
- **Alternativas:** clasificar el rechazo con el juez (otra llamada y otro error posible); umbral de score de retrieval (con RRF el score no discrimina, ADR-039).
- **Consecuencias:** el rechazo se mide con una comparación de texto. Si el modelo parafrasea el rechazo, cuenta como respuesta sin citas y se reintenta.

## ADR-045 — `chunks_related` solo con chunks citados, y citas validadas

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** la consigna define `chunks_related` como "los chunks usados para responder". Devolver todos los recuperados presentaría como fuente chunks que el modelo no usó.
- **Decisión:** el modelo cita inline `[chunk_id]`. El pipeline extrae las citas, rechaza ids que no se recuperaron en esa consulta y respuestas sin citas, y reintenta con el motivo. `chunks_related` = recuperados citados, en orden de recuperación, con `{chunk_id, source, section, content, score}`.
- **Consecuencias:** toda cita apunta a un chunk real del índice. Si se agotan los reintentos se responde un fallback sin citas, que se cuenta aparte.

## ADR-046 — Generador: Claude Haiku 5.5 con `effort: low`

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** la tarea es redactar 1–4 oraciones a partir de 3 chunks cortos. En M1, Haiku 5.5 fue el modelo más barato y el único que resistió los 10 ataques.
- **Decisión:** Haiku 5.5, `effort: low`, `max_tokens: 3000` (el thinking consume parte). `gpt-4.1-mini` queda configurado como alternativa.
- **Consecuencias:** unos USD 0,0002 por consulta. No se compararon generadores: el foco del módulo es el retrieval.

## ADR-047 — JSON de salida en UTF-8

- **Fecha:** 2026-11-25
- **Estado:** aceptada (reemplaza ADR-008 de M1 para este proyecto)
- **Contexto:** M1 escapaba los acentos para evitar problemas de consola en Windows. El formato de ejemplo de la consigna muestra texto con acentos.
- **Decisión:** `ensure_ascii=False` y `sys.stdout` reconfigurado a UTF-8.
- **Consecuencias:** JSON legible y válido para cualquier consumidor UTF-8.

## ADR-048 — Dataset congelado antes de la primera medición

- **Fecha:** 2026-11-25
- **Estado:** aceptada
- **Contexto:** si el dataset se ajusta después de ver qué estrategia gana, la comparación pierde valor.
- **Decisión:** el dataset (49 preguntas) y este criterio se commitean antes de correr el runner por primera vez. Un cambio posterior del dataset se documenta como corrección, con su motivo, y las corridas viejas dejan de compararse (el runner las filtra por sha256 del dataset).
- **Consecuencias:** la elección es auditable en el historial de Git. Las preguntas las redactó un asistente de IA y falta la revisión humana de cada etiqueta (ver limitaciones).
