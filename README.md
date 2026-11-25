# Nubbix Assist: asistente RAG de conocimiento interno (Proyecto Integrador M2)

Chatbot que responde preguntas de los empleados de Nubbix con la documentación oficial
de HR, IT y Finance, y cita los chunks que usó. Cada respuesta sale en JSON con el
contrato `{user_question, system_answer, chunks_related}`.

| | Resultado (top_k = 3, búsqueda híbrida) |
|---|---|
| Chunking elegido con datos | **structural** (una sección `##` por chunk): evidence_recall **0.982** contra 0.802 de fixed |
| Retrieval | hit@3 **1.000**, MRR **0.897**, recall@3 0.980, precision@3 0.373 (42 preguntas) |
| Búsqueda híbrida | supera a vectorial (0.923) y a BM25 (0.893) en evidence_recall, en ambas estrategias |
| Groundedness | **0.971** (juez Claude Sonnet 5.5, afirmación por afirmación); 90,5 % de respuestas 100 % sostenidas |
| Rechazo honesto (bonus) | **7/7** preguntas fuera de alcance rechazadas; **0** falsos rechazos en 42 |
| Costo | USD 0.0002 por consulta (Claude Haiku 5.5); embeddings locales, sin costo |
| Reproducible | ingesta y evals de retrieval sin claves; 64 tests sin red |

El proceso completo (decisiones, alternativas y errores) está en el
[manual de construcción](docs/README.md), un capítulo por fase, y en las
[decisiones técnicas (ADR)](docs/apendice-a-decisiones.md).

---

## Inicio rápido

```bash
uv venv --python 3.14
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
cp .env.example .env               # Windows: Copy-Item .env.example .env

# 1. Ingesta: de cero al índice, una vez por estrategia (sin claves)
python -m ingest.ingest --strategy all

# 2. Consulta (sin claves: proveedor mock extractivo, solo para probar la instalación)
python -m src.main --query "¿Hasta qué día del mes puedo cargar una rendición de gastos?" --pretty

# 3. Evals de retrieval: ambas estrategias, 3 modos (sin claves)
python evals/runner.py --no-save
```

La primera ingesta descarga el modelo de embeddings (~220 MB) a `data/models/`. El
índice queda en `data/index/`. Las dos carpetas están fuera de Git y la consulta no
vuelve a indexar.

## Requisitos y configuración

- Python 3.14 (fijado en `.python-version`) y [`uv`](https://docs.astral.sh/uv/).
  También funciona `python -m venv .venv && pip install -r requirements.txt`.
- Para generar respuestas con un LLM real hace falta `ANTHROPIC_API_KEY` en `.env`
  (o `OPENAI_API_KEY`, alternativa). Las claves solo se leen de variables de entorno.

| Variable | Default | Para qué |
|---|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | vacío | generación y juez de groundedness |
| `DEFAULT_PROVIDER` | `mock` | `mock`, `anthropic` u `openai` |
| `MAX_RETRIES` / `REQUEST_TIMEOUT_S` | `2` / `60` | reintentos ante respuesta inválida o error transitorio |
| `INDEX_DIR` / `MODELS_DIR` | `data/index` / `data/models` | índice persistido y caché del modelo de embeddings |
| `METRICS_PATH` / `LOG_LEVEL` | `data/metrics.jsonl` / `INFO` | observabilidad |

Parámetros del RAG (chunking, modelo de embeddings, `top_k`, RRF):
`config/rag.yaml`. Modelos LLM, precios y roles (generador, juez):
`config/providers.yaml`.

## Uso

```bash
# Respuesta con el LLM (requiere ANTHROPIC_API_KEY)
python -m src.main --provider anthropic --pretty \
  --query "¿Hasta qué día del mes puedo cargar una rendición de gastos?"
```

stdout (JSON del contrato, UTF-8):

```json
{
  "user_question": "¿Hasta qué día del mes puedo cargar una rendición de gastos?",
  "system_answer": "Podés cargar tu rendición de gastos en la plataforma ExpenseHub hasta el día 25 de cada mes [finance_processes.md_0]. Si la cargás después del día 25, se procesará en el ciclo de pagos del mes siguiente [finance_processes.md_0].",
  "chunks_related": [
    {
      "chunk_id": "finance_processes.md_0",
      "source": "finance_processes.md",
      "section": "Reintegro de Gastos y Viáticos",
      "content": "# Procesos Financieros y Reintegros - Nubbix\n## Reintegro de Gastos y Viáticos\nLos gastos operativos, viajes corporativos o comidas con clientes pueden ser reembolsados cumpliendo las siguientes condiciones:\n- Todos los comprobantes deben ser facturas A o B a nombre de Nubbix S.A.\n- El monto máximo diario permitido para almuerzo/cena corporativa en viajes es de $25,000 ARS.\n- Las rendiciones deben ser cargadas en la plataforma ExpenseHub antes del día 25 de cada mes.\n- Reintegros ingresados después del día 25 se procesarán en el ciclo de pagos del mes siguiente.",
      "score": 0.9919
    }
  ]
}
```

stderr: `INFO assist: request=3ccf2409 structural/hybrid k=3 provider=anthropic model=claude-haiku-5-5 attempts=1 refused=False cited=['finance_processes.md_0'] cost_usd=0.00020430 retrieval_ms=54.2 latency_s=1.072`

- `chunk_id` = `<documento>_<n>`: la cita se rastrea hasta su archivo en `data/corpus/`.
  Los chunks de cada estrategia se pueden leer en `data/chunks/<estrategia>.jsonl`.
- `chunks_related` contiene **solo los chunks citados** en la respuesta, no todos los
  recuperados (ADR-045).
- `score` es el RRF normalizado a [0, 1] en modo híbrido, la similitud coseno en modo
  `vector` y el puntaje BM25 en modo `bm25`.

Opciones:

```bash
python -m src.main --query "..." --strategy fixed          # otra colección
python -m src.main --query "..." --mode vector             # hybrid | vector | bm25
python -m src.main --query "..." --top-k 5
python -m src.main --query "..." --retrieve-only           # solo los chunks, sin LLM
python -m src.main --query "..." --provider openai         # gpt-4.1-mini
echo "¿Qué día cobro el sueldo?" | python -m src.main --query -
```

Códigos de salida: `0` ok; `2` error de configuración, índice faltante o
desactualizado, o clave inválida, con el motivo en stderr.

## Arquitectura

```
INGESTA (una vez por estrategia)                     CONSULTA (cada pregunta)
python -m ingest.ingest --strategy X                 python -m src.main --query "..."

data/corpus/*.md ─ verificación SHA256SUMS            pregunta ─┬─ embedding ─▶ Chroma (top 10) ─┐
      │                                                         └─ tokens ───▶ BM25   (top 10) ─┤
   chunking (fixed | structural)                                                                ▼
      │                                                                           RRF k=60 ─▶ top 3 chunks
      ├─ embeddings (MiniLM multilingüe, 384 d) ─▶ Chroma: colección rag_<X>               │
      ├─ tokens BM25 ──────────────────────────▶ data/index/bm25_<X>.json        prompt generation_v1
      └─ export legible ───────────────────────▶ data/chunks/<X>.jsonl                     │
                                                                              Claude Haiku 5.5 ─▶ validar citas
                                                                                     (reintento si no)
                                                                                           │
                                                                   JSON {user_question, system_answer, chunks_related}
```

```
data/corpus/       los 3 documentos provistos, sin modificar, + SHA256SUMS
data/chunks/       los chunks de cada estrategia en JSONL (para leerlos sin correr nada)
config/            rag.yaml (chunking, embeddings, retrieval), providers.yaml (LLMs, precios)
ingest/            loader.py (corpus), chunking.py (2 estrategias), ingest.py (comando)
src/               embeddings, vector_store (Chroma), lexical (BM25), retriever (RRF),
                   assistant (generación + citas), main.py (CLI), providers/ (de M1)
prompts/           generation_v1.yaml, judge_groundedness_v1.yaml
evals/             dataset.jsonl, dataset.py (validación y etiquetado), metrics.py,
                   judge.py, runner.py, results/
tests/             64 tests sin red
docs/              manual de construcción por fase y ADR
```

## Corpus

Tres documentos Markdown provistos por Henry, en `data/corpus/` sin modificar. La
ingesta falla si alguno no coincide con `data/corpus/SHA256SUMS` (ADR-035).

| Documento | Contenido | Secciones `##` |
|---|---|---|
| `hr_policies.md` | licencias, vacaciones, home office, enfermedad, cumpleaños, feriados, beneficios | 6 |
| `it_manual.md` | accesos, equipos, contraseñas y MFA, seguridad de datos, soporte, offboarding | 6 |
| `finance_processes.md` | reintegros y viáticos, compras, sueldos, capacitación, bonos, facturación | 6 |

Son ~8 KB en total, con secciones cortas (285–534 caracteres). Algunos temas cruzan
documentos: el crédito de capacitación (HR → su rendición en Finance) y el robo de un
equipo en viaje (IT Equipos + IT Soporte + Finance Reintegros).

## Chunking: dos estrategias y la decisión

| | `fixed` | `structural` |
|---|---|---|
| Regla | ventana de 300 caracteres con 60 de solapamiento, cortada en límite de palabra; ignora la estructura | un chunk por sección `##`, con el título del documento y de la sección antepuestos; subdivide por línea si supera 1.200 caracteres |
| Chunks | **35** (108–299 caracteres, promedio 285) | **18** (314–598, promedio 486) |
| Riesgo | corta listas y mezcla el final de una sección con la siguiente | chunks más largos: más texto por consulta |

Son estrategias realmente distintas: una ignora la estructura y otra se apoya en ella.
`fixed` cumple el mínimo de 20 chunks de la consigna.

**Criterio fijado antes de medir** (ADR-040, commit previo a la primera corrida):
mismo dataset, modo `hybrid` y `top_k = 3`. Se ordena por **evidence_recall@3**,
después MRR, después precision@3, después tamaño del contexto. Las diferencias
menores a 0,03 cuentan como empate.

Comparación (`python evals/runner.py --k 1 3 5`, 42 preguntas dentro de alcance,
`evals/results/retrieval_20261125T164259Z.json`):

| estrategia | modo | k | precision@k | recall@k | evidence_recall@k | complete@k | hit@k | MRR |
|---|---|---|---|---|---|---|---|---|
| fixed | hybrid | 3 | **0.452** | 0.793 | 0.802 | 0.738 | 0.929 | 0.762 |
| **structural** | **hybrid** | **3** | 0.373 | **0.980** | **0.982** | **0.952** | **1.000** | **0.897** |
| fixed | hybrid | 1 | 0.595 | 0.411 | 0.476 | 0.476 | 0.595 | 0.595 |
| structural | hybrid | 1 | 0.809 | 0.742 | 0.740 | 0.691 | 0.809 | 0.809 |
| fixed | hybrid | 5 | 0.324 | 0.899 | 0.881 | 0.857 | 0.952 | 0.768 |
| structural | hybrid | 5 | 0.229 | 0.992 | 0.994 | 0.976 | 1.000 | 0.897 |

**Decisión: `structural`** (ADR-049). Supera a `fixed` en evidence_recall@3 por 0,180,
muy por encima del umbral de empate, y también gana en MRR.

Trade-offs que se aceptan:

- **`fixed` tiene mejor precision@3** (0.452 contra 0.373): sus chunks son más
  chicos y cada pregunta tiene en promedio 1,83 chunks relevantes, contra 1,17 de
  `structural`. Con una sola evidencia por pregunta, precision@3 de `structural` tiene
  un techo cercano a 0,33.
- **`fixed` le pasa menos texto al modelo**: unos 855 caracteres con k = 3, contra
  1.458 de `structural`. En este corpus la diferencia de costo es despreciable.
- **Dónde pierde `fixed`:** en las preguntas que cruzan documentos (0.633 contra
  0.850), en los distractores (0.714 contra 1.000) y en los sinónimos (0.714 contra
  1.000). Un chunk que corta una lista o mezcla dos secciones se parece menos a la
  pregunta.
- La ventaja de `structural` depende de que las secciones sean cortas y temáticas. En
  documentos con secciones largas, la subdivisión por línea lo acercaría a `fixed`.

Qué métrica es cuál (detalle en `evals/metrics.py`):

- **precision@k** = chunks relevantes recuperados / k.
- **recall@k** = chunks relevantes recuperados / chunks relevantes esperados.
- **evidence_recall@k** = evidencias cubiertas / evidencias. Una evidencia está cubierta
  si los chunks recuperados contienen al menos el 80 % de su texto. Es la métrica que
  decide porque no castiga a `fixed` cuando un dato corto queda completo en dos chunks
  solapados y se recupera uno solo (ADR-042).
- **complete@k**: todas las evidencias de la pregunta cubiertas. **hit@k**: al menos un
  chunk relevante. **MRR**: 1 / posición del primer chunk relevante.

## Búsqueda híbrida

La búsqueda **vectorial** consulta Chroma, con similitud coseno sobre embeddings locales.
La **léxica** usa BM25 sobre texto normalizado: minúsculas, sin tildes, sin palabras
vacías y con stemming en español. Cada una aporta 10 candidatos y se fusionan con
**Reciprocal Rank Fusion**: cada chunk suma `1 / (60 + posición)` por ranking (ADR-039).
RRF combina posiciones y no puntajes, así que no hace falta calibrar la escala del
coseno (0–1) contra la de BM25 (0–∞). El parámetro `k = 60` es el del paper original
y no se ajustó.

evidence_recall@3 por modo y tipo de pregunta (`structural`):

| modo | total | directa | sinónimos | término exacto | distractor | cruza documentos |
|---|---|---|---|---|---|---|
| vector | 0.923 | 1.000 | 0.857 | 1.000 | 1.000 | 0.750 |
| bm25 | 0.893 | 1.000 | 0.786 | 1.000 | 0.857 | 0.900 |
| **hybrid** | **0.982** | 1.000 | **1.000** | 1.000 | 1.000 | 0.850 |

Casos concretos (`structural`, k = 3) que muestran que la hibridación funciona:

- **La vectorial rescata sinónimos que BM25 pierde.** En q07 ("¿un monto fijo por
  mes para pagar la conexión?") el documento dice *estipendio de conectividad*. En
  q16, "obra social" corresponde a *medicina prepaga*; en q21, "laptop" a *equipo*; en
  q24, "clave" a *credenciales*. BM25 no recupera la evidencia en ninguno de los
  cuatro y la búsqueda híbrida sí.
- **BM25 rescata términos que la vectorial pierde.** En q12 ("¿me descuentan ese día
  del sueldo?") y en q25 ("bar con wifi abierto" → *redes Wi-Fi públicas*), la
  vectorial no recupera la evidencia y BM25 sí.
- **Dónde la fusión todavía cuesta:** en q29 (robo de notebook en viaje, 3 secciones
  esperadas) BM25 recupera las tres. La vectorial, en cambio, solo trae chunks de
  Finance y uno de Home Office (`hr_policies.md_1`). Ese chunk aparece en ambos
  rankings (posiciones 4 y 10) y RRF lo sube al tercer lugar. `it_manual.md_4`
  (Soporte, la sección que menciona el reintegro en viaje) es el primero en BM25, pero
  no entra en el top 10 vectorial y RRF le da la mitad del puntaje. Es el costo de no
  ponderar los buscadores (ADR-039).

## Evaluación

```bash
python evals/dataset.py --check                         # valida el dataset y sus etiquetas
python evals/runner.py                                  # retrieval: 2 estrategias x 3 modos (sin claves)
python evals/runner.py --k 1 3 5                        # barrido de top_k
python evals/runner.py --generation --strategy structural --provider anthropic   # generación + juez
python evals/runner.py --generation --strategy fixed --provider anthropic
python evals/runner.py --generation --limit 5 --no-save                          # prueba rápida
python evals/runner.py --compare evals/results/*.json   # tablas markdown de corridas guardadas
```

El runner es un script, no un framework de testing. Mide retrieval y generación **por
separado**, guarda cada corrida en `evals/results/` con el sha256 del dataset y del
corpus, los parámetros y el commit, y `--compare` ignora corridas hechas con otra
versión del dataset.

### Golden dataset

`evals/dataset.jsonl`: **49 preguntas**, 42 dentro de alcance y 7 fuera de alcance.

| tipo | n | ejemplo |
|---|---|---|
| directa | 11 | "¿Cada cuánto se hacen los backups de las bases productivas y por cuánto tiempo se guardan?" |
| sinónimos | 14 | "¿La obra social también cubre a mis hijos?" (el documento dice *medicina prepaga*) |
| término exacto | 5 | "¿En qué canal de Slack se celebran los aniversarios?" |
| distractor | 7 | "Me caso el mes que viene, ¿cuántos días de licencia tengo?" (convive con nacimiento, fallecimiento y estudio) |
| cruza documentos | 5 | "Un curso me cuesta USD 700: ¿me lo cubre el crédito de capacitación entero y quién lo aprueba?" (HR + Finance) |
| fuera de alcance | 7 | "¿Cuántos días de licencia me dan por mudanza?" (hay licencias, pero no esa) |

Cada pregunta trae su **evidencia como fragmentos literales del corpus**. Los chunks
esperados de cada estrategia (`expected_chunks`) se calculan con una sola regla: un
chunk es relevante si se solapa con alguna evidencia en al menos 30 caracteres, o en
toda la evidencia si es más corta (ADR-041). Así las dos estrategias se miden con la
misma fórmula. Si cambian los parámetros de chunking, `python evals/dataset.py --write`
reetiqueta, y el runner se niega a medir con etiquetas desactualizadas. El dataset se
commiteó antes de la primera medición (ADR-048).

### Resultados de generación

Generador Claude Haiku 5.5 (`prompts/generation_v1.yaml`); juez Claude Sonnet 5.5
(`prompts/judge_groundedness_v1.yaml`). El juez divide cada respuesta en afirmaciones
atómicas y marca si los chunks **citados** sostienen cada una. El puntaje lo calcula el
código: sostenidas / total (ADR-043).

| estrategia | groundedness | 100 % sostenidas | citas a chunks relevantes | rechazo correcto | falso rechazo | retrieval completo | costo (gen + juez) |
|---|---|---|---|---|---|---|---|
| **structural** | **0.971** | **90,5 %** | **0.988** | **7/7** | **0/42** | 0.952 | USD 0.011 + 0.222 |
| fixed | 0.919 | 82,5 % | 0.933 | 7/7 | 2/42 | 0.738 | USD 0.011 + 0.202 |

Archivos: `evals/results/generation_20261125T164613Z.json` (structural) y
`generation_20261125T164923Z.json` (fixed). Latencia de generación p50 1,1 s, p95 1,8 s
(structural).

**Diagnóstico por etapa** (structural, 42 preguntas dentro de alcance): 36 correctas en
ambas etapas, 2 con falla de retrieval y 4 con falla de generación.

- **Retrieval (q29, q42):** preguntas que cruzan documentos en las que no llegó toda la
  evidencia. El modelo no inventó la parte faltante: dijo explícitamente que la
  documentación recuperada no la cubría (groundedness 1.0 en ambas).
- **Generación (q11, q12, q24, q41):** generalizaciones chicas que el juez marcó como no
  sostenidas. En q11 escribió "antes de las 10 AM *del primer día*" cuando el texto dice
  "del día de la ausencia". En q41, "*solo* para cursos de tecnología" cuando el texto
  no dice "solo". El caso q24 es discutible: el juez no acepta deducir que "mensaje
  privado" incluye Slack.

Con `fixed`, 7 preguntas fallan en ambas etapas y hay 2 falsos rechazos (q12, q20) en
preguntas cuya evidencia no llegó al contexto. Es la confirmación del lado de la
generación de que el retrieval era el problema.

## Rechazo honesto (bonus)

Si la respuesta no está en los chunks recuperados, el prompt obliga a responder un texto
fijo (`refusal_text` en `prompts/generation_v1.yaml`) con `chunks_related: []`. Es una
frase fija para poder detectarla con una comparación de texto (ADR-044). Las 7
preguntas fuera de alcance del dataset miden si rechaza cuando corresponde, y las 42
dentro de alcance miden si rechaza cuando no debería.

```bash
python -m src.main --provider anthropic --query "¿Nubbix tiene estacionamiento para empleados en la oficina?"
```

```json
{"user_question": "¿Nubbix tiene estacionamiento para empleados en la oficina?", "system_answer": "No encontré esa información en la documentación interna de Nubbix. Te sugiero consultarlo con People, IT o Finance según corresponda.", "chunks_related": []}
```

Resultado: **7/7 rechazos correctos y 0/42 falsos rechazos** con `structural`. Esto
incluye preguntas con distractores cercanos, como licencia por mudanza (hay otras
licencias), aguinaldo (hay bonos) y tarjeta de acceso al edificio (hay pérdida de
equipos). Una respuesta parcial ("esto sí figura, esto no") no cuenta como rechazo y
lleva sus citas.

## Decisiones técnicas

| Tema | Decisión | ADR |
|---|---|---|
| Modelo de embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (fastembed/ONNX, local, 384 dimensiones, español incluido). El mismo modelo indexa y consulta; la consulta falla si el índice se construyó con otro | 036 |
| Vector DB | Chroma local persistente (`data/index/chroma`), **distancia coseno**, una colección por estrategia, recreada desde cero en cada ingesta | 037 |
| Búsqueda léxica | BM25 (`rank_bm25`) con normalización y stemming Snowball en español; tokens persistidos en la ingesta | 038 |
| Fusión híbrida | Reciprocal Rank Fusion, k = 60, 10 candidatos por buscador, `top_k = 3` | 039 |
| Criterio de chunking | fijado y commiteado antes de medir | 040, 049 |
| Etiquetado | evidencia literal; chunks esperados por solapamiento, misma regla para ambas estrategias | 041, 042 |
| Generación | Claude Haiku 5.5, `effort: low`; citas inline validadas contra los chunks recuperados, con reintento | 045, 046 |
| Juez | Claude Sonnet 5.5, distinto del generador, por afirmación; puntaje calculado por código | 043 |
| Heredado de M1 | proveedores intercambiables, reintento en un solo lugar, prompts YAML inmutables, `mock` por defecto | tabla en el Apéndice A |

## Observabilidad

Cada consulta de la CLI agrega una línea a `data/metrics.jsonl`. La línea registra
estrategia, modo, ids recuperados y citados, si rechazó, intentos, tokens, costo y
latencias de retrieval y total. La pregunta no se guarda en texto plano: solo su sha256
y su longitud. El runner de evals no escribe ahí; guarda sus resultados en
`evals/results/`.

## Limitaciones conocidas

- **Corpus y dataset chicos.** Con 18 secciones, k = 3 trae un sexto del corpus y las
  métricas de retrieval son optimistas. Con un corpus más grande, la diferencia entre
  estrategias y modos se acentuaría.
- **Dataset redactado con asistencia de IA.** Las 49 preguntas y sus evidencias se
  redactaron con Claude y se validaron automáticamente: cada evidencia existe literal y
  una sola vez en el corpus. **Falta la revisión humana etiqueta por etiqueta** que pide
  la guía. Es el próximo paso antes de considerar finales estos números.
- **El juez no está calibrado contra personas.** Es estricto con las generalizaciones
  (q24 es discutible) y nadie midió cuánto acuerda con una revisión humana. Una
  corrida por estrategia: la varianza del generador y del juez no está medida.
- **Preguntas que cruzan 3 documentos** (q29) no entran completas con k = 3. Un `top_k`
  mayor o re-ranking (extra credit) lo mejorarían, con más contexto.
- **El `score` híbrido casi no discrimina** (0,9–1,0) porque con tan pocos chunks casi
  todos aparecen en ambos rankings. Sirve para ordenar, no como confianza; por eso el
  rechazo lo decide el modelo y no un umbral de score.
- **La primera ingesta necesita red** para descargar el modelo de embeddings.

## Uso de IA en el proyecto

El proyecto se construyó con Claude Code (Claude Opus 5.5) como par de programación.
Las instrucciones dadas fueron:

1. *"Revisá los requerimientos y dame un plan, aún no ejecutemos cambios"*: el asistente
   leyó la consigna y el corpus y propuso un plan por fases. En ese análisis detectó que
   `structural` daba solo 18 chunks y que `fixed` tenía que cubrir el mínimo de 20, y
   que el repo de M1 chocaba con las rutas pedidas.
2. Decisiones tomadas por el autor sobre ese plan: repo nuevo, embeddings locales
   (recomendación del asistente), un juez más capaz que el generador y continuar el
   manual por capítulos y ADR como en M1.

Cómo influyó en las decisiones técnicas:

- El asistente propuso el etiquetado por evidencia literal (ADR-041) y la métrica
  evidence_recall (ADR-042) para comparar estrategias con la misma fórmula.
- El criterio de decisión y el dataset se commitearon antes de medir, para que la IA
  tampoco pudiera acomodar la elección a los resultados.
- El asistente redactó el borrador del dataset. Revisarlo etiqueta por etiqueta queda a
  cargo del autor (ver limitaciones).

## Verificación

```bash
for f in tests/test_*.py; do python "$f" | tail -1; done   # 64 tests, sin red
python evals/dataset.py --check                            # dataset y etiquetas al día
python evals/runner.py --no-save                           # reproduce la tabla de retrieval
```

Costo total de las mediciones con modelos reales: unos USD 0,50 (dos corridas de
generación con juez, más pruebas).

## Manual de construcción

- [Índice](docs/README.md) · [Introducción](docs/00-introduccion.md)
- Fases: [0 Repositorio](docs/01-fase-0-repositorio.md) ·
  [1 Corpus y chunking](docs/02-fase-1-corpus-y-chunking.md) ·
  [2 Ingesta e índice](docs/03-fase-2-ingesta-e-indice.md) ·
  [3 Búsqueda híbrida](docs/04-fase-3-busqueda-hibrida.md) ·
  [4 Golden dataset](docs/05-fase-4-golden-dataset.md) ·
  [5 Generación con citas](docs/06-fase-5-generacion-con-citas.md) ·
  [6 Evals](docs/07-fase-6-evals.md) ·
  [7 Decisión de chunking](docs/08-fase-7-decision-chunking.md)
- [Apéndice A: decisiones (ADR)](docs/apendice-a-decisiones.md) ·
  [Apéndice B: glosario](docs/apendice-b-glosario.md)
