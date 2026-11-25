# Capítulo 1: Repositorio y entorno (Fase 0)

> M2 se entrega como repositorio propio. Lo que sirve de M1 se copia; lo que era del
> triage se va. Y antes de escribir código, se verifica que el stack instala.

---

## 1. Objetivo

- Un repositorio nuevo, separado de M1, con las rutas que pide la consigna.
- Reutilizar los componentes probados de M1 sin arrastrar los del triage.
- Confirmar que Chroma y la librería de embeddings funcionan en Python 3.14 antes de comprometerse con ellos.
- Dejar el corpus en `data/corpus/` sin modificar y con una forma de comprobarlo.

## 2. Resultado

```
.env.example        variables documentadas (sin credenciales)
.gitignore          .venv, .env, data/index/, data/models/
requirements.txt    9 dependencias con pin exacto
config/             providers.yaml, rag.yaml
data/corpus/        hr_policies.md, it_manual.md, finance_processes.md, SHA256SUMS
src/config.py       carga de .env, providers.yaml y rag.yaml
src/providers/      base, anthropic, openai (de M1); mock y factory (reescritos en la Fase 5)
src/observability/  pricing.py (de M1), metrics.py (reescrito en la Fase 5)
```

## 3. Por qué un repositorio nuevo (ADR-034)

La carpeta de trabajo de M2 empezó como una copia de M1, con el `origin` apuntando al repositorio de M1. La consigna de M2 pide `evals/dataset.jsonl`, `evals/runner.py`, `src/` y `prompts/`: las mismas rutas, con otro contenido. Había tres caminos:

| Opción | Problema |
|---|---|
| M2 dentro de M1, en subcarpetas | choca con las rutas pedidas; el corrector tendría que ignorar medio repo |
| M1 como dependencia instalable | acopla dos entregas que se corrigen por separado |
| **Repositorio nuevo, componentes copiados** | una mejora en M1 no llega sola a M2 (aceptable) |

El historial de M1 sigue completo en su repositorio y en su carpeta. M2 arranca con `git init` y su primer commit ya trae la estructura limpia.

### Qué se copió y qué se descartó

| Se copió de M1 | Se descartó |
|---|---|
| `src/providers/base.py`, `anthropic_provider.py`, `openai_provider.py` | `schemas.py` del triage (categorías, acciones) |
| `src/observability/pricing.py` | prompts v1–v4 del triage |
| la forma de `src/config.py` (`.env` + YAML, `ModelConfig`) | guardrails de entrada y salida, suite adversarial |
| el reintento con el error de validación (ahora en `src/llm.py`) | caché MD5, `select.py`, dataset de tickets |
| el criterio de trazabilidad de los resultados | |
| 24 tests de proveedores (sin los del pipeline de triage) | |

El mock de M1 clasificaba por palabras clave; en M2 se reescribió como un generador extractivo (Capítulo 6). `metrics.py` registra ahora estrategia, modo, chunks recuperados y citados.

## 4. Paso a paso

### 4.1 Verificar el stack en Python 3.14

El riesgo principal del plan era que `chromadb` o el runtime de embeddings no tuvieran wheels para 3.14. Se probó en un entorno descartable antes de tocar el repositorio:

```bash
uv venv --python 3.14 /tmp/venvtest
VIRTUAL_ENV=/tmp/venvtest uv pip install chromadb fastembed rank-bm25
```

Resultado: `chromadb 1.5.9`, `fastembed 0.9.1`, `onnxruntime 1.32.0`, `rank-bm25 0.2.2`, todos con wheels. Una prueba de humo embebió tres textos y los consultó en una colección coseno: la pregunta "¿cuánta plata me dan para internet?" quedó a distancia 0,57 del texto del estipendio y a 0,85 del de contraseñas. El stack funciona y no hizo falta bajar de versión de Python.

En la misma prueba apareció el primer cambio de plan: `fastembed` no ofrece `multilingual-e5-small`, el modelo recomendado en el plan. Se usó `paraphrase-multilingual-MiniLM-L12-v2` (ADR-036, Capítulo 3).

### 4.2 Dependencias

```
openai==3.37.0          anthropic==2.1.0        pydantic==2.14.1
PyYAML==6.0.3           python-dotenv==1.2.4    chromadb==1.5.9
fastembed==0.9.1        rank-bm25==0.2.2        snowballstemmer==3.1.1
```

Pins exactos, como en M1 (ADR-003). `snowballstemmer` es Python puro: aporta el stemming en español de BM25.

### 4.3 Higiene de Git

```gitignore
data/index/     # lo construye la ingesta: nunca se versiona
data/models/    # modelo de embeddings descargado (~240 MB en disco)
```

Sí se versionan `data/chunks/*.jsonl` (el export legible de los chunks), `data/metrics.jsonl` (evidencia de ejecuciones reales) y `evals/results/`.

### 4.4 El corpus, sin modificar (ADR-035)

Los tres documentos se descargaron de la carpeta de Drive de la consigna y se copiaron tal cual. Se guardó su hash:

```bash
cd data/corpus && shasum -a 256 *.md > SHA256SUMS
```

```
b8d558ced1f16eb5138f3609488edfb27ac83fa01e70a2764b47ba116f90ae99  finance_processes.md
1233f34807badbc12fe2666fc3dbb44331fdb56e662fd8bfec9619a1315c78ee  hr_policies.md
65f92df2e93fdb1777c4c1f8fa74b90e5af777e558cb861d5248ad2204bf2777  it_manual.md
```

`ingest/loader.py:39` (`load_corpus`) recalcula el hash de cada archivo y falla si no coincide. Importa por dos motivos: la consigna exige el corpus sin modificar, y el golden dataset ubica su evidencia por posición de caracteres (Capítulo 5), así que cualquier cambio la desplaza.

Los archivos vienen con fin de línea LF. `.gitattributes` normaliza a LF, así que Git no los reescribe en ningún sistema operativo.

## 5. Configuración

`config/rag.yaml` concentra todo lo que cambia los chunks o el ranking:

```yaml
embeddings:   {model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, dimensions: 384}
vector_store: {collection_prefix: rag, distance: cosine}
chunking:
  fixed:      {size_chars: 300, overlap_chars: 60}
  structural: {max_chars: 1200}
retrieval:    {top_k: 3, candidates: 10, rrf_k: 60, default_mode: hybrid, default_strategy: structural}
```

`config/providers.yaml` define los modelos y, además, **roles**: qué modelo genera (`anthropic/default`, Haiku 5.5) y cuál juzga (`anthropic/sonnet-5-5`). El proveedor por defecto sigue siendo `mock`.

`src/config.py:147` (`load_rag_config`) valida al cargar: `0 <= overlap < size`, `1 <= top_k <= candidates`, modo y estrategia conocidos. Un YAML inválido falla al arrancar, no en mitad de una corrida.

## 6. Verificación

```bash
python -c "from ingest.loader import load_corpus; from pathlib import Path; print([d.source for d in load_corpus(Path('data/corpus'))])"
# ['finance_processes.md', 'hr_policies.md', 'it_manual.md']
python tests/test_anthropic_provider.py | tail -1    # 12/12 tests pasaron
python tests/test_openai_provider.py | tail -1       # 12/12 tests pasaron
```

## 7. Errores comunes

| Síntoma | Causa | Corrección |
|---|---|---|
| `hr_policies.md no coincide con SHA256SUMS` | el archivo se editó o se guardó con otro fin de línea | volver a copiar el original de la consigna |
| `git push` va al repositorio de M1 | la carpeta conserva el `.git` copiado | `git init` en la carpeta nueva y `git remote add origin <repo M2>` |
| la ingesta descarga el modelo cada vez | `MODELS_DIR` apunta a una carpeta temporal | dejar el default `data/models` |

## 8. Checklist de cierre

- [x] Repositorio con historial propio, sin código de triage.
- [x] `chromadb` y `fastembed` verificados en Python 3.14.
- [x] Corpus en `data/corpus/` idéntico al provisto, con `SHA256SUMS`.
- [x] Índice y modelos fuera de Git.
- [x] Tests de proveedores heredados pasando.
