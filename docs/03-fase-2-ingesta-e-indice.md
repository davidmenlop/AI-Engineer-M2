# Capítulo 3: Ingesta e índice (Fase 2)

> La ingesta construye el índice una vez. La consulta solo lo lee, y se niega a
> responder con un índice que ya no corresponde a la configuración.

---

## 1. Objetivo

Un comando que va de cero al índice completo por estrategia: corpus → chunks → embeddings → vector DB con metadatos → tokens BM25. Ingesta y consulta como procesos separados.

## 2. Resultado

```bash
python -m ingest.ingest --strategy structural   # o fixed, o all
```

```
INFO ingest: fixed: 3 documentos -> 35 chunks (caracteres min/prom/max 108/285/299) en 3.89 s
INFO ingest: structural: 3 documentos -> 18 chunks (caracteres min/prom/max 314/486/598) en 0.96 s
INFO ingest: índice persistido en .../data/index
```

(La primera corrida tarda unos 35 s más: descarga el modelo.)

Queda en disco:

```
data/index/chroma/                 colecciones rag_fixed y rag_structural (fuera de Git)
data/index/bm25_fixed.json         tokens BM25 por chunk (fuera de Git)
data/index/bm25_structural.json
data/chunks/fixed.jsonl            export legible (versionado)
data/chunks/structural.jsonl
data/models/                       modelo de embeddings, ~240 MB (fuera de Git)
```

## 3. Conceptos

**Embedding.** Vector de números que representa el significado de un texto. Textos parecidos quedan cerca. El modelo que embebe los chunks tiene que ser el mismo que embebe las preguntas: dos modelos distintos producen espacios incomparables.

**Similitud coseno.** El coseno del ángulo entre dos vectores: 1 = misma dirección. Chroma trabaja con distancias, así que `similitud = 1 - distancia` (`src/vector_store.py:83`).

**Vector DB.** Almacén que indexa vectores para buscar los más cercanos sin compararlos todos. Chroma usa HNSW, un grafo de vecinos aproximados. Con 18–35 chunks la aproximación no importa; importa la persistencia y los metadatos.

## 4. Paso a paso: `ingest/ingest.py:44` (`ingest_strategy`)

1. **Cargar y verificar el corpus** (`load_corpus`): falla si un archivo no coincide con `SHA256SUMS`.
2. **Chunking** con los parámetros de `config/rag.yaml` para la estrategia pedida. Falla si hay `chunk_id` duplicados.
3. **Embeddings** de todos los chunks en un lote. Se comprueba que la dimensión devuelta (384) coincide con la declarada en `rag.yaml`.
4. **Recrear la colección** de la estrategia: la borra si existe y la crea de nuevo (`src/vector_store.py:37`). Nunca agrega sobre un índice viejo.
5. **Agregar** ids, textos, vectores y metadatos de cada chunk.
6. **Guardar los tokens BM25** en `data/index/bm25_<estrategia>.json` (Capítulo 4).
7. **Exportar** los chunks a `data/chunks/<estrategia>.jsonl`, para revisarlos sin correr nada.

## 5. Embeddings (ADR-036)

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, servido por `fastembed` (ONNX, sin PyTorch): 384 dimensiones, entrenado para paráfrasis en más de 50 idiomas, español incluido.

| Alternativa | Por qué no |
|---|---|
| `multilingual-e5-small` (la del plan) | `fastembed` no la ofrece |
| `multilingual-e5-large` | 2,2 GB para 53 chunks |
| `text-embedding-3-small` (OpenAI) | exige clave y red para indexar: rompe "clonar y correr sin claves" |
| `all-MiniLM-L6-v2` (default de Chroma) | entrenado en inglés; "plata" y "compu" no significan nada |

Es un modelo chico, y se nota: en los evals pierde casos con términos exactos que BM25 encuentra (Capítulo 4). Por eso la búsqueda es híbrida.

`src/embeddings.py` importa `fastembed` de forma diferida y guarda el modelo en `data/models/`. El objeto `Embedder` se crea una vez y sirve para las dos estrategias.

## 6. Chroma, una colección por estrategia (ADR-037)

```python
self._client.create_collection(
    name=name,                                   # rag_fixed | rag_structural
    configuration={"hnsw": {"space": distance}}, # cosine
    metadata=metadata,
)
```

`PersistentClient` en `data/index/chroma/`, con la telemetría desactivada. Una colección por estrategia evita que un filtro olvidado mezcle chunks de las dos.

### Metadatos de la colección

| Clave | Para qué |
|---|---|
| `embedding_model` | la consulta exige el mismo modelo |
| `chunking_sha256`, `chunking_params` | la consulta exige los mismos parámetros |
| `corpus_sha256` | trazabilidad: qué corpus se indexó |
| `strategy`, `dimensions`, `distance`, `created_at` | inspección |

Cada chunk guarda además `source`, `section`, `strategy`, `char_start` y `char_end`.

### Protección contra un índice desactualizado (`src/retriever.py:79`)

`Retriever.open` nunca indexa. Antes de abrir comprueba:

1. que la colección exista (si no, `IndexNotReady` con el comando de ingesta);
2. que `embedding_model` coincida con `rag.yaml` (`src/retriever.py:85`);
3. que `chunking_sha256` coincida con los parámetros actuales (`src/retriever.py:91`);
4. que los ids de BM25 y los de Chroma sean los mismos (`src/retriever.py:102`).

Si cambiás `size_chars` en `rag.yaml` y consultás sin reindexar:

```
error: los parámetros de chunking de 'fixed' cambiaron desde la ingesta: reindexá con python -m ingest.ingest --strategy fixed
```

## 7. Verificación

```bash
python -m ingest.ingest --strategy all
python -m src.main --query "¿Qué día cobro el sueldo?" --retrieve-only --pretty
```

`--retrieve-only` devuelve los chunks recuperados sin llamar a ningún LLM: sirve para comprobar el índice sin claves.

## 8. Errores comunes

| Síntoma | Causa | Corrección |
|---|---|---|
| `no existe la colección 'rag_fixed'` | no se corrió la ingesta de esa estrategia | `python -m ingest.ingest --strategy fixed` |
| `An instance of Chroma already exists ... with different settings` | dos `PersistentClient` sobre la misma ruta con configuración distinta en el mismo proceso | usar un solo `VectorStore` por proceso |
| la ingesta falla sin red la primera vez | el modelo se descarga de Hugging Face | correrla una vez con red; después funciona offline |
| `el modelo devolvió N dimensiones` | se cambió el modelo en `rag.yaml` sin cambiar `dimensions` | actualizar ambos |

## 9. Checklist de cierre

- [x] Ingesta de cero al índice con un comando por estrategia.
- [x] Índice persistido; la consulta no reindexa.
- [x] Mismo modelo para indexar y consultar, verificado.
- [x] Una colección por estrategia con metadatos de documento, sección y estrategia.
- [x] Chunks exportados en un formato legible.
