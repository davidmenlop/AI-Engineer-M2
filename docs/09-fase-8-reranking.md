# Capítulo 9: Re-ranking con cross-encoder (Fase 8, extra credit)

> Medir antes de activar: el re-ranking mejora el orden, pero en este corpus no paga
> su latencia.

---

## 1. Objetivo

Probar si un cross-encoder que reordena los candidatos de la búsqueda híbrida mejora el
retrieval, y cuánto cuesta en latencia. La consigna lo pide **con el delta de métricas
medido contra el pipeline sin re-ranking y su costo en latencia**. La guía advierte no
re-rankear todo por defecto sin medir.

## 2. Resultado

- `src/reranker.py`: `CrossEncoderReranker` sobre `fastembed` (`BAAI/bge-reranker-v2-m3-int8`).
- `src/retriever.py`: paso opcional después de la fusión (`retrieve(..., rerank=True)`).
- `--rerank` en la CLI (`src/main.py`) y en el runner (`evals/runner.py`), que mide con y
  sin re-ranking en la misma corrida y reporta latencia p50/p95.
- `config/rag.yaml`: sección `rerank`, con `enabled: false`.
- ADR-050 (criterio, escrito antes de medir) y ADR-051 (resultado).

## 3. Conceptos

**Bi-encoder vs cross-encoder.** Los embeddings (bi-encoder) codifican la pregunta y el
chunk **por separado** y comparan vectores: es rápido porque los chunks se codifican una
vez, en la ingesta. Un cross-encoder lee **pregunta y chunk juntos** y devuelve un
puntaje de relevancia. Es más preciso, pero hay que correrlo por cada par en cada
consulta. Por eso se usa como segunda etapa, sobre pocos candidatos.

**Por qué después de RRF.** La fusión decide *qué* chunks son candidatos, con dos
señales baratas. El cross-encoder solo reordena los 10 primeros. En q29, RRF deja
`it_manual.md_4` (primero en BM25, fuera del top 10 vectorial) fuera del top 6. El
cross-encoder lo lee junto con la pregunta y lo sube al top 3.

## 4. Paso a paso

```bash
python -m src.main --query "Me robaron la notebook en un viaje, ¿qué hago?" --rerank --retrieve-only --pretty
python evals/runner.py --modes hybrid --k 1 3 5 --rerank
```

La primera ejecución descarga el modelo (~570 MB) a `data/models/`.

## 5. Código explicado

- `CrossEncoderReranker.scores` (`src/reranker.py`) devuelve la sigmoide del logit del
  modelo: una probabilidad en [0, 1] que reemplaza al `score` de RRF en el contrato.
- `Retriever._rerank` (`src/retriever.py`) puntúa los `rerank.candidates` primeros
  fusionados, los ordena por ese puntaje y se queda con `top_k`. Mide su propia
  latencia en `last_rerank_ms`.
- Pedir `rerank=True` a un retriever sin reranker falla con `ValueError`. Así nunca se
  mide "con re-ranking" sin que lo haya.

## 6. Verificación

Criterio fijado antes de medir (ADR-050): se activa por defecto si en `structural` +
`hybrid` + k = 3 mejora evidence_recall@3 en al menos 0,03 **y** agrega menos de
1.000 ms al p95.

`evals/results/retrieval_20261125T174156Z.json` (42 preguntas):

| estrategia | k | re-ranking | precision@k | evidence_recall@k | complete@k | hit@k | MRR | ms p50 | ms p95 |
|---|---|---|---|---|---|---|---|---|---|
| structural | 1 | no | 0.809 | 0.740 | 0.691 | 0.809 | 0.809 | 9 | 17 |
| structural | 1 | sí | **1.000** | **0.931** | 0.881 | 1.000 | 1.000 | 1.870 | 2.321 |
| structural | 3 | no | 0.373 | 0.982 | 0.952 | 1.000 | 0.897 | 10 | 18 |
| structural | 3 | sí | 0.389 | **1.000** | **1.000** | 1.000 | **1.000** | 3.777 | 4.468 |
| fixed | 3 | no | 0.452 | 0.802 | 0.738 | 0.929 | 0.762 | 8 | 12 |
| fixed | 3 | sí | 0.508 | **0.921** | 0.881 | 1.000 | 0.988 | 941 | 1.218 |
| fixed | 5 | sí | 0.343 | 0.941 | 0.929 | 1.000 | 0.988 | 1.010 | 1.288 |

Qué cambia por pregunta (`structural`, k = 3):

- **Completa la evidencia en q29 y q42**, las dos preguntas que cruzan documentos donde
  fallaba el retrieval (0,75 → 1,0 y 0,5 → 1,0).
- **Sube la evidencia al primer lugar** en q07, q12, q16, q19, q21, q24, q25 y q37. Son
  casi todos los casos de sinónimos donde un buscador fallaba y el otro rescataba.
- En `fixed` mejora 7 preguntas y empeora 1 (q10, canal de Slack por enfermedad: un
  término exacto que BM25 ya ubicaba bien).

## 7. Decisiones y trade-offs

**Decisión (ADR-051): queda desactivado por defecto.** La mejora en `structural` es real
pero chica (+0,018), y multiplica por ~400 la latencia de recuperación: 3,8 s de p50
sobre una respuesta que hoy tarda ~1,1 s en total.

- El reranker rinde más donde el retrieval base es más débil: con `fixed` (+0,119) y con
  k = 1 (+0,191 en `structural`). Si se quisiera pasarle **un solo chunk** al modelo,
  el re-ranking sería la diferencia.
- Los chunks de `structural` son más largos (486 caracteres en promedio, contra 285), y
  el costo de un cross-encoder crece con el largo del texto. Por eso tarda 4 veces más
  que con `fixed`.
- La latencia se midió en CPU (Apple Silicon, ONNX INT8). Con GPU o un modelo más chico
  bajaría, pero los rerankers chicos disponibles en `fastembed` están entrenados solo en
  inglés.

## 8. Errores comunes

- **Medir la latencia con la primera consulta.** La primera llamada carga el modelo
  ONNX. El runner hace una consulta de calentamiento antes de medir.
- **Comparar con otro `top_k` o con otro dataset.** El runner mide ambas variantes en la
  misma corrida, con el mismo dataset y los mismos k.
- **Usar un reranker en inglés con un corpus en español.** `ms-marco-MiniLM` es 10 veces
  más chico, pero no entiende "obra social" ni "compu".

## 9. Checklist de cierre

- [x] Criterio escrito antes de medir (ADR-050).
- [x] Delta de métricas y de latencia medido contra el pipeline sin re-ranking.
- [x] Decisión documentada con números (ADR-051) y opción disponible con `--rerank`.
- [x] Tests sin red con un reranker falso (`tests/test_retrieval.py`).
