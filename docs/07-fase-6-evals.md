# Capítulo 7: Suite de evals (Fase 6)

> Si el asistente responde mal, primero hay que saber si el dato no llegó al contexto
> o si llegó y el modelo no lo usó bien. Son dos problemas con dos soluciones distintas.

---

## 1. Objetivo

Un script ejecutable por comando, sin framework de testing, que mida retrieval (precision y recall contra el golden dataset) y generación (groundedness) **por separado**, para cada estrategia, y que diga en qué etapa falla cada pregunta.

## 2. Resultado

```
evals/metrics.py    métricas puras (sin red)
evals/judge.py      juez de groundedness
evals/runner.py     CLI de la suite
evals/results/      una corrida por archivo JSON, con hashes y commit
```

```bash
python evals/runner.py                                  # retrieval: 2 estrategias × 3 modos, k de rag.yaml (sin claves)
python evals/runner.py --k 1 3 5                        # barrido de k
python evals/runner.py --generation --strategy structural --provider anthropic
python evals/runner.py --generation --strategy fixed --limit 5 --no-save   # prueba rápida
python evals/runner.py --compare evals/results/*.json   # tablas markdown de corridas guardadas
```

## 3. Métricas de retrieval (`evals/metrics.py`)

Se calculan sobre las 42 preguntas dentro de alcance, con los k chunks recuperados.

| Métrica | Definición | Qué responde |
|---|---|---|
| precision@k | relevantes recuperados / k | cuánto ruido acompaña a la evidencia |
| recall@k | relevantes recuperados / relevantes esperados | definición clásica, a nivel chunk |
| **evidence_recall@k** | evidencias cubiertas / evidencias | ¿llegó el dato al modelo? |
| complete@k | 1 si todas las evidencias quedaron cubiertas | ¿llegó **todo** lo necesario? |
| hit@k | 1 si al menos un recuperado es relevante | ¿llegó algo? |
| MRR | 1 / posición del primer relevante | ¿llegó primero? |

### Por qué dos recalls (ADR-042)

`fixed` tiene 60 caracteres de solapamiento. Un dato corto como "- Matrimonio: 10 días corridos." puede quedar completo en dos chunks vecinos; la regla de relevancia marca los dos como esperados, y el recall clásico exige recuperar ambos aunque uno ya alcance. Eso castiga a `fixed` por un artefacto de la regla.

**evidence_recall** mira la evidencia, no los chunks: una evidencia está cubierta si la unión de los tramos recuperados cubre al menos el 80 % de sus caracteres (`COVERAGE`, `evals/metrics.py:23`; `covered_fraction`, `evals/metrics.py:33`). Es la métrica que decide la estrategia (ADR-040). En la práctica, los dos recalls dieron casi lo mismo (por ejemplo, `structural` híbrido k = 3: 0,980 y 0,982), pero la elección no depende de esa casualidad.

### El techo de precision

Si una pregunta tiene un solo chunk esperado y se recuperan 3, precision@3 no puede superar 0,33. `structural` tiene 1,17 chunks esperados por pregunta y `fixed`, 1,83: `fixed` tiene un techo de precision más alto por construcción. Precision se lee junto con recall, nunca sola.

## 4. Groundedness con un juez (ADR-043)

**Groundedness**: qué parte de lo que afirma la respuesta se sostiene en los chunks que **cita**. No mide si la respuesta es completa ni si es correcta respecto de la referencia; eso es otra métrica.

`prompts/judge_groundedness_v1.yaml` le pide al juez:

1. Dividir la respuesta en afirmaciones atómicas (cada monto, plazo, canal o requisito es una).
2. Marcar cada una `supported: true/false` contra los fragmentos citados. Una afirmación parcialmente sostenida (monto correcto, plazo inventado) es `false`.
3. Devolver JSON: `{"claims": [{"claim", "supported", "chunk_id", "reason"}]}`.

El puntaje lo calcula el código, no el juez: `sostenidas / total` (`evals/metrics.py:85`). Las respuestas de rechazo no se juzgan.

| | Elección | Por qué |
|---|---|---|
| Modelo | Claude Sonnet 5.5, `effort: medium` | distinto y más capaz que el generador: evita la autoevaluación |
| Contexto | solo los chunks citados | la consigna pide que la respuesta se sostenga en lo que cita |
| Granularidad | por afirmación | cada `false` queda guardado con su motivo y se puede auditar |

Costo: unos USD 0,005 por respuesta juzgada (USD 0,22 para 42 respuestas). El juez reintenta con el error si su JSON no valida (`evals/judge.py:65`).

## 5. Métricas de generación

| Métrica | Sobre | Definición |
|---|---|---|
| groundedness | respuestas juzgadas | promedio de sostenidas / total |
| 100 % sostenidas | respuestas juzgadas | proporción con groundedness = 1 |
| citas relevantes | respuestas dentro de alcance | proporción de chunks citados que están entre los esperados |
| rechazo correcto | 7 fuera de alcance | proporción rechazada (bonus) |
| falso rechazo | 42 dentro de alcance | proporción rechazada sin deber |
| retrieval completo | 42 dentro de alcance | complete@k de la corrida |
| fallback / reintento | todas | sin respuesta válida / más de un intento |
| costo y latencia | todas | generador y juez por separado; p50 y p95 |

## 6. Diagnóstico por etapa (`evals/runner.py:212`)

Para cada pregunta dentro de alcance:

```python
retrieval_ok = complete == 1.0                                   # toda la evidencia llegó
generation_ok = success and not refused and grounded == 1.0      # respondió, sostenido al 100 %
```

| Diagnóstico | Significado | Qué se corrige |
|---|---|---|
| `ok` | llegó todo y la respuesta se sostiene entera | — |
| `retrieval` | faltó evidencia, pero la respuesta no inventó | chunking, búsqueda, k |
| `generacion` | llegó todo, pero la respuesta afirma algo no sostenido o rechaza | prompt, modelo |
| `ambas` | faltó evidencia y además la respuesta falla | primero el retrieval |

Fuera de alcance: `ok` si rechazó, `respondio_fuera_de_alcance` si no.

Es la pieza que responde la pregunta del módulo: "si el sistema responde mal, ¿cuál de las dos etapas falla?".

## 7. Trazabilidad

Cada archivo de `evals/results/` guarda:

- `dataset_sha256`, `corpus_sha256`, `git.commit` y `git.dirty`;
- parámetros de recuperación (modelo de embeddings, candidatos, `rrf_k`, chunking de ambas estrategias);
- en generación: proveedor, modelo, versión y sha256 de los prompts del generador y del juez;
- el detalle por pregunta: ids recuperados, esperados y citados, respuesta, afirmaciones del juez con su motivo, diagnóstico.

`--compare` ignora corridas hechas con otra versión del dataset. Un error fatal del proveedor (clave inválida, sin saldo) aborta sin guardar resultados parciales.

## 8. Verificación

```bash
python tests/test_evals.py | tail -1   # 10/10 tests pasaron
python evals/runner.py --no-save       # sin claves, ~10 s
```

Los tests comprueban el dataset (≥ 25 preguntas, etiquetas al día, preguntas que cruzan documentos con 2 fuentes), la regla de solapamiento en el borde de 30 caracteres, la cobertura por unión de tramos, `retrieval_scores` sobre un caso armado a mano y las cuatro ramas del diagnóstico.

## 9. Limitaciones

- **El juez no se calibró contra etiquetas humanas.** Sus decisiones se revisaron a mano en las respuestas marcadas (Capítulo 8) y son estrictas pero en general defendibles; una de seis es discutible.
- **Una corrida por configuración.** Haiku 5.5 no admite `temperature`; la variabilidad entre corridas no está medida.
- **Groundedness no es correctness.** Una respuesta puede estar 100 % sostenida y ser incompleta (q29). Lo completo se ve en `complete@k` y en la respuesta de referencia, que hoy no se compara automáticamente.

## 10. Checklist de cierre

- [x] Script por comando, sin framework de testing.
- [x] Precision y recall de retrieval contra el golden dataset, por estrategia y modo.
- [x] Groundedness con LLM-as-judge, separado del retrieval.
- [x] Rechazo correcto medido (bonus).
- [x] Diagnóstico por etapa para cada pregunta.
