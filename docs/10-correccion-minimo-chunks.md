# Capítulo 10: Corrección, `structural` cumple el mínimo de 20 chunks

> Leer bien la consigna también es parte del sistema: el mínimo lo tiene que cumplir
> lo que se entrega, no una alternativa descartada.

---

## 1. Objetivo

La consigna pide **al menos 20 chunks**. En el Capítulo 2 se vio que `structural` daba
18 y se dio por cumplido el mínimo con `fixed` (35). Pero el asistente usa `structural`
(ADR-049), así que lo entregado no cumplía. Este capítulo corrige eso con el menor
cambio posible y vuelve a medir todo lo que depende de los chunks.

## 2. Resultado

- `config/rag.yaml`: `chunking.structural.max_chars` pasa de 1.200 a **590**.
- `data/chunks/structural.jsonl`: **20 chunks** (164 a 580 caracteres, 445 en promedio).
- `evals/dataset.jsonl`: chunks esperados regenerados. Las evidencias no cambian.
- `tests/test_chunking.py`: exige al menos 20 chunks en las dos estrategias y en la que usa el asistente.
- Nuevas corridas en `evals/results/` y ADR-052.

## 3. Conceptos

**Cambiar un parámetro después de medir.** Elegir un valor mirando las métricas del
golden dataset es sobreajustar: se termina optimizando para estas 49 preguntas. Por eso
el valor se elige con un criterio que no mira resultados (cuántos chunks da) y recién
después se mide.

**Subdivisión con encabezado repetido.** `structural` ya tenía la regla (Capítulo 2,
§5): una sección más larga que `max_chars` se parte por líneas completas y cada parte
lleva el título del documento y el de la sección. Bajar `max_chars` no agrega código.

## 4. Paso a paso

1. **Criterio antes de medir:** el mayor múltiplo de 10 que dé al menos 20 chunks. Así
   el diseño cambia lo mínimo respecto del que ya se evaluó.
2. **Contar sin medir:**

   | `max_chars` | chunks |
   |---|---|
   | 600 | 18 |
   | 599 | 19 |
   | 591 | 20 |
   | **590** | **20** |
   | 580 | 21 |
   | 500 | 27 |

   Con 590 solo se parten las dos secciones más largas de HR: *Licencias y Vacaciones*
   (la última línea, "Días por estudio", queda en su propio chunk) y *Licencias por
   Enfermedad y Ausentismo* (la última oración, sobre el descuento de haberes, queda
   aparte).
3. **Aplicar y reindexar:**

   ```bash
   python -m ingest.ingest --strategy structural
   python evals/dataset.py --write      # chunks esperados con los nuevos ids
   python evals/review_labels.py --summary   # sigue: aprobada 49
   ```

4. **Medir de nuevo:** retrieval (`python evals/runner.py`), generación con juez
   (`python evals/runner.py --generation --strategy structural --provider anthropic`,
   USD 0,23) y re-ranking (`python evals/runner.py --modes hybrid --k 1 3 5 --rerank`).

## 5. Verificación

### Retrieval (`hybrid`, k = 3, `evals/results/retrieval_20261127T002855Z.json`)

| `structural` | 18 chunks | 20 chunks |
|---|---|---|
| evidence_recall@3 | 0,982 | **0,958** |
| MRR | 0,897 | 0,873 |
| precision@3 | 0,373 | 0,365 |
| hit@3 | 1,000 | 0,976 |

`fixed` no cambia (0,802). La única pregunta nueva que falla es **q12** (*"Si falto por
enfermedad y presento todo en regla, ¿me descuentan ese día del sueldo?"*). Su evidencia
quedó sola en la última parte de la sección, sin la palabra "enfermedad", y no entra en
el top 3. Es el costo esperable de partir una sección: la oración pierde su contexto.

### Generación (`evals/results/generation_20261127T003206Z.json`)

| `structural` | 18 chunks | 20 chunks |
|---|---|---|
| groundedness | 0,971 | **0,981** |
| respuestas 100 % sostenidas | 90,5 % | **92,9 %** |
| citas a chunks relevantes | 0,988 | 0,948 |
| rechazo correcto | 7/7 | 7/7 |
| falsos rechazos | 0/42 | 0/42 |
| diagnóstico (ok / retrieval / generación) | 36 / 2 / 4 | 36 / 3 / 3 |

En q12 el modelo no inventa: responde lo que figura en el chunk recuperado (cómo avisar
y justificar la ausencia) y dice que no encontró si se descuenta el día.

### Re-ranking

`python evals/runner.py --modes hybrid --k 1 3 5 --rerank`
(`evals/results/retrieval_20261127T004928Z.json`):

| `structural`, k = 3 | sin re-ranking | con re-ranking |
|---|---|---|
| evidence_recall@3 | 0,958 | **1,000** (+0,042) |
| MRR | 0,873 | **1,000** |
| latencia de recuperación p50 | 22 ms | 3.615 ms |

Con 20 chunks, el re-ranking recupera q12, q29 y q42 y **ahora sí supera** la mejora
mínima de ADR-050 (+0,03). Pero el criterio exige las dos condiciones, y la otra no se
cumple: suma más de 3,5 s al p50 (el límite era 1 s al p95). **Sigue desactivado por
defecto** (ADR-051, actualizado). Las latencias de esta corrida tienen ruido porque
compartió la CPU con otro proceso (el p95 llegó a 7 s); el p50 coincide con la
medición original (3,8 s).

`fixed` no cambia: 0,802 → 0,921 con re-ranking, todavía por debajo de `structural` sin él.

## 6. Decisiones y trade-offs

- **`structural` sigue siendo la estrategia elegida** (ADR-049): 0,958 contra 0,802 de
  `fixed`, más de cinco veces el umbral de empate.
- **Se acepta perder q12 en retrieval** a cambio de cumplir la consigna con el mínimo
  cambio. Un corte balanceado (partes de tamaño parecido) o repetir la primera línea de
  la sección en cada parte probablemente la recuperaría, pero sería elegir el diseño
  mirando el dataset de evaluación. Queda como mejora posible, a medir con preguntas nuevas.
- **20 es justo el mínimo.** El test de chunking falla si una edición del corpus lo baja.

## 7. Errores comunes

| Síntoma | Causa | Corrección |
|---|---|---|
| `IndexNotReady: los parámetros de chunking de 'structural' cambiaron` | se cambió `max_chars` sin reindexar | `python -m ingest.ingest --strategy structural` |
| `expected_chunks desactualizado` en el runner | ids de chunks nuevos | `python evals/dataset.py --write` |

## 8. Checklist de cierre

- [x] `structural`, la estrategia del asistente, tiene al menos 20 chunks (20).
- [x] Valor elegido con un criterio que no mira métricas.
- [x] Retrieval, generación y re-ranking medidos de nuevo.
- [x] La decisión de estrategia se revisó con los números nuevos y se mantiene.
- [x] Test que protege el mínimo.
