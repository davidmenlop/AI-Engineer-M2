# Capítulo 5: Golden dataset (Fase 4)

> Los chunks esperados dependen de cómo se corte el corpus. Si se etiquetan a mano
> para cada estrategia, la comparación depende de quién etiquetó. Por eso acá se
> etiqueta la evidencia, no los chunks.

---

## 1. Objetivo

Un dataset de al menos 25 preguntas con sus chunks esperados para cada estrategia, que incluya preguntas difíciles (sinónimos, distractores, que cruzan documentos) y preguntas fuera de alcance para el bonus. Y que esté congelado antes de la primera medición.

## 2. Resultado

- `evals/dataset.jsonl`: 49 preguntas.
- `evals/dataset.py`: validación, etiquetado y `--check`.

```bash
python evals/dataset.py --check
```

```
49 preguntas (42 dentro de alcance)
  directa            11
  sinonimos          14
  termino_exacto     5
  distractor         7
  cruza_documentos   5
  fuera_de_alcance   7
  chunks esperados por pregunta (fixed): min 1 / prom 1.83 / max 5
  chunks esperados por pregunta (structural): min 1 / prom 1.17 / max 3
```

## 3. Tipos de pregunta

| Tipo | Qué prueba | Ejemplo |
|---|---|---|
| directa (11) | el caso fácil, con palabras del documento | q27 "¿Cada cuánto se hacen los backups de las bases productivas y por cuánto tiempo se guardan?" |
| sinonimos (14) | que el retrieval entienda sin coincidencia literal | q20 "¿Cada cuánto tiempo me cambian la compu por una nueva?" → "renovación de equipos de cómputo" |
| termino_exacto (5) | canales, plataformas, montos | q10 "¿por qué canal de Slack aviso y hasta qué hora?" → `#people-avisos`, 10 AM |
| distractor (7) | la respuesta convive con otras parecidas | q03 "Me caso el mes que viene" → matrimonio (10 días), junto a nacimiento (15), fallecimiento (3), estudio (2 por examen) |
| cruza_documentos (5) | evidencia en dos documentos | q17 crédito de capacitación: monto en HR, rendición en Finance |
| fuera_de_alcance (7) | rechazo honesto | o02 "¿Cuántos días de licencia me dan por mudanza?" |

Los fuera de alcance incluyen **distractores de rechazo**: preguntas que se parecen a algo que sí está en el corpus. o02 (licencia por mudanza, cuando hay otras licencias), o04 (aguinaldo, cuando hay bonos), o05 (tarjeta de acceso perdida, cuando hay pérdida de equipos) y o06 (adelanto de sueldo, cuando hay fechas de cobro). Un asistente que "completa" con lo parecido falla ahí.

Las preguntas son paráfrasis propias. Los ejemplos de la consigna se usaron como guía de tipos y no se copiaron.

## 4. Evidencia literal y etiquetado por solapamiento (ADR-041)

Cada fila guarda su evidencia como fragmentos **literales** del corpus:

```json
{"id": "q17", "type": "cruza_documentos", "in_scope": true,
 "question": "¿Cuánto es el crédito de capacitación y cómo pido que me devuelvan lo que pagué por un curso?",
 "evidence": [
   {"source": "hr_policies.md", "quote": "- Crédito de capacitación: USD 500 anuales reembolsables para cursos, certificaciones y libros de tecnología."},
   {"source": "finance_processes.md", "quote": "se rinde cargando la factura del curso o certificación en ExpenseHub, dentro de los 30 días posteriores a la compra."}],
 "reference_answer": "USD 500 anuales; se rinde cargando la factura en ExpenseHub dentro de los 30 días de la compra.",
 "expected_chunks": {"fixed": ["finance_processes.md_5", "finance_processes.md_6", "hr_policies.md_10", "hr_policies.md_11"],
                     "structural": ["finance_processes.md_3", "hr_policies.md_5"]}}
```

`evals/dataset.py:72` (`locate`) ubica cada cita en su documento y exige que aparezca **exactamente una vez**. Así cada evidencia es un tramo `[start, end)` sin ambigüedad.

`expected_chunks` no se escribe a mano. Lo calcula `python evals/dataset.py --write` con una sola regla para las dos estrategias (`evals/dataset.py:53`):

```python
def is_relevant(self, chunk: Chunk) -> bool:
    return self.overlap(chunk) >= min(MIN_OVERLAP_CHARS, self.end - self.start)   # MIN_OVERLAP_CHARS = 30
```

Un chunk es relevante si se solapa con alguna evidencia en al menos 30 caracteres, o en la evidencia entera si es más corta. Ejemplo q17: en `structural` la evidencia cae en 2 chunks (una sección por documento); en `fixed`, cada cita cruza un borde y cae en 2 chunks, 4 en total.

Que `fixed` tenga 1,83 chunks esperados por pregunta y `structural` 1,17 es un efecto real de la estrategia: con ventanas de 300 caracteres y 60 de solapamiento, un dato suele quedar partido o repetido. Esto afecta a precision y recall (Capítulo 7).

### Validaciones (`evals/dataset.py:105`, `parse_rows`)

- ids únicos; tipo dentro de la lista cerrada; pregunta no vacía;
- `in_scope = true` exige evidencia y `in_scope = false` no la admite;
- `fuera_de_alcance` corresponde exactamente a `in_scope = false`;
- toda evidencia existe literal y una sola vez en su documento.

`load_dataset` (usado por el runner) además falla si `expected_chunks` quedó desactualizado respecto de los parámetros de chunking actuales: cambiar `size_chars` sin reetiquetar no pasa desapercibido. El runner verifica también que cada chunk esperado exista en el índice que se mide (`evals/runner.py:84`).

## 5. Congelado antes de medir (ADR-048)

El dataset se commiteó (`9d2242c`) junto con el criterio de decisión (`0205a48`) **antes** de la primera corrida del runner (`evals/results/retrieval_20261125T164259Z.json`, commit `c6a7f3c`). Ninguna pregunta se agregó, quitó ni reetiquetó después de ver resultados. Un cambio futuro se documenta como corrección y el runner deja de comparar las corridas anteriores (filtra por sha256 del dataset).

## 6. Limitaciones

- **Redactado con asistencia de IA.** Las 49 preguntas, sus evidencias y sus respuestas de referencia se escribieron con un asistente de IA a partir de la lectura del corpus. Las evidencias están validadas automáticamente (existen literal en el corpus), y después una persona las revisó una por una con `evals/review_labels.py`: que cada evidencia sea *toda* la evidencia necesaria y que cada pregunta fuera de alcance realmente no tenga respuesta. Las 49 quedaron aprobadas sin disputas (`evals/dataset_review.jsonl`). Fue un solo revisor, así que no hay acuerdo entre revisores medido.
- **Sintético y chico.** 42 preguntas dentro de alcance, sin consultas reales de empleados. 5 preguntas que cruzan documentos son pocas para sacar conclusiones por tipo.
- **Sin holdout.** Con 53 chunks en total, partir el dataset dejaría muy pocas preguntas por lado. La protección contra el sobreajuste es el congelado previo, no una partición.
- **Evidencia mínima.** Se etiquetó la oración que responde, no el contexto que ayuda. Una respuesta que necesita el contexto de la sección se apoya en que `structural` lo trae completo.

## 7. Errores comunes

| Síntoma | Causa | Corrección |
|---|---|---|
| `la evidencia no aparece literal` | una comilla, tilde o espacio distinto al del corpus | copiar la cita del archivo, no reescribirla |
| `la evidencia aparece más de una vez` | cita demasiado corta (p. ej. "15 días") | extender la cita hasta que sea única |
| `expected_chunks desactualizado` | se cambiaron parámetros de chunking | `python evals/dataset.py --write` y reindexar |

## 8. Checklist de cierre

- [x] ≥ 25 preguntas (49), con sinónimos, distractores y preguntas que cruzan documentos.
- [x] Chunks esperados por estrategia, con la misma regla para ambas.
- [x] Preguntas fuera de alcance marcadas (bonus).
- [x] Dataset congelado antes de medir.
- [x] Revisión humana de cada etiqueta (49 aprobadas, 0 en disputa).
