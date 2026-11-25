# Capítulo 8: Decisión de chunking (Fase 7)

> El criterio se escribió y se commiteó antes de correr el runner. Este capítulo solo
> lo aplica a los números y después mira qué hay detrás de ellos.

---

## 1. Objetivo

Elegir entre `fixed` y `structural` con métricas de retrieval, con el mismo dataset y el mismo `top_k`, y documentar los trade-offs.

## 2. El criterio, fijado antes de medir (ADR-040)

Commit `0205a48`, anterior a la primera corrida (`c6a7f3c`):

- Mismo dataset (49 preguntas, 42 dentro de alcance), modo `hybrid` (el de producción), `top_k = 3`.
- Orden de criterios, con diferencias menores a 0,03 tratadas como empate:
  1. **evidence_recall@3**: ¿la evidencia llega al contexto?
  2. **MRR**: ¿llega primero?
  3. **precision@3**: cuánto ruido la acompaña.
  4. **Tamaño del contexto**: menos es más barato.
- k = 3 se eligió porque con 18 chunks, k = 5 ya trae más de un cuarto del corpus de `structural`. k = 1 y k = 5 y los modos `vector` y `bm25` se reportan como contexto.

## 3. Resultados de retrieval

Fuente: `evals/results/retrieval_20261125T164259Z.json` (commit `c6a7f3c`, árbol limpio).

```bash
python evals/runner.py --k 1 3 5
```

| estrategia | modo | k | chunks | precision@k | recall@k | evidence_recall@k | complete@k | hit@k | MRR |
|---|---|---|---|---|---|---|---|---|---|
| fixed | hybrid | 1 | 35 | 0.595 | 0.411 | 0.476 | 0.476 | 0.595 | 0.595 |
| **fixed** | **hybrid** | **3** | 35 | **0.452** | 0.793 | **0.802** | 0.738 | 0.929 | **0.762** |
| fixed | hybrid | 5 | 35 | 0.324 | 0.899 | 0.881 | 0.857 | 0.952 | 0.768 |
| fixed | vector | 1 | 35 | 0.691 | 0.499 | 0.566 | 0.548 | 0.691 | 0.691 |
| fixed | vector | 3 | 35 | 0.413 | 0.722 | 0.736 | 0.691 | 0.905 | 0.790 |
| fixed | vector | 5 | 35 | 0.305 | 0.871 | 0.903 | 0.857 | 0.976 | 0.807 |
| fixed | bm25 | 1 | 35 | 0.571 | 0.368 | 0.411 | 0.405 | 0.571 | 0.571 |
| fixed | bm25 | 3 | 35 | 0.397 | 0.709 | 0.724 | 0.667 | 0.881 | 0.706 |
| fixed | bm25 | 5 | 35 | 0.276 | 0.788 | 0.768 | 0.738 | 0.905 | 0.711 |
| structural | hybrid | 1 | 18 | 0.809 | 0.742 | 0.740 | 0.691 | 0.809 | 0.809 |
| **structural** | **hybrid** | **3** | 18 | **0.373** | 0.980 | **0.982** | 0.952 | 1.000 | **0.897** |
| structural | hybrid | 5 | 18 | 0.229 | 0.992 | 0.994 | 0.976 | 1.000 | 0.897 |
| structural | vector | 1 | 18 | 0.762 | 0.706 | 0.704 | 0.667 | 0.762 | 0.762 |
| structural | vector | 3 | 18 | 0.349 | 0.925 | 0.923 | 0.905 | 0.952 | 0.849 |
| structural | vector | 5 | 18 | 0.214 | 0.948 | 0.946 | 0.929 | 0.976 | 0.855 |
| structural | bm25 | 1 | 18 | 0.762 | 0.694 | 0.693 | 0.643 | 0.762 | 0.762 |
| structural | bm25 | 3 | 18 | 0.349 | 0.893 | 0.893 | 0.881 | 0.905 | 0.821 |
| structural | bm25 | 5 | 18 | 0.219 | 0.929 | 0.929 | 0.929 | 0.929 | 0.827 |

### Por tipo de pregunta (evidence_recall@3)

| estrategia | modo | cruza_documentos (5) | directa (11) | distractor (7) | sinonimos (14) | termino_exacto (5) |
|---|---|---|---|---|---|---|
| fixed | hybrid | 0.633 | 0.955 | 0.714 | 0.714 | 1.000 |
| fixed | vector | 0.383 | 0.818 | 0.714 | 0.786 | 0.800 |
| fixed | bm25 | 0.583 | 0.864 | 0.714 | 0.643 | 0.800 |
| structural | hybrid | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| structural | vector | 0.750 | 1.000 | 1.000 | 0.857 | 1.000 |
| structural | bm25 | 0.900 | 1.000 | 0.857 | 0.786 | 1.000 |

## 4. Aplicación del criterio: `structural` (ADR-049)

1. **evidence_recall@3:** 0,982 contra 0,802. Diferencia de 0,180, seis veces el umbral de empate. **Decide el primer criterio.**
2. MRR (como confirmación): 0,897 contra 0,762.
3. precision@3: `fixed` gana (0,452 contra 0,373). No llega a jugar.
4. Contexto: `fixed` pasa unos 855 caracteres al modelo (3 × 285) y `structural` unos 1.458 (3 × 486). No llega a jugar.

`config/rag.yaml` queda con `default_strategy: structural`.

## 5. Por qué gana `structural`, pregunta por pregunta

- **Distractores (1,000 contra 0,714).** Las licencias parecidas (q03, q04, q06) las resuelven bien las dos estrategias. `fixed` falla en q11 y q21. En q21 ("se me cayó la laptop") no recupera ningún chunk de IT Equipos: trae dos de HR (el final de Home Office y el aviso de enfermedad) y uno de IT Offboarding ("MacBook o Dell XPS… devolverse"), que habla de equipos pero no de daños. En q11 recupera `hr_policies.md_5`, que contiene solo una parte de la regla de los 4 días: el dato está partido entre dos chunks y el otro no entra en el top 3.
- **Sinónimos (1,000 contra 0,714).** Mismo patrón. En `fixed`, q09 (trabajar desde otro país) y q24 (pasar la clave) recuperan un chunk relevante (`hit = 1`), pero ese chunk cubre menos del 80 % de la evidencia, porque la oración quedó partida entre dos ventanas (`evidence_recall = 0`). En `structural` la oración está entera en su sección.
- **Cruzan documentos (0,850 contra 0,633).** Cada evidencia de `structural` ocupa un solo chunk; en `fixed`, una pregunta que necesita 2 evidencias puede necesitar 4 chunks, más de los 3 disponibles.
- **Términos exactos:** empate en 1,000. BM25 resuelve estos casos en las dos estrategias.

## 6. Generación en ambas estrategias

La generación no decide (ADR-040), pero confirma si el retrieval elegido sostiene buenas respuestas. Generador Claude Haiku 5.5, juez Claude Sonnet 5.5, `hybrid`, k = 3, 49 preguntas.

```bash
python evals/runner.py --generation --strategy structural
python evals/runner.py --generation --strategy fixed
```

| | structural | fixed |
|---|---|---|
| archivo | `generation_20261125T164613Z.json` | `generation_20261125T164923Z.json` |
| groundedness | **0,971** | 0,919 |
| respuestas 100 % sostenidas | **90,5 %** (38/42) | 82,5 % (33/40) |
| citas relevantes | **0,988** | 0,933 |
| rechazo correcto (7 fuera de alcance) | **7/7** | **7/7** |
| falso rechazo (42 dentro de alcance) | **0** | 2 (q12, q20) |
| retrieval completo | 0,952 | 0,738 |
| reintentos | 0 % | 2 % (un timeout de la API) |
| costo generador + juez | USD 0,011 + 0,222 | USD 0,011 + 0,202 |
| latencia de generación p50 / p95 | 1,10 s / 1,79 s | 1,16 s / 3,86 s |
| diagnóstico | ok 43 · generación 4 · retrieval 2 | ok 36 · ambas 7 · retrieval 4 · generación 2 |

Con `fixed`, 7 preguntas fallan en las dos etapas a la vez: cuando falta evidencia, el modelo tiende a completar (q24, groundedness 0,0) o a rechazar (q12, q20). Con `structural`, los dos fallos de retrieval terminan en respuestas 100 % sostenidas.

### Los fallos de `structural`, uno por uno

**Retrieval (2), y el modelo no inventó:**

- **q29**, robo de notebook en un viaje. Recupera IT Equipos y Finance Reintegros, pero no IT Soporte (reemplazo de equipo y reintegro en viajes). La respuesta da los pasos de IT y aclara que la documentación recuperada no dice si el reintegro aplica. Groundedness 1,0, `complete = 0`.
- **q42**, VPN en el aeropuerto y tope de la cena. Recupera Finance Reintegros pero no IT Seguridad de Datos (VPN). Ningún modo la trae en el top 3. La respuesta se limita a lo recuperado. Groundedness 1,0.

**Generación (4), según el juez.** Se revisaron a mano las afirmaciones marcadas como no sostenidas:

| Pregunta | Afirmación marcada | Motivo del juez | Revisión |
|---|---|---|---|
| q11 | avisar "antes de las 10 AM del **primer** día de ausencia" | el texto dice "del día de la ausencia" | correcto: el modelo agregó "primer" |
| q12 | el aviso a las 10 AM es requisito para que la licencia cuente como justificada | el texto lo pone como obligación, no como condición de justificación | correcto: es una inferencia no escrita |
| q24 | "no podés pasarle tu clave por mensaje privado" | el texto prohíbe Slack o correo, no "mensajes privados" en general | **discutible**: la pregunta hablaba de un mensaje privado y Slack es el canal obvio; el juez es más estricto que un lector humano |
| q41 | el crédito es reembolsable "**solo**" para tecnología; un curso de cocina "no encaja" | el texto no dice "solo"; la pertinencia depende del rol | correcto: "solo" es una exclusividad que el texto no afirma |

Tres de cuatro son sobregeneralizaciones reales y chicas: una palabra ("primer", "solo") que vuelve más fuerte una afirmación. Es el tipo de error que un juez por afirmación detecta y un puntaje global no. q24 muestra el costo de un juez estricto: puede castigar una inferencia razonable. Con 4 casos no alcanza para calibrarlo.

## 7. Trade-offs aceptados

| A favor de `fixed` | Por qué no pesa acá |
|---|---|
| mejor precision@3 (0,452) | se debe en parte al techo: `fixed` tiene 1,83 chunks esperados por pregunta y `structural` 1,17 |
| contexto más corto (~855 caracteres) | con Haiku 5.5, la diferencia son centésimos de centavo por consulta |
| cumple por sí sola los 20 chunks mínimos | la consigna pide el mínimo en *una* de las dos estrategias |

La ventaja de `structural` depende del corpus: secciones cortas (285–534 caracteres) y de un solo tema. Con secciones largas o mal delimitadas, un chunk por sección diluye el embedding; ahí entra la subdivisión por `max_chars`, que en este corpus no se activó.

## 8. Qué queda abierto

- Las preguntas que cruzan documentos son el punto débil (complete@3 = 0,6 en `structural`: q29 y q42). k = 5 sube evidence_recall a 0,994, a costo de casi el doble de contexto. No se cambió k después de ver los resultados: el criterio lo fijaba en 3.
- Re-ranking, multi-query y otros tamaños de `fixed` no se probaron.
- El dataset espera la revisión humana etiqueta por etiqueta (Capítulo 5).

## 9. Checklist de cierre

- [x] Dos estrategias medidas con el mismo dataset, el mismo k y la misma fórmula.
- [x] Decisión explícita con los números del criterio previo.
- [x] Trade-offs documentados.
- [x] Generación medida en ambas estrategias, con las fallas revisadas a mano.
