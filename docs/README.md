# Manual de construcción

## Nubbix Assist: asistente RAG de conocimiento interno (Proyecto Integrador M2)

> De un corpus de tres documentos Markdown a un asistente que responde con citas,
> con dos estrategias de chunking comparadas con datos y evals que separan
> retrieval de generación.

**Estado del manual:** v1.0, capítulos 0 a 9.
**Última actualización:** 2026-11-25.
**Fuente de verdad:** el código del repositorio. Si el manual y el código difieren, gana el código; el manual se corrige en el mismo commit.

---

## Qué es este manual

No es un README ni una guía de instalación. Es el **registro de construcción** del proyecto: qué decidimos, por qué, qué alternativas descartamos y cómo verificamos que funciona. Continúa el manual de M1 (triage de tickets) con la misma estructura y la misma regla.

Cada capítulo corresponde a una fase y usa, cuando aporta, estas secciones:

1. **Objetivo**: qué problema resuelve la fase.
2. **Resultado**: qué queda en el repositorio al terminarla.
3. **Conceptos**: lo que hay que entender antes de leer el código.
4. **Paso a paso**: comandos y archivos, con su justificación.
5. **Código explicado**: las partes que no son obvias.
6. **Verificación**: cómo saber que está bien, con la salida esperada.
7. **Decisiones y trade-offs**: lo que se eligió y lo que se descartó.
8. **Errores comunes**: cómo se ven y cómo se corrigen.
9. **Checklist de cierre**: criterio objetivo para avanzar.

La regla sigue siendo la de M1: **ninguna decisión técnica se sostiene con intuición**. Si no hay una medición o un criterio explícito, queda marcada como pendiente.

---

## Cómo está organizado

| Capítulo | Fase | Contenido | Estado |
|---|---|---|---|
| [0: Introducción](00-introduccion.md) | — | Problema, contrato, principios y arquitectura | ✅ |
| [1: Repositorio](01-fase-0-repositorio.md) | Fase 0 | Repo nuevo, qué se reutilizó de M1, entorno, corpus verificado | ✅ |
| [2: Corpus y chunking](02-fase-1-corpus-y-chunking.md) | Fase 1 | Lectura del corpus, estrategias `fixed` y `structural` | ✅ |
| [3: Ingesta e índice](03-fase-2-ingesta-e-indice.md) | Fase 2 | Embeddings locales, Chroma por colección, protección contra índices viejos | ✅ |
| [4: Búsqueda híbrida](04-fase-3-busqueda-hibrida.md) | Fase 3 | BM25 en español, fusión RRF, casos donde una búsqueda rescata a la otra | ✅ |
| [5: Golden dataset](05-fase-4-golden-dataset.md) | Fase 4 | 49 preguntas, evidencia literal, etiquetado por solapamiento | ✅ |
| [6: Generación con citas](06-fase-5-generacion-con-citas.md) | Fase 5 | Prompt, validación de citas, rechazo honesto, CLI | ✅ |
| [7: Suite de evals](07-fase-6-evals.md) | Fase 6 | Métricas de retrieval, juez de groundedness, diagnóstico por etapa | ✅ |
| [8: Decisión de chunking](08-fase-7-decision-chunking.md) | Fase 7 | Criterio previo, resultados completos, elección de `structural` | ✅ |
| [9: Re-ranking](09-fase-8-reranking.md) | Fase 8 | Cross-encoder multilingüe, delta medido y costo en latencia (extra credit) | ✅ |

**Apéndices**

- [A: Registro de decisiones (ADR)](apendice-a-decisiones.md), ADR-034 a ADR-049 (continúa la numeración de M1)
- [B: Glosario](apendice-b-glosario.md)

---

## Convenciones

- Los bloques ` ```bash ` se ejecutan **desde la raíz del repositorio**, con el entorno activado.
- Las referencias `src/retriever.py:113` significan *archivo:línea* del código vigente.
- Una **Decisión** es un trade-off registrado también en el Apéndice A.
- Los ids de pregunta (`q07`, `o02`) son los de `evals/dataset.jsonl`; los ids de chunk (`hr_policies.md_2`) dependen de la estrategia.
- Los números de resultados salen de los archivos de `evals/results/` que se citan en cada tabla.

---

## Cómo exportar el libro

El manual es Markdown plano. Si querés un PDF o un `.docx`:

```bash
pandoc docs/00-introduccion.md docs/01-fase-0-repositorio.md docs/02-fase-1-corpus-y-chunking.md \
  docs/03-fase-2-ingesta-e-indice.md docs/04-fase-3-busqueda-hibrida.md docs/05-fase-4-golden-dataset.md \
  docs/06-fase-5-generacion-con-citas.md docs/07-fase-6-evals.md docs/08-fase-7-decision-chunking.md docs/09-fase-8-reranking.md \
  docs/apendice-a-decisiones.md docs/apendice-b-glosario.md \
  -o manual.pdf --toc --number-sections
```

`pandoc` es opcional. El repositorio no depende de él.

---

## Historial del manual

| Versión | Fecha | Cambio |
|---|---|---|
| 1.0 | 2026-11-25 | Capítulos 0 a 8, apéndices A (ADR-034 a ADR-049) y B |
| 1.1 | 2026-11-25 | Capítulo 9 (Fase 8): re-ranking con cross-encoder, ADR-050 y ADR-051 |
