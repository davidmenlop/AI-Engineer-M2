# Apéndice B: Glosario

> Términos de RAG que aparecen en el manual, en el orden en que conviene entenderlos.
> Los de LLM y prompting (token, system prompt, few-shot, temperatura) están en el
> glosario de M1.

---

## Arquitectura RAG

**RAG (Retrieval-Augmented Generation).** Patrón que, antes de generar, recupera fragmentos de una base documental y se los pasa al modelo como contexto. Conecta un LLM con conocimiento privado y actualizable sin reentrenarlo.

**Corpus.** El conjunto de documentos sobre el que se responde. Acá: `hr_policies.md`, `it_manual.md` y `finance_processes.md`.

**Ingesta.** Proceso que convierte el corpus en un índice consultable: carga, chunking, embeddings, indexación. Se corre una vez por estrategia, separado de la consulta.

**Chunk.** Fragmento de un documento que se indexa y se recupera como unidad. Lleva metadatos: documento de origen, sección, estrategia y tramo de caracteres.

**Chunking.** La forma de cortar el corpus en chunks. Define qué puede recuperarse junto y qué queda separado.

**Chunking de tamaño fijo (`fixed`).** Ventanas de N caracteres con solapamiento, sin mirar la estructura del documento.

**Solapamiento (overlap).** Caracteres que comparten dos chunks consecutivos, para que una oración cortada en el borde aparezca entera en alguno.

**Chunking estructural (`structural`).** Cortes por la estructura del documento (acá, secciones `##`), con el encabezado repetido para que cada chunk se entienda solo.

**Tramo (`char_start`, `char_end`).** Posición del chunk en el documento original. Permite comparar chunks y evidencia de estrategias distintas con la misma regla.

## Representación y búsqueda

**Embedding.** Vector numérico que representa el significado de un texto. Textos de significado parecido quedan cerca. Se usa el mismo modelo para indexar y para consultar.

**Similitud coseno.** Coseno del ángulo entre dos vectores; 1 = misma dirección. Chroma devuelve distancia coseno: `similitud = 1 − distancia`.

**Vector DB.** Base que indexa embeddings y devuelve los más cercanos a una consulta. Acá, Chroma local persistente, con una colección por estrategia.

**HNSW (Hierarchical Navigable Small World).** Grafo de vecinos que usa Chroma para la búsqueda aproximada de vecinos cercanos. Con 18–35 chunks la aproximación no tiene efecto práctico.

**Búsqueda léxica.** Búsqueda por coincidencia de términos. Encuentra nombres exactos, montos y canales; falla con sinónimos.

**BM25.** Función de puntaje léxico: suma, por cada término de la consulta, su frecuencia en el chunk (con saturación) por su rareza en el corpus (IDF), normalizada por el largo del chunk.

**IDF (Inverse Document Frequency).** Peso de un término según en cuántos chunks aparece: cuanto más raro, más informativo.

**Stemming.** Reducir palabras a una raíz común ("licencias", "licencia" → `licenci`). Acá, Snowball en español.

**Palabras vacías (stopwords).** Palabras muy frecuentes y poco informativas ("de", "la", "que") que se quitan antes de BM25.

**Búsqueda híbrida.** Combinar la búsqueda léxica y la vectorial para cubrir los fallos de cada una.

**RRF (Reciprocal Rank Fusion).** Método de fusión de rankings: cada documento suma `1 / (k + posición)` por cada ranking en que aparece. Usa posiciones y no puntajes, así que no hay que calibrar escalas. Acá, k = 60.

**Candidatos.** Chunks que aporta cada buscador antes de fusionar (acá, 10).

**top_k.** Cantidad de chunks que se pasan al modelo (acá, 3). Las estrategias se comparan con el mismo k.

## Evaluación

**Golden dataset.** Conjunto de preguntas con su respuesta esperada y su evidencia etiquetada, contra el que se mide el sistema. Acá, 49 preguntas en `evals/dataset.jsonl`.

**Evidencia.** Fragmento literal del corpus que contiene la respuesta a una pregunta. Es independiente de la estrategia de chunking.

**Chunks esperados (`expected_chunks`).** Los chunks de una estrategia que se solapan con la evidencia en al menos 30 caracteres (o la evidencia entera si es más corta). Se calculan, no se escriben a mano.

**Distractor.** Información parecida a la respuesta que convive con ella (la licencia por matrimonio junto a la de nacimiento) o que se parece a algo que no está (licencia por mudanza).

**Precision@k.** Chunks relevantes entre los k recuperados, dividido k. Mide el ruido. Tiene techo: con un solo chunk relevante y k = 3, no supera 0,33.

**Recall@k.** Chunks relevantes recuperados dividido chunks relevantes esperados. Mide lo que faltó, a nivel chunk.

**evidence_recall@k.** Evidencias cubiertas dividido evidencias, donde una evidencia está cubierta si los chunks recuperados cubren al menos el 80 % de sus caracteres. Mide si el dato llegó al modelo; no castiga que un dato aparezca repetido en dos chunks solapados. Es la métrica principal de este proyecto.

**complete@k.** 1 si todas las evidencias de la pregunta quedaron cubiertas. Importa en las preguntas que cruzan documentos.

**Hit rate (hit@k).** Proporción de preguntas con al menos un chunk relevante recuperado.

**MRR (Mean Reciprocal Rank).** Promedio de `1 / posición del primer chunk relevante`. Mide si lo relevante llega primero.

**Groundedness.** Proporción de las afirmaciones de una respuesta que se sostienen en los chunks que cita. No mide si la respuesta está completa ni si es correcta respecto de la referencia.

**Afirmación atómica (claim).** Un dato verificable de la respuesta (un monto, un plazo, un canal, un requisito). El juez evalúa una por una.

**LLM-as-judge.** Usar un LLM para evaluar la salida de otro. Acá, Claude Sonnet 5.5 juzga las respuestas de Claude Haiku 5.5. Hay que auditarlo: puede ser más estricto o más laxo que una persona.

**Diagnóstico por etapa.** Clasificación de cada pregunta fallida como fallo de retrieval (la evidencia no llegó), de generación (llegó y la respuesta no se sostiene o rechaza) o de ambas.

**Rechazo honesto.** Responder que la información no está en la documentación en lugar de inventar. Acá, con un texto fijo que se detecta automáticamente.

**Falso rechazo.** Rechazar una pregunta cuya respuesta sí estaba en la documentación.

**Congelar el dataset.** Commitear el dataset y el criterio de decisión antes de la primera medición, para que la elección no se ajuste a los resultados.
