# Capítulo 4: Búsqueda híbrida (Fase 3)

> La búsqueda vectorial entiende que "obra social" es "medicina prepaga" pero pierde la
> pregunta del bar con wifi; BM25 la encuentra por "conectarme" pero no sabe de sinónimos.
> Combinadas, se cubren.

---

## 1. Objetivo

Servir búsqueda léxica + vectorial con un método de combinación documentado, y mostrar con casos concretos que la combinación aporta.

## 2. Resultado

- `src/lexical.py`: tokenización para español y `BM25Index`.
- `src/retriever.py`: `Retriever.retrieve(query, top_k, mode)` con `mode ∈ {hybrid, vector, bm25}` y la función `rrf_fuse`.
- `python -m src.main --mode vector|bm25|hybrid` para comparar a mano.

## 3. BM25 con normalización para español (ADR-038)

BM25 puntúa un chunk por los términos de la consulta que contiene, pesados por lo raros que son en el corpus (IDF) y con saturación por frecuencia. Funciona tan bien como su tokenización. `src/lexical.py:46`:

```python
def tokenize(text: str) -> list[str]:
    """minúsculas, sin tildes, sin palabras vacías, con stemming (licencias -> licenci)."""
    words = [w for w in TOKEN.findall(normalize(text)) if w not in STOPWORDS]
    return _stemmer().stemWords(words)
```

- **Sin tildes ni mayúsculas:** "Días Hábiles" y "dias habiles" son lo mismo.
- **Palabras vacías:** una lista corta (`src/lexical.py:23`). Números y siglas (USD, ARS, VPN, MFA) se conservan.
- **Stemming Snowball en español:** "licencias" y "licencia" comparten raíz. Sin esto, BM25 falla con plurales y conjugaciones.
- **Canales de Slack:** `#people-avisos` se tokeniza como `peopl`, `avis`; alcanza para encontrarlo. El costo: "wifi" no coincide con "Wi-Fi" (`wi`, `fi`), porque el guion separa tokens.

Los tokens se calculan en la ingesta y se guardan en `data/index/bm25_<estrategia>.json`. La consulta los lee y arma `BM25Okapi` en memoria (milisegundos): no reprocesa el corpus. `search` descarta los chunks con puntaje ≤ 0, que no comparten ningún término informativo con la consulta (`src/lexical.py:84`).

## 4. Fusión con Reciprocal Rank Fusion (ADR-039)

La similitud coseno vive en [0, 1] y BM25 en [0, ∞). Sumarlas exige normalizar y elegir un peso. RRF evita las dos cosas: usa posiciones, no puntajes.

```
RRF(chunk) = Σ  1 / (k + posición del chunk en el ranking r),   k = 60
             r ∈ {vectorial, BM25}
```

`src/retriever.py:113` (`retrieve`), modo `hybrid`:

1. La vectorial aporta sus 10 mejores candidatos (`candidates` en `rag.yaml`).
2. BM25 aporta los suyos (puede traer menos de 10 si pocos chunks comparten términos).
3. `rrf_fuse` suma las contribuciones y ordena.
4. Se devuelven los `top_k` (3) primeros.

`k = 60` es el valor del paper original (Cormack, Clarke y Büttcher, 2009). No se ajustó: con 49 preguntas, calibrarlo sería sobreajustar.

### El score que se devuelve

En modo híbrido, `score` es el RRF dividido por su máximo posible (`2 / 61`, primero en ambos rankings), en [0, 1] (`src/retriever.py:139`). En los otros modos es la similitud coseno o el puntaje BM25 crudo.

**El score híbrido casi no discrimina.** Con 18 chunks y 10 candidatos por buscador, casi todos los chunks aparecen en los dos rankings. Un chunk en la posición 5 de ambos obtiene `2/65 ÷ 2/61 = 0,94`. En la práctica los tres recuperados quedan entre 0,93 y 1,0. Sirve para ordenar, no como medida de confianza, y por eso el rechazo honesto no usa un umbral de score (ADR-044).

## 5. Casos donde una búsqueda rescata a la otra

De `evals/results/retrieval_20261125T164259Z.json`, estrategia `structural`, k = 3, evidence_recall por pregunta:

| Pregunta | vectorial | BM25 | híbrida | Qué pasó |
|---|---|---|---|---|
| q07 "¿Me dan algún monto fijo por mes para pagar la conexión cuando trabajo desde casa?" | 1,00 | 0,00 | 1,00 | sinónimo: "conexión… desde casa" ↔ "estipendio… conectividad… Home Office" |
| q16 "¿La obra social también cubre a mis hijos?" | 1,00 | 0,00 | 1,00 | sinónimo: "obra social" ↔ "medicina prepaga" |
| q21 "Se me cayó la laptop de la empresa y no prende más" | 1,00 | 0,00 | 1,00 | sinónimo: "laptop" ↔ "equipo corporativo", "daño grave" |
| q24 "¿Le puedo pasar mi clave a un compañero por mensaje privado?" | 1,00 | 0,00 | 1,00 | sinónimo: "clave" ↔ "credenciales" |
| q12 "Si falto por enfermedad y presento todo en regla, ¿me descuentan ese día del sueldo?" | 0,00 | 1,00 | 1,00 | término compartido: "descuentan" y "descuenta" dan el mismo stem, `descuent` |
| q25 "Estoy en un bar con wifi abierto, ¿tengo que hacer algo antes de conectarme a trabajar?" | 0,00 | 1,00 | 1,00 | término compartido: "conectarme" y "conectarse" dan `conect`. Ojo: "wifi" **no** coincide con "Wi-Fi" (`wi`, `fi`) |
| q29 robo de notebook en un viaje (cruza documentos) | 0,25 | 1,00 | 0,75 | BM25 trae la sección de Soporte; la fusión deja afuera una parte |

La vectorial rescata los **sinónimos**; BM25 rescata las preguntas que comparten una palabra poco frecuente con la evidencia, gracias al stemming. En `structural` con k = 3, la híbrida se queda con lo mejor de ambas en los seis primeros casos.

q29 muestra el límite: BM25 solo cubría las cuatro evidencias, pero al fusionar, `hr_policies.md_1` (Home Office), que aparecía en posiciones intermedias de los dos rankings, sumó más RRF que `it_manual.md_4` (Soporte Técnico, el que habla del reemplazo y el reintegro en viajes), primero en BM25 pero lejos en la vectorial, y lo dejó fuera del top 3. La fusión no es la unión: es un compromiso. En q42 (VPN en el aeropuerto) ningún modo trae la sección de Seguridad de Datos en el top 3: ahí falla el retrieval entero.

## 6. Los tres modos, en números

Misma corrida, k = 3, 42 preguntas dentro de alcance:

| estrategia | modo | evidence_recall@3 | MRR | precision@3 |
|---|---|---|---|---|
| structural | **hybrid** | **0,982** | **0,897** | 0,373 |
| structural | vector | 0,923 | 0,849 | 0,349 |
| structural | bm25 | 0,893 | 0,821 | 0,349 |
| fixed | **hybrid** | **0,802** | 0,762 | 0,452 |
| fixed | vector | 0,736 | 0,790 | 0,413 |
| fixed | bm25 | 0,724 | 0,706 | 0,397 |

La híbrida gana en evidence_recall en las dos estrategias. En `fixed`, la vectorial sola tiene mejor MRR (0,790 contra 0,762): la fusión a veces baja el chunk correcto del primer lugar al segundo. Con k = 1 en `fixed`, la vectorial también supera a la híbrida (evidence_recall 0,566 contra 0,476). Esa es la razón para usar k = 3 y no k = 1.

Latencia del retrieval (p50): unos 12 ms en modo vectorial o híbrido (casi todo es el embedding de la pregunta) y 0,2 ms en BM25.

## 7. Verificación

```bash
python tests/test_retrieval.py | tail -1   # 10/10 tests pasaron
python -m src.main --query "Estoy en un bar con wifi abierto" --mode vector --retrieve-only
python -m src.main --query "Estoy en un bar con wifi abierto" --mode bm25 --retrieve-only
```

Los tests verifican la aritmética de RRF, que el score híbrido máximo es 1,0 y, con dobles de la vectorial, que la híbrida rescata un chunk que la vectorial ubica último y BM25 primero.

## 8. Decisiones

- **RRF y no suma ponderada:** sin hiperparámetros que calibrar con un dataset chico.
- **10 candidatos por buscador:** más de la mitad del corpus en `structural`. Con un corpus grande este número tendría que crecer junto con el corpus.
- **BM25 propio y no el de la vector DB:** Chroma local no trae BM25; mantenerlo aparte lo hace inspeccionable y probable sin red.

## 9. Checklist de cierre

- [x] Búsqueda léxica y vectorial combinadas, con el método y sus parámetros documentados.
- [x] Casos concretos en ambas direcciones.
- [x] Los tres modos medidos con el mismo dataset y k.
