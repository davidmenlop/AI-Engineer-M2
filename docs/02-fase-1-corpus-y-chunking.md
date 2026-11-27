# Capítulo 2: Corpus y chunking (Fase 1)

> Comparar dos estrategias casi iguales no enseña nada. Acá se comparan dos formas
> opuestas de cortar: ignorando la estructura y respetándola.

---

## 1. Objetivo

Leer el corpus antes de decidir nada, e implementar dos estrategias de chunking que se diferencien de verdad, seleccionables por parámetro y con suficiente información en cada chunk para citar y para evaluar.

## 2. Resultado

- `ingest/chunking.py`: `chunk_fixed`, `chunk_structural` y la dataclass `Chunk`.
- `data/chunks/fixed.jsonl` (35 chunks) y `data/chunks/structural.jsonl` (18 chunks), generados por la ingesta.

## 3. Lectura del corpus

| Documento | Secciones `##` |
|---|---|
| `hr_policies.md` | Licencias y Vacaciones · Modalidad de Trabajo y Home Office · Licencias por Enfermedad y Ausentismo · Cumpleaños y Reconocimientos · Feriados y Días No Laborables · Beneficios |
| `it_manual.md` | Solicitud de Accesos y Permisos · Equipos de Trabajo y Hardware · Políticas de Contraseñas y MFA · Seguridad de Datos y Backups · Soporte Técnico y Mesa de Ayuda · Offboarding y Baja de Colaboradores |
| `finance_processes.md` | Reintegro de Gastos y Viáticos · Solicitud de Compras de Insumos y Licencias · Fechas y Proceso de Cobro de Haberes · Presupuesto de Capacitación y Reembolsos Educativos · Bonos y Compensación Variable · Facturación y Comprobantes Fiscales |

Lo que importa para el diseño:

- **Corpus chico y muy estructurado:** 18 secciones con cuerpos de 285 a 534 caracteres. Cada sección trata un solo tema.
- **Datos exactos que la búsqueda vectorial suele perder:** canales (`#people-avisos`, `#it-ops`, `#it-help`, `#nubbix-family`), plataformas (ExpenseHub, Google Authenticator, 1Password), montos ($45,000 ARS, $25,000 ARS, USD 500, USD 200/2000).
- **Distractores dentro de una misma sección:** "Licencias y Vacaciones" lista nacimiento, matrimonio, fallecimiento y estudio, cada uno con otro plazo. "Solicitud de Compras" tiene tres rangos de monto con aprobadores distintos.
- **Temas que cruzan documentos:** el crédito de capacitación (monto en HR, rendición en Finance); el robo de un equipo en un viaje (pasos en IT Equipos, reemplazo en IT Soporte, reintegro en Finance); la VPN en un aeropuerto y el tope de comidas en viaje.
- **Consecuencia directa:** cortar por encabezados da 18 chunks, menos que los 20 que exige la consigna. En esta fase se dio por cubierto el mínimo con la estrategia de tamaño fijo. Fue un error de lectura: el mínimo tiene que cumplirlo la estrategia que usa el asistente. Se corrigió en el Capítulo 10 (ADR-052) bajando `max_chars` a 590, con lo que `structural` da 20 chunks.

## 4. Las dos estrategias

### 4.1 `fixed`: ventana de tamaño fijo con solapamiento

Ventana de 300 caracteres que avanza dejando 60 de solapamiento, sobre el texto crudo del documento, encabezados incluidos. **Ignora la estructura a propósito**: es la línea base "ingenua". Solo evita cortar palabras.

- 300 caracteres ≈ 2–3 oraciones del corpus: un chunk alcanza para un dato con su contexto inmediato.
- 60 de solapamiento (20 %) es el rango habitual: una oración partida en el borde aparece entera en el chunk siguiente.
- Resultado: **35 chunks** (12 de HR, 11 de IT, 12 de Finance), de 108 a 299 caracteres, 285 en promedio.

### 4.2 `structural`: un chunk por sección

Un chunk por sección `##`, con el título del documento y el de la sección antepuestos para que el chunk se entienda solo. Si una sección supera `max_chars` (1.200) se subdivide por líneas completas repitiendo el encabezado en cada parte. En este corpus ninguna sección llega a ese límite.

- Resultado: **18 chunks** (6 por documento), de 314 a 598 caracteres, 486 en promedio.

### 4.3 Cómo se ven (de `data/chunks/*.jsonl`)

`fixed`, `hr_policies.md_1` (caracteres 237–534):

```
menos 15 días de anticipación.
Existen licencias especiales remuneradas:
- Nacimiento o adopción: 15 días corridos para el progenitor no gestante.
- Matrimonio: 10 días corridos.
- Fallecimiento de familiar directo (padres, hijos, cónyuge): 3 días hábiles.
- Días por estudio: Hasta 10 días al año
```

Empieza en la mitad de una oración, no dice de qué documento ni de qué sección es, y corta la línea de "Días por estudio" en la mitad. `hr_policies.md_2` arranca con "3 días hábiles." y mezcla el final de "Licencias y Vacaciones" con el principio de "Modalidad de Trabajo y Home Office".

`structural`, `hr_policies.md_2` (caracteres 1070–1627):

```
# Políticas de Recursos Humanos - Nubbix
## Licencias por Enfermedad y Ausentismo
Ante una enfermedad o accidente, el empleado debe avisar a su Manager y a People antes de las
10 AM del día de la ausencia, utilizando el canal de Slack #people-avisos.
Las ausencias de hasta 3 días corridos se justifican cargando una declaración jurada ...
Las ausencias de 4 días o más requieren certificado médico digitalizado ...
Nubbix no descuenta haberes por licencias médicas debidamente justificadas ...
```

Un tema completo, con su encabezado.

### 4.4 Por qué difieren de verdad

| | `fixed` | `structural` |
|---|---|---|
| Unidad | caracteres | sección temática |
| Contexto propio | ninguno (sin encabezado) | documento + sección |
| Mezcla temas | sí, en los bordes | nunca |
| Chunks | 35 | 18 |
| Precisión potencial | alta: chunks chicos | baja: un chunk trae toda la sección |
| Riesgo | un dato partido entre dos chunks | una sección larga diluye el embedding |

La hipótesis a medir: en un corpus de secciones cortas y temáticas, respetar la estructura recupera mejor la evidencia completa, a costa de pasarle al modelo más texto por chunk.

## 5. Código explicado

### Metadatos de cada chunk (`ingest/chunking.py:27`)

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # "<source>_<n>", n correlativo por documento y estrategia
    source: str        # documento de origen: lo que permite rastrear la cita
    section: str       # sección donde empieza el chunk
    strategy: str
    text: str          # lo que se embebe, se indexa y se muestra como contenido
    char_start: int    # tramo [char_start, char_end) del documento original
    char_end: int
```

`char_start`/`char_end` es la pieza que permite evaluar las dos estrategias con la misma regla: el golden dataset etiqueta la evidencia como un tramo del documento, y un chunk es relevante si su tramo se solapa con él (Capítulo 5). En `fixed`, `text` es exactamente `documento[char_start:char_end]`. En `structural`, `text` agrega el encabezado sintético y el tramo cubre desde la línea `##` hasta el final de la sección.

En `fixed`, `section` es la última sección `##` que empieza en o antes del inicio del chunk (`_section_at`, `ingest/chunking.py:55`). Un chunk que empieza antes del primer `##` (como `hr_policies.md_0`) lleva el título del documento.

### Corte en límite de palabra (`ingest/chunking.py:87`)

```python
cut = text.rfind(" ", position + size_chars // 2, end)
cut_nl = text.rfind("\n", position + size_chars // 2, end)
cut = max(cut, cut_nl)
```

Retrocede hasta el último espacio o salto de línea, sin achicar el chunk a menos de la mitad. El inicio del chunk siguiente (`ingest/chunking.py:108`) retrocede `overlap_chars` y avanza hasta el comienzo de una palabra.

### Encabezado repetido (`ingest/chunking.py:123`)

```python
header = f"# {title}\n## {heading}\n"
for part_start, part_end in _split_body(text, body_start, end, max_chars - len(header)):
```

El presupuesto de cada parte descuenta el encabezado, así que ninguna parte supera `max_chars` aunque la sección se subdivida.

## 6. Verificación

```bash
python tests/test_chunking.py | tail -1     # 9/9 tests pasaron
```

Los tests comprueban, entre otras cosas, que `fixed` no corta palabras y que su `text` coincide con el tramo original, que `structural` subdivide repitiendo el encabezado, y que el corpus real da al menos 20 chunks con `fixed` y exactamente 18 con `structural`.

## 7. Decisiones

- **Dos estrategias opuestas** en vez de dos tamaños de ventana: la consigna pide que se diferencien.
- **`fixed` no recibe encabezado**, aunque la guía recomienda repetirlo al subdividir: agregárselo la acercaría a `structural` y la comparación perdería contraste. La recomendación se aplica en `structural`, que es donde una sección se subdivide.
- **Parámetros fijados antes de medir** y no ajustados después. Probar otros tamaños de `fixed` queda como trabajo futuro.

## 8. Checklist de cierre

- [x] Corpus leído; temas cruzados y distractores identificados.
- [x] Dos estrategias seleccionables por parámetro.
- [x] Al menos 20 chunks con una estrategia (`fixed`: 35). Corregido en el Capítulo 10: ahora también `structural` (20), que es la que usa el asistente.
- [x] Cada chunk con documento, sección, estrategia y tramo.
- [x] Chunks leídos a ojo antes de indexar.
