# Nubbix Assist — asistente RAG de conocimiento interno (Proyecto Integrador M2)

> En construcción. La versión final de este README incluye resultados de evals,
> la comparación de chunking y las decisiones técnicas.

Chatbot que responde preguntas de los empleados de Nubbix a partir de la documentación
oficial (HR, IT, Finance) y cita los chunks que usó.

```bash
uv venv --python 3.14 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
python -m ingest.ingest --strategy all
python -m src.main --query "¿Hasta qué día del mes puedo cargar una rendición de gastos?" --pretty
```
