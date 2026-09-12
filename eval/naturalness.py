"""Juez de naturalidad/tono (LLM-as-judge) para el golden set de eia-rag.

Puntúa la NATURALIDAD de una respuesta del bot en escala 1-5, sin evaluar la
exactitud (esa dimensión la cubre el golden set con criterios debe_incluir/
no_debe). Dev-only: requiere la API key de Groq configurada.

Uso desde eval/eval.py con --judge (ver README de eval/).
"""

import re
from typing import Any

JUDGE_SYSTEM = """Eres un evaluador de respuestas de un chatbot de e-commerce en español.
Califica únicamente la NATURALIDAD y el TONO de la respuesta (ignora la exactitud).

Natural = suena a un vendedor real escribiendo en chat: frase variada, prioriza
1-2 productos relevantes, describe en una frase natural y hace una pregunta para
afinar. Poco natural = suena a plantilla o a salida de base de datos: enumera
fichas técnicas completas con el mismo patrón por cada producto, o a bot genérico.

Responde ÚNICAMENTE con JSON como este:
{"puntaje": 4, "razon": "una frase breve"}
Puntaje (1-5): 1 = muy robótico/plantilla, 3 = aceptable, 5 = conversación humana natural."""


def parse_score(raw: str) -> int:
    match = re.search(r'"puntaje"\s*:\s*(\d+)', raw)
    return int(match.group(1)) if match else 0


async def score_naturalness(query: str, answer: str, client: Any, model: str) -> dict:
    """Puntúa una respuesta en 1-5. Nunca lanza: devuelve 0 si falla."""
    if not answer.strip():
        return {"puntaje": 0, "razon": "respuesta vacía"}
    try:
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Consulta del cliente: {query}\n\n"
                        f"Respuesta del bot: {answer}"
                    ),
                },
            ],
            model=model,
            temperature=0.0,
            max_tokens=160,
        )
        raw = completion.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001 — fail-safe para eval
        return {"puntaje": 0, "razon": f"error: {e}"}
    return {"puntaje": parse_score(raw), "razon": raw}