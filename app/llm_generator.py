from groq import Groq
from app.config import settings
import logging

logger = logging.getLogger(__name__)

client = Groq(api_key=settings.GROQ_API_KEY)

async def generate_response(
    query: str,
    context: list[dict],
    intent: str,
    channel: str,
    history: list[dict],
) -> str:
    """
    Genera la respuesta final usando Groq con historial y contexto RAG.
    """
    
    # Formatear contexto RAG
    context_text = "\n\n".join(
        f"[Fragmento {i+1}]: {item.get('content', '')}"
        for i, item in enumerate(context)
    ) or "Sin contexto adicional disponible."

    # Etiqueta de canal para el system prompt
    channel_labels = {
        "web": "chat web de la tienda",
        "whatsapp": "WhatsApp",
        "instagram": "Instagram Direct",
    }
    channel_label = channel_labels.get(channel, channel)

    system_prompt = f"""Eres un asistente de ventas amable y experto para una tienda de e-commerce. El usuario te escribe desde el {channel_label}.

CONTEXTO RELEVANTE RECUPERADO:
{context_text}

INSTRUCCIONES:
- Responde SIEMPRE en español.
- Sé conciso, amable y útil.
- Usa el contexto para responder con precisión sobre productos, políticas o información general.
- Si el usuario pregunta algo que no está en el contexto, sé honesto y ofrece alternativas.
- Recuerda el historial de la conversación para dar respuestas coherentes.
- Intención detectada: {intent}
- Nunca uses markdown ni formatos especiales.
- Máximo 150 tokens en la respuesta."""

    # Construir mensajes: system + historial + mensaje actual
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)  # Últimos 5 mensajes
    messages.append({"role": "user", "content": query})

    logger.info(f"🧠 Generando respuesta | canal: {channel} | intent: {intent} | historial: {len(history)} msgs")

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=800,
        )

        answer = response.choices[0].message.content
        logger.info("✅ Respuesta generada exitosamente.")
        return answer
    except Exception as e:
        logger.error(f"❌ Error generando respuesta: {e}")
        return "Lo siento, estoy teniendo problemas técnicos en este momento para procesar tu solicitud."

