from groq import Groq
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# The Groq client uses the API key from environment via settings; never log the key.
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
    
    # Formatear contexto RAG y detectar enlaces
    context_fragments = []
    link_fragments = []
    for i, item in enumerate(context):
        payload = item.get('payload', {})
        # Build fragment text
        fragment_text = item.get('content', '')
        if fragment_text:
            context_fragments.append(f"[Fragmento {i+1}]: {fragment_text}")
        # Detect URL
        url = payload.get('url')
        if url:
            link_fragments.append(url)
    context_text = "\n\n".join(context_fragments) or "Sin contexto adicional disponible."
    if link_fragments:
        context_text += "\nEnlaces: " + ", ".join(link_fragments)

    # Etiqueta de canal para el system prompt
    channel_labels = {
        "web": "chat web de la tienda",
        "whatsapp": "WhatsApp",
        "instagram": "Instagram Direct",
    }
    channel_label = channel_labels.get(channel, channel)

    system_prompt = f"""Eres un asistente de ventas amable y experto para una tienda de e-commerce. El usuario te escribe desde el {channel_label}.

 Eres 'IA Engineering Assistant', el guía experto de nuestra tienda online, tienes informacion sobre productos, parte legal de la tienda y ciertas cosas sobre su funcionamiento, te alimentan a traves de un RAG. 
    Tu objetivo es que el cliente se sienta acompañado. Eres entusiasta, usas un lenguaje cercano (tuteo) y siempre resuelves dudas con amabilidad. Debes presentar links sobre el producto que hablas, sino contiene ningun link evadelo con naturalidad. Siempre habla de forma amigable con el usuario.

    REGLAS DE ORO:
    1. IDIOMA: Responde SIEMPRE en Español, con un tono natural de Latinoamérica/España (neutro).
    2. Devuelve solo texto plano, sin formato Markdown ni negritas.
    3. No uses asteriscos, backticks, guiones de lista ni ningún símbolo de Markdown.
    4. FIDELIDAD: Usa SOLO la información del 'CONTEXTO' para dar detalles técnicos o de stock.
    5. PRECIOS/STOCK: Si el usuario pregunta por precios y no están en el contexto, di algo como: 
       "¡Buena elección! Por ahora no tengo el precio exacto aquí conmigo, pero puedo confirmarte que el modelo está en nuestro catálogo. ¿Te gustaría que te ayude con algo más sobre sus características?"
    6. SIN ALUCINACIONES: Si no hay contexto, no inventes. Invita al usuario a preguntar por otra categoría.
    7. FORMATO: NUNCA uses saltos de línea. Usa emojis como separadores:
       - 👉 antes de cada producto mencionado
       - 🔗 antes de cada enlace
       - 💡 antes de tips o información adicional
    8. ENLACES: Siempre que menciones un producto, incluye su enlace directo después de 🔗

    9.ENLACES RESTRINGIDOS: Usa ÚNICAMENTE las URLs que aparecen en el CONTEXTO. 
      NUNCA inventes, modifiques o adivines URLs. Si no hay URL en el contexto 
      para un producto, NO incluyas ningún link.

    10.Siempre debes dar respuestas cortas, de maximo 150 tokens, entonces debes limitar tu respuesta para que no quede cortada ni se vea raro. 

    CONTEXTO ACTUAL DE LA BASE DE DATOS:
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
            max_tokens=300,
        )

        answer = response.choices[0].message.content
        logger.info("✅ Respuesta generada exitosamente.")
        return answer
    except Exception as e:
        logger.error(f"❌ Error generando respuesta: {e}")
        return "Lo siento, estoy teniendo problemas técnicos en este momento para procesar tu solicitud."

