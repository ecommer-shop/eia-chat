from groq import Groq
from app.config import settings

client = Groq(api_key=settings.GROQ_API_KEY)

def generate_final_response(query: str, intent: str, context_items: list) -> str:
    """
    Genera una respuesta amigable, profesional y 100% en español basada en el contexto RAG.
    """
    print(f"🧠 [LLM GENERATOR] Redactando respuesta para intención: {intent}")
    
    # 1. Formateo de Contexto optimizado para legibilidad del LLM
    context_text = ""
    if not context_items:
        context_text = "No se encontraron registros específicos en el catálogo o políticas."
    else:
        for i, item in enumerate(context_items, 1):
            payload = item.get("payload", {})
            if intent == "CATALOGO":
                nombre = payload.get('name', 'Producto sin nombre')
                Atributos = ", ".join(payload.get('attributes', []))
                url = payload.get('url', '#')
                context_text += f"\n- {nombre} (Atributos: {Atributos}). Link: {url}"
            else:
                texto_doc = payload.get('text', 'Sin contenido')
                context_text += f"\n- Referencia {i}: {texto_doc}"

    # 2. Construcción del Prompt con Personalidad (System Message)
    system_prompt = f"""
    PERSONALIDAD:
    Eres 'IA Engineering Assistant', el guía experto de nuestra tienda online. 
    Tu objetivo es que el cliente se sienta acompañado. Eres entusiasta, usas un lenguaje cercano (tuteo) y siempre resuelves dudas con amabilidad. Debes presentar links sobre el producto que hablas

    REGLAS DE ORO:
    1. IDIOMA: Responde SIEMPRE en Español, con un tono natural de Latinoamérica/España (neutro).
    2. FIDELIDAD: Usa SOLO la información del 'CONTEXTO' para dar detalles técnicos o de stock.
    3. PRECIOS/STOCK: Si el usuario pregunta por precios y no están en el contexto, di algo como: 
       "¡Buena elección! Por ahora no tengo el precio exacto aquí conmigo, pero puedo confirmarte que el modelo está en nuestro catálogo. ¿Te gustaría que te ayude con algo más sobre sus características?"
    4. SIN ALUCINACIONES: Si no hay contexto, no inventes. Invita al usuario a preguntar por otra categoría.
    5. FORMATO: Usa negritas para nombres de productos y listas para que sea fácil de leer.

    CONTEXTO ACTUAL DE LA BASE DE DATOS:
    {context_text}

    INTENCIÓN DETECTADA: {intent}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            # 🔥 CAMBIO AQUÍ: Usamos el modelo más reciente de Groq 🔥
            model="llama-3.3-70b-versatile", 
            temperature=0.3,
            max_tokens=500,
        )
        
        return chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ [LLM GENERATOR ERROR] Error en Groq: {e}")
        return "Lo siento, estoy teniendo problemas técnicos en este momento para procesar tu solicitud."