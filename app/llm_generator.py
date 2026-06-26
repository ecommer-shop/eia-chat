from groq import Groq
from app.config import settings

client = Groq(api_key=settings.GROQ_API_KEY)

def generate_final_response(query: str, intent: str, context_items: list) -> str:
    """
    Genera una respuesta amigable, profesional y 100% en español basada en el contexto RAG.
    """
    print(f"🧠 [LLM GENERATOR] Redactando respuesta para intención: {intent}")
    
    # 1. Formateo de Contexto optimizado para legibilidad del LLM
    def _find_url_in_payload(p: dict) -> str:
        # Buscar en claves comunes y en valores anidados la primera URL HTTP válida
        if not isinstance(p, dict):
            return ""

        # Comprueba claves directas comunes
        for key in ("url", "link", "href", "source", "uri"):
            val = p.get(key)
            if isinstance(val, str) and val.startswith("http"):
                return val

        # Busca en metadatos anidados
        for meta_key in ("metadata", "meta", "extras", "data"):
            meta = p.get(meta_key)
            if isinstance(meta, dict):
                for v in meta.values():
                    if isinstance(v, str) and v.startswith("http"):
                        return v

        # Último recurso: inspecciona cualquier valor string dentro del dict
        for v in p.values():
            if isinstance(v, str) and v.startswith("http"):
                return v

        return ""

    context_text = ""
    if not context_items:
        context_text = "No se encontraron registros específicos en el catálogo o políticas."
    else:
        parts = []
        # detectar si la intención contiene ambas partes
        is_catalogo = isinstance(intent, str) and "CATALOGO" in intent
        is_politicas = isinstance(intent, str) and "POLITICAS" in intent

        for i, item in enumerate(context_items, 1):
            payload = item.get("payload", {}) or {}

            # Heurística: si tiene 'name' o 'title' lo tratamos como producto
            if payload.get('name') or payload.get('title') or (is_catalogo and not payload.get('text')):
                nombre = payload.get('name') or payload.get('title') or 'Producto sin nombre'
                atributos = payload.get('attributes') or payload.get('atributos') or []
                atributos_text = ", ".join(atributos) if isinstance(atributos, (list, tuple)) else str(atributos)
                url = _find_url_in_payload(payload)
                segment = f"👉 Producto: {nombre}"
                if url:
                    segment += f" 🔗 {url}"
                if atributos_text and atributos_text != "[]":
                    segment += f" 💡 {atributos_text}"
                parts.append(segment)
                continue

            # Si tiene 'text' o similares, lo tratamos como política/documento
            texto_doc = payload.get('text') or payload.get('content') or payload.get('body') or payload.get('policy')
            if texto_doc:
                # Resumir o incluir el texto breve
                snippet = texto_doc if isinstance(texto_doc, str) else str(texto_doc)
                parts.append(f"👉 Política: {snippet}")
                continue

            # Fallback: intenta extraer alguna URL o representación
            url = _find_url_in_payload(payload)
            if url:
                parts.append(f"👉 Link: {url}")

        # Unir todo en una sola línea (sin saltos) para que el LLM lo lea según las reglas
        context_text = " ".join(parts) if parts else "No se encontraron registros específicos en el catálogo o políticas."

    # --- DEBUG LOGS (temporal) ---
    try:
        print("🔍 [DEBUG] context_items raw:", context_items)
        extracted_urls = []
        for item in context_items:
            p = (item or {}).get('payload') or {}
            extracted_urls.append(_find_url_in_payload(p))
        print("🔍 [DEBUG] extracted_urls:", extracted_urls)
        print("🔍 [DEBUG] context_text:", context_text)
    except Exception as _e:
        print("🔍 [DEBUG] fallo al imprimir debug:", _e)

    # 2. Construcción del Prompt con Personalidad (System Message)
    system_prompt = f"""
    PERSONALIDAD:
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

    INTENCIÓN DETECTADA: {intent}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            
            model="llama-3.3-70b-versatile", 
            temperature=0.2,
            max_tokens=150,
        )
        
        respuesta = chat_completion.choices[0].message.content.strip()
        respuesta = respuesta.replace("\n", " ").replace("\r", "")
        return respuesta
    
        
    except Exception as e:
        print(f"❌ [LLM GENERATOR ERROR] Error en Groq: {e}")
        return "Lo siento, estoy teniendo problemas técnicos en este momento para procesar tu solicitud."
