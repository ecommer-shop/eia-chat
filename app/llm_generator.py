from app.config import settings
from app.templates.prompts import SYSTEM_PROMPT_TEMPLATE


def _format_context(context_items: list[dict], intent: str) -> str:
    """Serializa los items recuperados para el prompt del LLM.

    - NO incluye URLs: el renderer agrega los links en postproceso cuando el
      LLM menciona el producto, y la regla del template le prohíbe escribirlas.
      Así el modelo no gasta tokens ni tentación de inventar links.
    - Los snippets de documento se truncan a `RETRIEVER_DOC_MAX_CHARS` para
      que una política larga no dispare el prompt (prefill = latencia).
    """
    if not context_items:
        return "No se encontraron registros en la base de datos."

    parts: list[str] = []
    is_catalogo = "CATALOGO" in intent
    max_doc_chars = settings.RETRIEVER_DOC_MAX_CHARS

    for item in context_items:
        payload = item.get("payload", {}) or {}

        if payload.get("name") or payload.get("title") or (is_catalogo and not payload.get("text")):
            name = payload.get("name") or payload.get("title") or "Producto sin nombre"
            attrs = payload.get("attributes") or payload.get("atributos") or []
            attrs_text = ", ".join(attrs) if isinstance(attrs, (list, tuple)) else str(attrs)
            segment = f">> Producto: {name}"
            if attrs_text and attrs_text != "[]":
                segment += f" Info: {attrs_text}"
            parts.append(segment)
            continue

        text = payload.get("text") or payload.get("content") or payload.get("body") or payload.get("policy")
        if text:
            snippet = text if isinstance(text, str) else str(text)
            if max_doc_chars > 0 and len(snippet) > max_doc_chars:
                snippet = snippet[:max_doc_chars].rsplit(" ", 1)[0] + "…"
            parts.append(f">> Documento: {snippet}")
            continue

    return " ".join(parts) if parts else "No se encontraron registros en la base de datos."


def build_prompt(
    query: str,
    intent: str,
    context_items: list[dict],
    history: list[dict],
    store_prompt: str,
    nocontext_guard: bool = False,
    few_shot: list[dict] | None = None,
) -> list[dict]:
    context_text = _format_context(context_items, intent)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        store_prompt=store_prompt,
        context_text=context_text,
        intent=intent,
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    for example in (few_shot or [])[-4:]:
        if isinstance(example, dict) and example.get("role") in ("user", "assistant"):
            messages.append({
                "role": example["role"],
                "content": example.get("content", ""),
            })

    if nocontext_guard and not context_items:
        messages.append({
            "role": "system",
            "content": (
                "No se recuperaron coincidencias del catálogo ni de documentos para esta consulta. "
                "No inventes datos, precios ni URLs, y no afirmes que no existen productos ni que no hay artículos. "
                "Si el usuario pregunta por un producto o categoría, responde con honestidad que no vimos "
                "coincidencias exactas y ofrécele intentar con otra palabra, categoría o marca. "
                "Pide más detalle si es necesario."
            ),
        })

    for msg in history[-6:]:
        role = msg.get("role", "user")
        messages.append({"role": role, "content": msg.get("content", "")})

    messages.append({"role": "user", "content": query})
    return messages
