"""Guardrail de grounding (post-generación).

Verifica que los productos y enlaces mencionados por el LLM en una respuesta
estén respaldados por el CONTEXTO del turno actual o por productos ya
mostrados en el historial de la conversación. Si el modelo inventó algo
(p. ej. un "iPhone 15" inexistente), la respuesta se reemplaza por un
mensaje honesto y determinista.

Razones de diseño:
- Es independiente del prompt: un modelo puede ignorar "no inventes", por eso
  se valida en código lo que efectivamente se va a enviar.
- Nombres: comparación normalizada (sin acentos/mayúsculas) y por subcadena
  para tolerar que el modelo abrevie ("KZ Castor" dentro de "KZ CASTOR PRO
  BASS EDITION").
- URLs: solo se permiten las que aparecen en el contexto actual o en el
  historial del mismo `conversation_id`.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_ARROW_NAME_RE = re.compile(r"👉\s*([^🔗💡\n\r]+)")
_PRODUCT_NAME_IN_TEXT_RE = re.compile(r"Producto:\s*([^.\n:]+)")

# Mensaje honesto cuando el LLM menciona algo que no está en el catálogo.
FALLBACK_UNGROUNDED = (
    "No encuentro ese producto en nuestro catálogo en este momento. "
    "¿Te gustaría que busque algo parecido o que te cuente qué tenemos disponible?"
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c.lower() for c in text if c.isalnum() or c.isspace()).strip()


def _extract_urls(payload: dict) -> set[str]:
    """Cosecha URLs del payload (metadata anidada + texto de embedding)."""
    urls: set[str] = set()

    def _walk(obj, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(obj, str):
            for u in _URL_RE.findall(obj):
                urls.add(u.rstrip("/"))
        elif isinstance(obj, dict):
            for value in obj.values():
                _walk(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                _walk(value, depth + 1)

    _walk(payload)
    return urls


def _extract_names(payload: dict) -> set[str]:
    """Cosecha nombres de producto del payload (metadata name/title + texto)."""
    names: set[str] = set()
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("name", "title", "producto", "product"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                names.add(_normalize(value))
        for key in ("categories", "lineas", "categorias"):
            value = metadata.get(key)
            if isinstance(value, list):
                names.update(
                    _normalize(c) for c in value if isinstance(c, str) and c.strip()
                )
            elif isinstance(value, str) and value.strip():
                names.update(
                    _normalize(c) for c in value.split(",") if c.strip()
                )
        in_text = metadata.get("text") or metadata.get("description")
        if isinstance(in_text, str):
            for m in _PRODUCT_NAME_IN_TEXT_RE.finditer(in_text):
                cand = m.group(1).strip()
                if cand:
                    names.add(_normalize(cand))
    text = payload.get("text") or payload.get("content")
    if isinstance(text, str):
        for m in _PRODUCT_NAME_IN_TEXT_RE.finditer(text):
            cand = m.group(1).strip()
            if cand:
                names.add(_normalize(cand))
    name = payload.get("name") or payload.get("title")
    if isinstance(name, str) and name.strip():
        names.add(_normalize(name))
    return names


def collect_known(context_items: list[dict], history: list[dict]) -> dict[str, set[str]]:
    """Set de URLs y nombres autorizados para el turno.

    Autoriza lo recuperado en este turno (`context_items`) más lo que el
    asistente ya mostró en turnos anteriores (`history`), respetando la regla
    de follow-ups de la conversación.
    """
    urls: set[str] = set()
    names: set[str] = set()

    for item in context_items or []:
        urls |= _extract_urls(item.get("payload", {}) or {})
        names |= _extract_names(item.get("payload", {}) or {})

    for msg in history or []:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        urls |= {u.rstrip("/") for u in _URL_RE.findall(content)}
        for m in _ARROW_NAME_RE.finditer(content):
            cand = m.group(1).strip()
            if cand:
                names.add(_normalize(cand))
        for m in _PRODUCT_NAME_IN_TEXT_RE.finditer(content):
            cand = m.group(1).strip()
            if cand:
                names.add(_normalize(cand))

    return {"urls": urls, "names": names}


def find_ungrounded(text: str, known: dict[str, set[str]]) -> list[str]:
    """Devuelve qué menciones de la respuesta no están respaldadas.

    - Toda URL que no esté en `known["urls"]` es una alucinación.
    - Todo nombre resaltado con "👉" que no coincida (normalizado, por
      subcadena) con ningún nombre conocido es una alucinación.
    """
    issues: list[str] = []
    if not text:
        return issues

    for u in _URL_RE.findall(text):
        if u.rstrip("/") not in known["urls"]:
            issues.append(f"url:{u}")

    for m in _ARROW_NAME_RE.finditer(text):
        cand = _normalize(m.group(1))
        if not cand:
            continue
        if not any(cand in known_name or known_name in cand
                   for known_name in known["names"]):
            issues.append(f"producto:{m.group(1).strip()}")

    return issues


def apply_grounding(
    text: str,
    context_items: list[dict],
    history: list[dict],
    intent: str = "",
) -> tuple[str, list[str]]:
    """Verifica la respuesta y, si inventa algo, devuelve el fallback honesto.

    La validación de nombres solo aplica cuando la intención pide productos
    (CATALOGO) o hay productos en el historial: en respuestas de políticas o
    info general los "👉" no tienen sentido. Las URLs se validan siempre.
    """
    known = collect_known(context_items, history)
    issues = find_ungrounded(text, known)

    if not issues:
        return text, []

    relevant = issues
    if "CATALOGO" not in intent and not known["names"]:
        only_urls = [i for i in issues if i.startswith("url:")]
        if not only_urls:
            return text, []
        relevant = only_urls

    if relevant:
        logger.warning(
            "Grounding: respuesta bloqueada por menciones no respaldadas: %s",
            " | ".join(relevant),
        )
        return FALLBACK_UNGROUNDED, relevant

    return text, []