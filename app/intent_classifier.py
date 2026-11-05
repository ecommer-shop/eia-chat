import logging
import re
import unicodedata
from dataclasses import dataclass, field

from langfuse import observe

from app.clients import get_llm, pick_model
from app.config import settings

logger = logging.getLogger(__name__)

# Intenciones válidas del enrutador. `BUSCAR` es la intención explícita de
# búsqueda de catálogo con query estructurada: se extrae la query para
# pasarla tal cual al endpoint de catálogo (a diferencia de `CATALOGO`, que
# es búsqueda semántica sobre el embedding). Mantener ambas para no cambiar
# el contrato del pipeline RAG existente.
#
# `CARRITO` y `PEDIDO` (Fase 1b) son consultas de identidad: el carrito activo
# y un pedido concreto del comprador de esa conversación de Chatwoot. No son
# búsquedas de catálogo: `PEDIDO` implica un carrito ya creado.
VALID_INTENTS = {
    "CATALOGO",
    "BUSCAR",
    "POLITICAS",
    "INFO_GENERAL",
    "CONVERSACIONAL",
    "CARRITO",
    "PEDIDO",
}

ROUTER_SYSTEM_PROMPT = """Eres el enrutador de intenciones de la plataforma Ecommer. Clasifica la entrada del usuario en una o varias de estas categorías:

1. "BUSCAR": El usuario pide explícitamente buscar, encontrar o listar productos del catálogo (p. ej. "busca", "quién vende", "qué productos hay de ...", "encuéntrame ...").
2. "CATALOGO": El usuario quiere comprar o pregunta por productos, características, disponibilidad o precios.
3. "POLITICAS": Pregunta por envíos, devoluciones, garantías o reglas específicas de una compra.
4. "INFO_GENERAL": Pregunta sobre qué es Ecommer, cómo funciona, costos de suscripción, pasarelas de pago, facturación, soporte técnico, misión o visión de la empresa.
5. "CARRITO": El usuario pregunta por su carrito actual, lo que ya agregó o el total a pagar de su compra en curso (p. ej. "¿qué tengo en mi carrito?", "ver carrito", "cuánto llevo").
6. "PEDIDO": El usuario pregunta por el estado, la entrega o el detalle de un pedido YA realizado (p. ej. "dónde está mi pedido", "estado de mi pedido", "mi pedido 1234"). NO uses PEDIDO cuando el usuario quiere comprar o hacer un pedido nuevo: eso es CATALOGO.
7. "CONVERSACIONAL": Saludos, agradecimientos o preguntas totalmente fuera de contexto.

Si la consulta cubre más de una categoría, responde con las categorías separadas por comas.
Si la categoría es BUSCAR o CATALOGO, agrega además, en una línea aparte con el prefijo "QUERY:", la consulta de búsqueda extraída (máx. 10 palabras) que usarías para buscar en el catálogo.
Si la categoría es PEDIDO y el usuario menciona el código o número de su pedido, agrega además, en una línea aparte con el prefijo "CODE:", ese código exactamente como lo escribió.
Responde ÚNICAMENTE con las categorías en mayúsculas y, opcionalmente, las líneas QUERY y CODE."""


@dataclass
class IntentClassification:
    """Resultado detallado de la clasificación de intención.

    `confidence` es una estimación en [0, 1] de cuán seguro está el
    clasificador. Por debajo de `settings.BIFROST_INTENT_CONFIDENCE_THRESHOLD`
    el caller decide degradar al RAG actual (keywords).
    """

    intents: list[str]
    params: dict = field(default_factory=dict)
    confidence: float = 1.0


def _parse_intent_output(raw: str) -> list[str]:
    # Separa la línea QUERY (si el modelo la emitió) del cuerpo de intenciones.
    body = raw
    if "QUERY:" in body:
        body = body.split("QUERY:", 1)[0]
    normalized = re.sub(r"[\n;]+", ",", body)
    tokens = [t.strip() for t in normalized.split(",") if t.strip()]
    return list(dict.fromkeys(t for t in tokens if t in VALID_INTENTS))


def _parse_query_param(raw: str) -> str | None:
    """Extrae la query de búsqueda de la línea `QUERY:` si el modelo la emitió."""
    m = re.search(r"QUERY:\s*(.+)", raw, re.IGNORECASE)
    if not m:
        return None
    query = m.group(1).strip().strip('"').strip("'")
    query = re.sub(r"[\n;]+", " ", query).strip()
    return query or None


def _parse_code_param(raw: str) -> str | None:
    """Extrae el código de pedido de la línea `CODE:` si el modelo la emitió."""
    m = re.search(r"CODE:\s*(.+)", raw, re.IGNORECASE)
    if not m:
        return None
    code = m.group(1).strip().strip('"').strip("'").strip(".").strip()
    code = re.sub(r"[\n;,]+", " ", code).strip()
    return code or None


def _normalize(text: str) -> str:
    """Normaliza texto para comparación: minúsculas y sin tildes."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


_BUSCAR_RE = re.compile(
    r"\b(?:"
    r"busca|buscas|buscame|buscamos|buscar|encuentra|encuentrame|"
    r"busco|busquemos|buscad|"
    r"dime que|dime quienes|que venden|quien vende|quienes venden|"
    r"que productos hay|que articulos hay|listame|listar|muestrame|muestra"
    r")\b"
)

_CATALOGO_RE = re.compile(
    r"\b(?:"
    r"compr(?:a|as|o|amos|an|ar|e)|"
    r"quiero|quieres|quiere|quisiera|necesito|necesitamos|necesitas|"
    r"vende|venden|vendes|vendo|vendemos|"
    r"tiene|tienen|hay|existe|existen|disponible|disponibles|"
    r"cuesta|cuestan|cuanto|cuanta|vale|valor|precio|precios|costo|stock|"
    r"producto|productos|articulo|articulos|"
    r"talla|tallas|modelo|modelos|marca|marcas|color|colores|"
    r"caracteristica|caracteristicas|catalogo|catalogos|categoria|categorias"
    r")\b"
)

_POLITICAS_RE = re.compile(
    r"\b(?:"
    r"devolucion|devoluciones|devolver|reversion|revertir|"
    r"garantia|garantias|"
    r"envio|envios|envian|enviamos|envias|despacho|despachos|"
    r"politica|politicas|"
    r"cambio|cambios|cambiar|cambias|"
    r"reembolso|reembolsos|"
    r"retracto|retractarse|"
    r"domicilio|domicilios|"
    r"cancelar|cancela"
    r")\b"
)

# --- Fase 1b: consultas de identidad (carrito / pedido) -------------------
# Precisión sobre recall, a propósito: un falso PEDIDO manda la consulta a una
# tool de identidad que no tiene nada que ver con un pedido real. Los verbos
# de compra ("quiero hacer un pedido", "comprar") NO deben caer acá; el guard
# `_NEW_ORDER_RE` los excluye explícitamente porque `_CATALOGO_RE` los captura
# primero (CATALOGO).
_CARRITO_RE = re.compile(
    r"\b(?:mi|el|tu|del) carrito\b"
    r"|\bcarrito (?:de compras|actual|activo)\b"
    r"|\b(?:ver|revisar|abrir|mostrar|mostrame|mostrarme|consultar) (?:mi |el )?carrito\b"
    r"|\bque (?:tengo|hay|tiene|tenes|tienes) en (?:mi |el )?carrito\b"
    r"|\b(?:agregue|agregue al|tengo agregado|tengo en el) carrito\b"
)

# Un "pedido nuevo" es una compra: lo atiende el catálogo, no la tool de pedidos.
_NEW_ORDER_RE = re.compile(
    r"\b(?:hacer|hago|hice|realizar|realizo|hacerle|nuevo|nueva|otro|otra) (?:un |una |el |la )?pedido\b"
    r"|\b(?:quiero|queria|quisiera|necesito) (?:hacer |realizar )?(?:un |una |el |la )?pedido\b"
)

_PEDIDO_RE = re.compile(
    r"\bmi pedido\b"
    r"|\bpedido (?:numero|nro|n\u00b0|#)\b"
    r"|\b(?:estado|seguimiento|detalle|informacion|info) (?:de |del )?(?:mi |el )?pedido\b"
    r"|\b(?:donde|cuando) (?:esta|llega|sale|arriba|me llega) (?:mi |el )?pedido\b"
    r"|\b(?:rastrear|seguir|consultar|ver) (?:mi |el )?pedido\b"
    r"|\bnumero de pedido\b"
    r"|\b(?:ya|casi) (?:llego|llega|enviaste|salio) (?:mi |el )?pedido\b"
)

# Código de pedido tal como lo emite Vendure: alfanumérico, con guiones o con
# prefijo `#`. Se exige al menos un dígito para no capturar palabras sueltas.
_ORDER_CODE_RE = re.compile(
    r"(?:\bpedido\b|\borden\b|\bcode\b|\bcodigo\b)[^\w\n]{0,10}?(#?[A-Za-z0-9][A-Za-z0-9\-_]{2,63})\b"
    r"|#([A-Za-z0-9][A-Za-z0-9\-_]{2,63})\b",
    re.IGNORECASE,
)

_INFO_GENERAL_RE = re.compile(
    r"\b(?:"
    r"ecommer|empresa|"
    r"suscripcion|suscripciones|membresia|membresias|"
    r"costo|costos|cobro|cobros|"
    r"wompi|dian|factura|facturacion|facturar|"
    r"pago|pagos|pagar|pasarela|pasarelas|transaccion|transacciones|"
    r"mision|vision|"
    r"soporte|"
    r"horario|horarios|ubicacion|ubicaciones|direccion|contacto|contactar|contactanos|"
    r"quienes|funciona|funcionan|como funciona"
    r")\b"
)

_CONVERSACIONAL_RE = re.compile(
    r"\b(?:"
    r"hola|hey|saludos|"
    r"buenos dias|buen dia|buenas tardes|buenas noches|buenas|"
    r"gracias|agradezco|"
    r"que tal|como estas|como te va|que mas|"
    r"todo bien|bien y tu|genial|excelente|perfecto|listo"
    r")\b"
)


def _extract_order_code(query: str) -> str | None:
    """Best-effort: código de pedido mencionado en el texto crudo.

    Se aplica sobre el texto original (no el normalizado) para no perder
    mayúsculas ni guiones, que forman parte del código.
    """
    for match in _ORDER_CODE_RE.finditer(query or ""):
        candidate = next((g for g in match.groups() if g), None)
        if not candidate:
            continue
        code = candidate.strip().lstrip("#")
        if not any(ch.isdigit() for ch in code):
            continue
        return code
    return None


def _keyword_fallback(query: str) -> list[str]:
    q = _normalize(query)
    detected: list[str] = []

    if _BUSCAR_RE.search(q):
        detected.append("BUSCAR")

    if _CATALOGO_RE.search(q):
        detected.append("CATALOGO")

    if _CARRITO_RE.search(q):
        detected.append("CARRITO")

    # `PEDIDO` apunta a un pedido ya realizado; un pedido nuevo es una compra
    # (CATALOGO) y lo resuelve el catálogo.
    if _PEDIDO_RE.search(q) and not _NEW_ORDER_RE.search(q):
        detected.append("PEDIDO")

    if _POLITICAS_RE.search(q):
        detected.append("POLITICAS")

    if _INFO_GENERAL_RE.search(q):
        detected.append("INFO_GENERAL")

    if not detected and _CONVERSACIONAL_RE.search(q):
        detected.append("CONVERSACIONAL")

    return list(dict.fromkeys(detected))


@observe(as_type="generation", name="classify_intent")
async def classify_intent_detailed(
    query: str, vk: str | None = None
) -> IntentClassification:
    """Clasifica la intención y devuelve intención + parámetros + confianza.

    El modelo siempre es el más pequeño que el plan permite (Phi-4-mini por
    defecto). Sin VK no hay LLM: se usa solo el fallback por keywords con
    confianza baja.
    """
    logger.info("Clasificando intención: '%s'", query)

    if not vk:
        logger.warning("Sin VK Bifrost: clasificación solo por keywords")
        intents = _keyword_fallback(query) or ["CONVERSACIONAL"]
        params: dict = {}
        if "PEDIDO" in intents:
            code = _extract_order_code(query)
            if code:
                params["code"] = code
        return IntentClassification(intents=intents, params=params, confidence=0.3)

    try:
        model = await pick_model(
            vk,
            [settings.BIFROST_ROUTER_MODEL, settings.BIFROST_CHAT_MODEL_FALLBACK],
        )
        client = get_llm(vk)
        completion = await client.with_options(
            timeout=settings.BIFROST_ROUTER_TIMEOUT
        ).chat.completions.create(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            model=model,
            temperature=0.0,
            max_tokens=256,
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw_upper = raw.upper()
        detected = _parse_intent_output(raw_upper)
        fallback = _keyword_fallback(query)

        combined = list(dict.fromkeys(detected + fallback))
        if not combined:
            return IntentClassification(intents=["CONVERSACIONAL"], confidence=0.3)

        if "CONVERSACIONAL" in combined and len(combined) > 1:
            combined = [i for i in combined if i != "CONVERSACIONAL"]

        # Confianza: si el LLM no detectó nada distinto del fallback por
        # keywords, se considera baja. Si el LLM dio una señal clara, alta.
        llm_only = [i for i in detected if i not in fallback]
        confidence = 0.9 if llm_only or detected else 0.5

        params: dict = {}
        if "BUSCAR" in combined or "CATALOGO" in combined:
            # La query se extrae del raw original (preserva mayúsculas/acentos).
            extracted = _parse_query_param(raw)
            params["query"] = (extracted or query).strip()

        if "PEDIDO" in combined:
            # Sin código la tool no puede consultar el pedido: se pide el código.
            code = _parse_code_param(raw) or _extract_order_code(query)
            if code:
                params["code"] = code

        return IntentClassification(
            intents=combined,
            params=params,
            confidence=confidence,
        )
    except Exception as e:
        logger.error("Error clasificando intención: %s", e)
        intents = _keyword_fallback(query) or ["CONVERSACIONAL"]
        params = {}
        if "PEDIDO" in intents:
            code = _extract_order_code(query)
            if code:
                params["code"] = code
        return IntentClassification(intents=intents, params=params, confidence=0.3)


async def classify_intent(query: str, vk: str | None = None) -> list[str]:
    """Compatibilidad hacia atrás: devuelve solo la lista de intenciones.

    El resto del pipeline (`app/agent/core.py`, tests) consume este contrato.
    Quien necesite parámetros o confianza usa `classify_intent_detailed`.
    """
    result = await classify_intent_detailed(query, vk=vk)
    return result.intents
