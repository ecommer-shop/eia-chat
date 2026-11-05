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
VALID_INTENTS = {"CATALOGO", "BUSCAR", "POLITICAS", "INFO_GENERAL", "CONVERSACIONAL"}

ROUTER_SYSTEM_PROMPT = """Eres el enrutador de intenciones de la plataforma Ecommer. Clasifica la entrada del usuario en una o varias de estas categorías:

1. "BUSCAR": El usuario pide explícitamente buscar, encontrar o listar productos del catálogo (p. ej. "busca", "quién vende", "qué productos hay de ...", "encuéntrame ...").
2. "CATALOGO": El usuario quiere comprar o pregunta por productos, características, disponibilidad o precios.
3. "POLITICAS": Pregunta por envíos, devoluciones, garantías o reglas específicas de una compra.
4. "INFO_GENERAL": Pregunta sobre qué es Ecommer, cómo funciona, costos de suscripción, pasarelas de pago, facturación, soporte técnico, misión o visión de la empresa.
5. "CONVERSACIONAL": Saludos, agradecimientos o preguntas totalmente fuera de contexto.

Si la consulta cubre más de una categoría, responde con las categorías separadas por comas.
Si la categoría es BUSCAR o CATALOGO, agrega además, en una línea aparte con el prefijo "QUERY:", la consulta de búsqueda extraída (máx. 10 palabras) que usarías para buscar en el catálogo.
Responde ÚNICAMENTE con las categorías en mayúsculas y, opcionalmente, la línea QUERY."""


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


def _keyword_fallback(query: str) -> list[str]:
    q = _normalize(query)
    detected: list[str] = []

    if _BUSCAR_RE.search(q):
        detected.append("BUSCAR")

    if _CATALOGO_RE.search(q):
        detected.append("CATALOGO")

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
        return IntentClassification(intents=intents, confidence=0.3)

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

        return IntentClassification(
            intents=combined,
            params=params,
            confidence=confidence,
        )
    except Exception as e:
        logger.error("Error clasificando intención: %s", e)
        intents = _keyword_fallback(query) or ["CONVERSACIONAL"]
        return IntentClassification(intents=intents, confidence=0.3)


async def classify_intent(query: str, vk: str | None = None) -> list[str]:
    """Compatibilidad hacia atrás: devuelve solo la lista de intenciones.

    El resto del pipeline (`app/agent/core.py`, tests) consume este contrato.
    Quien necesite parámetros o confianza usa `classify_intent_detailed`.
    """
    result = await classify_intent_detailed(query, vk=vk)
    return result.intents
