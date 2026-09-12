import re
import logging
import unicodedata
from groq import AsyncGroq
from langfuse import observe
from app.config import settings

logger = logging.getLogger(__name__)

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    return _client


ROUTER_SYSTEM_PROMPT = """Eres el enrutador de intenciones de la plataforma Ecommer. Clasifica la entrada del usuario en una o varias de estas categorías:

1. "CATALOGO": Busca comprar, pregunta por productos, características o disponibilidad de stock.
2. "POLITICAS": Pregunta por envíos, devoluciones, garantías o reglas específicas de una compra.
3. "INFO_GENERAL": Pregunta sobre qué es Ecommer, cómo funciona, costos de suscripción, pasarelas de pago, facturación, soporte técnico, misión o visión de la empresa.
4. "CONVERSACIONAL": Saludos, agradecimientos o preguntas totalmente fuera de contexto.

Si la consulta cubre más de una categoría, responde con las categorías separadas por comas.
Responde ÚNICAMENTE con las palabras exactas de las categorías, en mayúsculas."""


def _parse_intent_output(raw: str) -> list[str]:
    normalized = re.sub(r"[\n;]+", ",", raw)
    tokens = [t.strip() for t in normalized.split(",") if t.strip()]
    valid = {"CATALOGO", "POLITICAS", "INFO_GENERAL", "CONVERSACIONAL"}
    return list(dict.fromkeys(t for t in tokens if t in valid))


def _normalize(text: str) -> str:
    """Normaliza texto para comparación: minúsculas y sin tildes."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


_CATALOGO_RE = re.compile(
    r"\b(?:"
    r"compr(?:a|as|o|amos|an|ar|e)|"
    r"quiero|quieres|quiere|quisiera|necesito|necesitamos|necesitas|"
    r"busco|busca|buscas|buscamos|buscar|"
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
async def classify_intent(query: str) -> list[str]:
    logger.info("Clasificando intención: '%s'", query)
    try:
        client = _get_client()
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            model=settings.GROQ_ROUTER_MODEL,
            temperature=0.0,
            max_tokens=256,
        )
        raw = completion.choices[0].message.content.strip().upper()
        detected = _parse_intent_output(raw)
        fallback = _keyword_fallback(query)

        combined = list(dict.fromkeys(detected + fallback))
        if not combined:
            return ["CONVERSACIONAL"]

        if "CONVERSACIONAL" in combined and len(combined) > 1:
            combined = [i for i in combined if i != "CONVERSACIONAL"]

        return combined

    except Exception as e:
        logger.error("Error clasificando intención: %s", e)
        return _keyword_fallback(query) or ["CONVERSACIONAL"]
