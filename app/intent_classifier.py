import re
import logging
import unicodedata
from langfuse import observe
from app.clients import get_llm, pick_model
from app.config import settings

logger = logging.getLogger(__name__)


ROUTER_SYSTEM_PROMPT = """Eres el enrutador de intenciones de la plataforma Ecommer. Clasifica la entrada del usuario en una o varias de estas categorias:

1. "CATALOGO": Busca comprar, pregunta por productos, caracteristicas o disponibilidad de stock.
2. "POLITICAS": Pregunta por envios, devoluciones, garantias o reglas especificas de una compra.
3. "INFO_GENERAL": Pregunta sobre que es Ecommer, como funciona, costos de suscripcion, pasarelas de pago, facturacion, soporte tecnico, mision o vision de la empresa.
4. "CONVERSACIONAL": Saludos, agradecimientos o preguntas totalmente fuera de contexto.

Si la consulta cubre mas de una categoria, responde con las categorias separadas por comas.
Responde UNICAMENTE con las palabras exactas de las categorias, en mayusculas."""


def _parse_intent_output(raw: str) -> list[str]:
    normalized = re.sub(r"[\n;]+", ",", raw)
    tokens = [t.strip() for t in normalized.split(",") if t.strip()]
    valid = {"CATALOGO", "POLITICAS", "INFO_GENERAL", "CONVERSACIONAL"}
    return list(dict.fromkeys(t for t in tokens if t in valid))


def _normalize(text: str) -> str:
    """Normaliza texto para comparar: minúsculas y sin tildes."""
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


def _merge_llm_and_keywords(llm_intents: list[str], keyword_intents: list[str]) -> list[str]:
    """Fusión determinista: keywords anclan (nunca se pierden) y el LLM agrega
    lo que los regex no cubren. CONVERSACIONAL solo sobrevive si es único."""
    combined = list(dict.fromkeys(keyword_intents + llm_intents))
    if not combined:
        return ["CONVERSACIONAL"]
    if "CONVERSACIONAL" in combined and len(combined) > 1:
        combined = [i for i in combined if i != "CONVERSACIONAL"]
    return combined


async def _llm_classify(query: str, vk: str, model: str) -> list[str]:
    """Llamada LLM al router con `model` ya elegido según el plan.

    El prompt va COMPLETO en el rol `user` (instrucción + query): el despliegue
    de Bifrost para Phi-4-mini descarta el system prompt, y solo con este
    formato clasifica de forma fiable. `temperature=0` y `max_tokens` mínimo
    para que responda solo categorías, nunca prosa conversacional.
    """
    prompt = (
        f"{ROUTER_SYSTEM_PROMPT}\n\n"
        f"Mensaje del cliente a clasificar:\n{query}"
    )
    client = get_llm(vk)
    completion = await client.with_options(
        timeout=settings.BIFROST_ROUTER_TIMEOUT
    ).chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=32,
    )
    raw = (completion.choices[0].message.content or "").strip().upper()
    logger.info("Router LLM (%s): %r -> %r", model, query[:60], raw)
    return _parse_intent_output(raw)


@observe(as_type="generation", name="classify_intent")
async def classify_intent(query: str, vk: str | None = None) -> list[str]:
    """Clasifica la intención: keywords primero, Phi-4 solo si hace falta.

    1. Regex keywords (~0ms) — si detectan algo, no se llama al LLM.
    2. Sin VK → keywords o CONVERSACIONAL (sin LLM).
    3. Ambiguo + plan con Phi-4 → LLM router (prompt en rol user, temp 0).
    4. Ambiguo + plan sin Phi-4 (solo mini) → CONVERSACIONAL: el mini en
       Bifrost no clasifica de forma fiable, y los mensajes ambiguos de
       planes free los cubren los keywords cuando son relevantes.

    El LLM se reserva para mensajes que los regex no cubren: los típicos
    ("tienen X", "¿cómo devuelvo?") ya los resuelven los keywords al instante.
    """
    logger.info("Clasificando intención: '%s'", query)
    keyword_intents = _keyword_fallback(query)
    if keyword_intents:
        logger.info("Intención por keywords: %s", keyword_intents)
        return keyword_intents
    if not vk:
        logger.warning("Sin VK Bifrost: clasificación solo por keywords")
        return ["CONVERSACIONAL"]
    try:
        model = await pick_model(
            vk, [settings.BIFROST_CHAT_MODEL, settings.BIFROST_ROUTER_MODEL]
        )
        if model != settings.BIFROST_CHAT_MODEL:
            logger.info(
                "Plan sin Phi-4 (modelo=%s): sin keywords, respuesta CONVERSACIONAL", model
            )
            return ["CONVERSACIONAL"]
        llm_intents = await _llm_classify(query, vk, model)
    except Exception as e:
        logger.error("Error clasificando intención: %s", e)
        return ["CONVERSACIONAL"]
    return _merge_llm_and_keywords(llm_intents, keyword_intents)
