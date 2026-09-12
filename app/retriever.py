import logging
import re
import unicodedata

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Filter, FieldCondition, MatchAny, IsEmptyCondition, PayloadField
from openai import AsyncAzureOpenAI
from app.config import settings
from app.store_resolver import StoreConfig

logger = logging.getLogger(__name__)

# Puente léxico para términos amplios de clientes que el embedding denso no
# asocia a las categorías del catálogo (p. ej. "tecnología" → Audio/Electrónica).
# Claves y valores normalizados (sin acentos, minúsculas). Solo se usa como
# segundo intento cuando la búsqueda CATALOGO no devuelve resultados.
_CATALOG_ALIASES: dict[str, list[str]] = {
    "tecnologia": ["electronica", "audio", "tv y video", "dispositivos", "computo"],
    "electronica": ["tecnologia", "audio"],
    "audio": ["electronica", "audifonos"],
    "computo": ["electronica", "tecnologia"],
}

_qdrant: AsyncQdrantClient | None = None
_azure: AsyncAzureOpenAI | None = None


def _get_qdrant() -> AsyncQdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0,
        )
    return _qdrant


def _get_azure() -> AsyncAzureOpenAI:
    global _azure
    if _azure is None:
        _azure = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )
    return _azure


async def _embed(text: str) -> list[float]:
    client = _get_azure()
    response = await client.embeddings.create(
        input=[text],
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        dimensions=1536,
    )
    return response.data[0].embedding


def _normalize_token(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c.lower() for c in text if c.isalnum() or c.isspace()).strip()


def expand_catalog_query(query: str) -> str:
    """Devuelve una consulta ampliada si el texto menciona una categoría amplia.

    Ej.: "quiero algo de tecnología" → "quiero algo de tecnología
    (electronica, audio, tv y video, dispositivos, computo)".
    Retorna la consulta original si no hay ninguna categoría conocida.
    """
    norm = _normalize_token(query)
    for key, aliases in _CATALOG_ALIASES.items():
        if key in norm:
            extra = ", ".join(aliases)
            return f"{query} ({extra})"
    return query


def _apply_min_score(results: list[dict]) -> list[dict]:
    """Descarta resultados por debajo del umbral de similitud.

    Evita que contexto débil (p.ej. un atributo compartido como "Calzado" en un
    producto que no es zapato) alimente al LLM y dispare una respuesta
    inventada. Con `RETRIEVER_MIN_SCORE=0` (o negativo) el filtro queda
    desactivado.
    """
    min_score = settings.RETRIEVER_MIN_SCORE
    if not min_score:
        return results
    kept = [r for r in results if (r.get("score") or 0.0) >= min_score]
    if len(kept) != len(results):
        logger.info(
            "Umbral de score (>=%.3f): %d/%d resultados", min_score, len(kept), len(results)
        )
    return kept


async def search_context(
    query: str,
    store: StoreConfig,
    intents: list[str],
    limit: int = 10,
) -> list[dict]:
    logger.info("Buscando: '%s' (store=%s, intents=%s)", query, store.store_name, intents)

    if "CONVERSACIONAL" in intents and len(intents) == 1:
        logger.info("Intención CONVERSACIONAL — saltando búsqueda en Qdrant")
        return []

    try:
        query_vector = await _embed(query)
    except Exception as e:
        logger.error("Error generando embedding: %s", e)
        return []

    must_conditions: list[FieldCondition] = []

    has_catalogo = "CATALOGO" in intents

    if not (has_catalogo and store.is_global):
        must_conditions.append(
            FieldCondition(
                key="audience",
                match=MatchAny(any=[store.audience]),
            )
        )

    content_type_values = []
    for intent in intents:
        if intent in ("CATALOGO", "POLITICAS", "INFO_GENERAL"):
            content_type_values.append(intent)

    if content_type_values:
        must_conditions.append(
            FieldCondition(
                key="content_type",
                match=MatchAny(any=content_type_values),
            )
        )
    _platform_only = IsEmptyCondition(is_empty=PayloadField(key="metadata.channel_tokens"))

    if has_catalogo and store.is_global:
        pass
    elif has_catalogo and store.channel_tokens:
        must_conditions.append(
            FieldCondition(
                key="metadata.channel_tokens",
                match=MatchAny(any=store.channel_tokens),
            )
        )
    elif store.is_global:
        must_conditions.append(_platform_only)
    elif store.channel_tokens:
        must_conditions.append(
            Filter(should=[
                FieldCondition(
                    key="metadata.channel_tokens",
                    match=MatchAny(any=store.channel_tokens),
                ),
                _platform_only,
            ])
        )

    search_filter = Filter(must=must_conditions) if must_conditions else None
    client = _get_qdrant()

    async def _query_once(vector: list[float]) -> list[dict]:
        try:
            response = await client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
            )
            points = response.points or []
            logger.info("Encontrados %d resultados", len(points))
            return _apply_min_score(
                [{"score": hit.score, "payload": hit.payload} for hit in points]
            )
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.warning("Colección '%s' no existe", settings.COLLECTION_NAME)
                return []
            if e.status_code == 400:
                logger.warning("Error 400 de Qdrant — reintentando sin filtro de audiencia")
                try:
                    fallback_conditions = [
                        c for c in must_conditions
                        if not (isinstance(c, FieldCondition) and c.key == "audience")
                    ]
                    fallback_filter = Filter(must=fallback_conditions) if fallback_conditions else None
                    response = await client.query_points(
                        collection_name=settings.COLLECTION_NAME,
                        query=vector,
                        query_filter=fallback_filter,
                        limit=limit,
                        with_payload=True,
                    )
                    points = response.points or []
                    logger.info("Encontrados %d resultados", len(points))
                    return _apply_min_score(
                        [{"score": hit.score, "payload": hit.payload} for hit in points]
                    )
                except Exception as f:
                    logger.error("Fallo también el fallback: %s", f)
                    return []
            logger.error("Error Qdrant HTTP: %s", e)
            return []
        except Exception as e:
            logger.error("Error en retriever: %s", e)
            return []

    results = await _query_once(query_vector)

    if not results and has_catalogo:
        expanded = expand_catalog_query(query)
        if expanded != query:
            logger.info(
                "Sin resultados CATALOGO — reintentando con consulta ampliada: '%s'",
                expanded,
            )
            try:
                results = await _query_once(await _embed(expanded))
            except Exception as e:
                logger.error("Error ampliando consulta: %s", e)

    return results


async def list_categories(store: StoreConfig, limit: int = 1000) -> list[str]:
    """Categorías reales del catálogo de la tienda (fuente estructurada).

    Recorre el payload indexado en Qdrant (misma colección/filtros que la
    búsqueda del catálogo) y devuelve las categorías distintas, ordenadas.
    Se usa para consultas de agregación ("¿qué categorías tienen?") que no
    deben pasar por búsqueda semántica.
    """
    must_conditions = [
        FieldCondition(
            key="content_type",
            match=MatchAny(any=["CATALOGO"]),
        )
    ]
    if not store.is_global:
        must_conditions.append(
            FieldCondition(
                key="audience",
                match=MatchAny(any=[store.audience]),
            )
        )
        if store.channel_tokens:
            must_conditions.append(
                FieldCondition(
                    key="metadata.channel_tokens",
                    match=MatchAny(any=store.channel_tokens),
                )
            )

    categories: set[str] = set()
    offset = None
    fetched = 0
    page_size = 100
    try:
        client = _get_qdrant()
        while fetched < limit:
            points, offset = await client.scroll(
                collection_name=settings.COLLECTION_NAME,
                scroll_filter=Filter(must=must_conditions),
                limit=min(page_size, limit - fetched),
                offset=offset,
                with_payload=True,
            )
            for point in points or []:
                payload = point.payload or {}
                feed = [payload]
                meta = payload.get("metadata")
                if isinstance(meta, dict):
                    feed.append(meta)
                for obj in feed:
                    cats = obj.get("categories")
                    if isinstance(cats, list):
                        categories.update(
                            c.strip() for c in cats if isinstance(c, str) and c.strip()
                        )
                    elif isinstance(cats, str) and cats.strip():
                        categories.update(
                            c.strip() for c in cats.split(",") if c.strip()
                        )
                text = payload.get("text") or ""
                if isinstance(text, str):
                    for m in re.finditer(r"Categor[íi]as:\s*([^.\n]+)", text):
                        categories.update(
                            c.strip() for c in m.group(1).split(",") if c.strip()
                        )
            fetched += len(points or [])
            if not points or offset is None:
                break
    except Exception as e:
        logger.error("Error obteniendo categorías: %s", e)
        return []

    return sorted(categories)
