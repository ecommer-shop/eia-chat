"""
Cliente PostgreSQL para gestión de conversaciones multi-canal.
Maneja contacts, conversations y messages.
"""
import logging
import asyncpg
from uuid import UUID
from typing import Optional
from app.config import settings

logger = logging.getLogger(__name__)

# Pool de conexiones global
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Retorna el pool de conexiones, creándolo si no existe."""
    global _pool
    if _pool is None:
        logger.info("🗄️ Creando pool de conexiones PostgreSQL...")
        try:
            _pool = await asyncpg.create_pool(
                host=settings.PG_HOST,
                port=settings.PG_PORT,
                database=settings.PG_DB,
                user=settings.PG_USER,
                password=settings.PG_PASSWORD,
                min_size=2,
                max_size=10,
            )
            logger.info("✅ Pool PostgreSQL listo.")
        except Exception as e:
            logger.error(f"❌ Error creando pool PostgreSQL: {e}")
            raise
    return _pool


async def close_pool():
    """Cierra el pool al apagar la app."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("🔌 Pool PostgreSQL cerrado.")


# ───────────────────────────────────────────── 
# CONTACTS
# ─────────────────────────────────────────────


async def upsert_contact(
    tenant_id: str,
    channel: str,
    external_id: str,
    name: Optional[str] = None,
) -> str:
    """
    Crea o recupera un contacto por (tenant_id, channel, external_id).
    Retorna el UUID del contacto en string.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO contacts (tenant_id, channel, external_id, name)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (tenant_id, channel, external_id)
                DO UPDATE SET name = COALESCE(EXCLUDED.name, contacts.name),
                              updated_at = NOW()
                RETURNING id
                """,
                tenant_id,
                channel,
                external_id,
                name,
            )
            contact_id = str(row["id"])
            logger.info(f"✅ Contacto upserted: {contact_id} [{channel}:{external_id}]")
            return contact_id
        except Exception as e:
            logger.error(f"❌ Error en upsert_contact: {e}")
            raise


# ───────────────────────────────────────────── 
# CONVERSATIONS
# ─────────────────────────────────────────────


async def get_or_create_conversation(
    tenant_id: str,
    contact_id: str,
    channel: str,
) -> str:
    """
    Busca una conversación activa para el contacto o crea una nueva.
    Retorna el UUID de la conversación en string.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            # Buscar conversación activa existente
            row = await conn.fetchrow(
                """
                SELECT id FROM conversations
                WHERE contact_id = $1 AND status = 'active'
                ORDER BY last_activity DESC
                LIMIT 1
                """,
                contact_id,
            )

            if row:
                conv_id = str(row["id"])
                # Actualizar timestamp de actividad
                await conn.execute(
                    "UPDATE conversations SET last_activity = NOW() WHERE id = $1",
                    conv_id,
                )
                logger.info(f"🔄 Conversación activa encontrada: {conv_id}")
                return conv_id

            # Crear nueva conversación
            row = await conn.fetchrow(
                """
                INSERT INTO conversations (tenant_id, contact_id, channel, status)
                VALUES ($1, $2, $3, 'active')
                RETURNING id
                """,
                tenant_id,
                contact_id,
                channel,
            )
            conv_id = str(row["id"])
            logger.info(f"✅ Nueva conversación creada: {conv_id} [{channel}]")
            return conv_id
        except Exception as e:
            logger.error(f"❌ Error en get_or_create_conversation: {e}")
            raise


# ───────────────────────────────────────────── 
# MESSAGES
# ─────────────────────────────────────────────


async def get_recent_messages(
    conversation_id: str,
    limit: int = 5,
) -> list[dict]:
    """
    Recupera los últimos N mensajes de una conversación.
    Retorna lista de dicts con 'role' y 'content', en orden cronológico.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT role, content FROM (
                    SELECT role, content, created_at
                    FROM messages
                    WHERE conversation_id = $1
                      AND role IN ('user', 'assistant')
                    ORDER BY created_at DESC
                    LIMIT $2
                ) sub
                ORDER BY created_at ASC
                """,
                conversation_id,
                limit,
            )
            messages = [{"role": r["role"], "content": r["content"]} for r in rows]
            logger.info(f"🧠 {len(messages)} mensajes de historial cargados.")
            return messages
        except Exception as e:
            logger.error(f"❌ Error en get_recent_messages: {e}")
            return []


async def save_message(
    conversation_id: str,
    role: str,
    content: str,
    tokens: int = 0,
) -> None:
    """Guarda un mensaje en la base de datos."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, tokens)
                VALUES ($1, $2, $3, $4)
                """,
                conversation_id,
                role,
                content,
                tokens,
            )
            logger.info(f"💾 Mensaje [{role}] guardado en conversación {conversation_id}")
        except Exception as e:
            logger.error(f"❌ Error en save_message: {e}")
            raise
