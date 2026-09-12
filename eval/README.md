# Golden set de eia-rag (`eval/`)

Suite de evaluación **dev-only** de eia-rag. Contiene un conjunto simulado de
casos restringido a lo que **realmente existe** en la colección Qdrant
`COMPLETA` y un runner que mide la tasa de acierto de `POST /chat` contra una
**instancia local**.

> No toca producción: no escribe en Qdrant, no usa Redis en prod y no consume
> tokens de otra cosa que no sea tu instancia local.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `golden_set.jsonl` | Golden set (un caso por línea, JSON). |
| `eval.py` | Runner (stdlib + asyncio, sin dependencias nuevas). |
| `report.json` | Reporte generado por `eval.py` (no se commitear cambios de una corrida). |

## Por qué existe

Fase 1 del plan de agentificación: un baseline medible de la calidad actual de
`/chat` **antes** de construir `/agent/chat`. Cada caso exige respuestas que solo
se pueden dar si el RAG (retrieval + memoria + generación) funciona bien: nombrar
el producto correcto, incluir su link, no inventar precios, aislar tiendas y no
alucinar cuando el inbox no está configurado.

## Productos reales que cubre

Los casos **solo** mencionan productos verificados en `COMPLETA` (15 puntos
`content_type=CATALOGO`). Mapa tienda → productos visibles:

| Tienda | channel_token | Productos |
|--------|---------------|-----------|
| `ecommer` (global) | `[]` (ve todo) | Los 15, incluye KZ Castor Pro, Stewart Cálculo 1, Café orgánico Alem, champiñones Orellana, NARAYANA, etc. |
| `sol-y-luna` | `sol-y-luna-token` | Panela orgánica, Café orgánico, Collar Búho, Amigurumi, Árbol de la vida, Tope de puerta, Mandalas, Separador de libros |
| tokens sin tienda mapeada | `fungogenix-token`, `narayana-token`, `ziru-acoustics-token`, `legaltech-token`, `cos_store-token`, `phybuch-token` | Solo visibles vía ecommer global (no hay inbox configurado aún) |

Documentos no-catálogo en `COMPLETA` (fundamentan los casos de
POLITICAS/INFO_GENERAL): `TemsAndConds.pdf` (reversión de pagos), `Contexto
General.pdf`, `Quienes somos.pdf`, `Misión & Visión.pdf`,
`Preguntas frecuentes _ Vendedores.pdf`.

## Formato del golden set

Un caso por línea en `golden_set.jsonl`:

```json
{
  "id": "gs_001",
  "tienda_id": "ecommer",
  "inbox_id": 2,
  "canal": "whatsapp",
  "turnos": ["¿Tienen audífonos KZ Castor Pro?", "¿y en qué colores vienen?"],
  "debe_incluir": ["kz castor"],
  "no_debe": ["regex:[$€]\\s?[0-9]"],
  "intencion_esperada": "CATALOGO",
  "tags": ["catalogo", "multi_turno", "ecommer"]
}
```

| Campo | Significado |
|-------|-------------|
| `turnos[]` | Se envían secuencialmente al mismo `conversation_id` (`eval-{id}`). Mide memoria (Fase 2). |
| `debe_incluir[]` | Substrings (case-insensitive) que **todos** deben aparecer en alguna respuesta. |
| `no_debe[]` | Substrings que **ninguna** respuesta debe contener. |
| `regex:` | Prefijo para criterios con patrón, p.ej. no inventar precios: `regex:[$€]\s?[0-9]`. |
| `intencion_esperada` | Referencia para el `--check-intent` (informativo si no se activa). |
| `tags` | Agrupación del reporte por área. |

Un caso **pasa** si todo `debe_incluir` acierta y nada de `no_debe` aparece.
La intención detectada solo se verifica con `--check-intent`.

## Cobertura (36 casos)

| Área | Casos | Ejemplo |
|------|-------|---------|
| CATALOGO single-turn | 12 | `gs_001`–`gs_012` |
| No inventar precio | 4 | `gs_013`–`gs_016` |
| Multi-turno (memoria) | 5 | `gs_017`–`gs_021` |
| Aislamiento de tienda | 3 | `gs_022`–`gs_024` |
| Inbox desconocido (fail-closed) | 2 | `gs_025`–`gs_026` |
| POLITICAS / INFO_GENERAL | 4 | `gs_027`–`gs_030` |
| Mixta catálogo + política | 2 | `gs_031`–`gs_032` |
| Escalada a humano | 1 | `gs_033` |
| Conversacional | 3 | `gs_034`–`gs_036` |

## Uso

```bash
# 1) Levantar eia-rag local
uv sync
uv run dev          # arranca en :8000

# 2) Correr la evaluación completa contra la instancia local
uv run python eval/eval.py --endpoint http://localhost:8000/chat

# Filtros útiles
uv run python eval/eval.py --tag catalogo          # solo un área
uv run python eval/eval.py --limit 5               # primeros N casos
uv run python eval/eval.py --check-intent          # valida intención detectada
uv run python eval/eval.py --dry-run               # simula respuestas, sin red
```

El runner genera `report.json` con el detalle de cada caso y el resumen por tag.
Salida del proceso: `0` si todos pasan, `1` si alguno falla (útil en CI local).

## Notas

- Timeout por request: 80 s por defecto (`--timeout`). Los modelos Groq
  `gpt-oss` razonan antes de responder; no bajar el presupuesto.
- `user_id` es derivado del `id` del caso (memoria por conversación, no por
  usuario).
- `eval.py` usa solo la stdlib — nada que instalar.
- Los casos de POLITICAS/INFO usan tokens verificados en `TemsAndConds.pdf`
  (`reversi`, `ecommer sas`) y `Contexto General.pdf` (`plataforma`,
  `comisiones por venta`). Si cambias esos documentos, ajusta los criterios.