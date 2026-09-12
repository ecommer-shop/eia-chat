# Analisis Detallado de Arquitectura — EIA-RAG Gateway

---

## Tabla de Contenidos

1. [Vision General](#1-vision-general)
2. [Stack Tecnologico](#2-stack-tecnologico)
3. [Arquitectura por Capas](#3-arquitectura-por-capas)
4. [Flujo de Datos — End to End](#4-flujo-de-datos--end-to-end)
5. [Analisis de Cada Componente](#5-analisis-de-cada-componente)
6. [Por que se tomo esta Arquitectura](#6-por-que-se-tomo-esta-arquitectura)
7. [Fallas y Problemas Detectados](#7-fallas-y-problemas-detectados)
8. [Seguridad](#8-seguridad)
9. [Observabilidad](#9-observabilidad)
10. [Escalabilidad](#10-escalabilidad)
11. [Gestion del Historial Conversacional](#11-gestion-del-historial-conversacional)
12. [Recomendaciones](#12-recomendaciones)

---

## 1. Vision General

**EIA-RAG** es un microservicio Python que implementa un gateway de **RAG (Retrieval Augmented Generation)** multi-tenant para la plataforma e-commerce **Ecommer**. Sirve como capa intermedia entre **Chatwoot** (plataforma de mensajeria omnicanal) y multiples servicios de IA, respondiendo preguntas de clientes sobre productos, politicas e informacion general de la tienda.

### Arquitectura de Alto Nivel

```
┌──────────────────────────────────────────────────────────────────┐
│                     CANALES DE ENTRADA                           │
│         WhatsApp  │  Instagram  │  Messenger  │  Web            │
└─────────────┬────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CHATWOOT (Omnichannel)                         │
│              Gestiona conversaciones y canales                   │
└─────────────┬────────────────────────────────────────────────────┘
              │  POST /chat {query, conversation_id, inbox_id}
              ▼
┌──────────────────────────────────────────────────────────────────┐
│                   EIA-RAG GATEWAY (FastAPI)                      │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Resolvedor│→ │ Clasificador │→ │Retriever  │→ │Generador  │  │
│  │  Multi-   │  │  de Intenc.  │  │ Vectorial │  │   LLM     │  │
│  │  Tenant   │  │  (Groq 8B)   │  │ (Qdrant)  │  │ (Groq70B) │  │
│  └──────────┘  └──────────────┘  └───────────┘  └───────────┘  │
│                                        │                         │
│                                   ┌────┴────┐                   │
│                                   │  Redis   │                   │
│                                   │ Historial│                   │
│                                   └─────────┘                   │
└─────────────┬────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│               RESPUESTA AL CLIENTE VIA CHATWOOT                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Stack Tecnologico

| Capa | Tecnologia | Version | Justificacion |
|------|-----------|---------|---------------|
| **Lenguaje** | Python | >=3.12 | Ecosistema ML/AI, simplicidad |
| **Framework API** | FastAPI | >=0.115.0 | Async nativo, validacion automatica, docs |
| **Servidor ASGI** | Uvicorn | >=0.30.0 | Alto rendimiento, event loop asyncio |
| **Validacion** | Pydantic v2 | >=2.10.0 | Serializacion, validacion de schemas |
| **Clasificacion** | Groq API | Llama 3.1 8B Instant | Rapido, bajo costo, clasificacion simple |
| **Generacion** | Groq API | Llama 3.3 70B Versatile | Alta calidad, contexto amplio |
| **Embeddings** | Azure OpenAI | text-embedding-3-small | 1536 dimensiones, multilingue |
| **BD Vectorial** | Qdrant Cloud | Managed | Busqueda semantica, filtros por payload |
| **Memoria** | Redis | Cloud-hosted | Lista FIFO, TTL, bajo latencia |
| **Empaquetado** | uv + Hatchling | Astral | Instalacion rapida de dependencias |
| **Contenedor** | Docker | python:3.12-slim | imagen minima, reproduciBLE |
| **Deploy** | Railway | Dockerfile | Deploy automatico, health checks |

---

## 3. Arquitectura por Capas

### 3.1 Capa de Configuracion (`config.py`)

Clase `Settings` que carga variables de entorno usando `pydantic-settings`. Almacena:

- `GROQ_API_KEY` / `GROQ_MODEL_INTENT` / `GROQ_MODEL_CHAT`
- `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY`
- `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION`
- `REDIS_URL`

**Problema:** Los clientes se inicializan como globales mutables con patron lazy singleton. No usan el sistema de dependency injection de FastAPI.

### 3.2 Capa de Multi-Tenant (`store_loader.py` + `store_resolver.py` + `stores.json`)

**`stores.json`** define 6 tiendas:

| Tienda | inbox_ids | Global | Descripcion |
|--------|-----------|--------|-------------|
| ecommer | 1-5 | Si | Tienda principal, ve todos los productos |
| sol-y-luna | 10-14 | No | Tienda de velas/aromas, aislada |
| ziru-acoustics | - | No | Audio profesional |
| legaltech | - | No | Productos legales |
| cos-store | - | No | Cosmetica |
| phybuch | - | No | Libros |

Cada tienda tiene: `inbox_map`, `channel_tokens`, `system_prompt`, `is_global`, `audience`, `language`.

**`resolve_store(inbox_id)`** resuelve el inbox_id de Chatwoot a un `StoreConfig` dataclass.

### 3.3 Capa de Clasificacion de Intenciones (`intent_classifier.py`)

Sistema dual de clasificacion:

1. **LLM (Groq Llama 3.1 8B):** Clasifica la query en categorias. Rapido, temperature=0.0, max 10 tokens.
2. **Keywords (fallback):** Regex en espanol para detectar patrones como "envio", "precio", "garantia", "hola".
3. **Fusion:** Union de resultados, eliminacion de CONVERSACIONAL si hay otras intenciones.

Categorias:
- `CATALOGO` — Busqueda de productos, precios, stock
- `POLITICAS` — Envios, devoluciones, garantias
- `INFO_GENERAL` — Pagos, facturacion, soporte tecnico
- `CONVERSACIONAL` — Saludos, agradecimientos, fuera de contexto

### 3.4 Capa de Recuperacion Vectorial (`retriever.py`)

Flujo:
1. Si es solo CONVERSACIONAL → no busca en Qdrant
2. Embedding de la query via Azure OpenAI (1536 dims)
3. Construccion de filtros Qdrant:
   - `content_type` = CATALOGO/POLITICAS/INFO_GENERAL
   - `audience` = CLIENTE
   - `metadata.channel_tokens` = tokens de la tienda
4. Busqueda top-10 en coleccion `COMPLETA`
5. Fallback: si falla con filtro `audience`, reintenta sin el

### 3.5 Capa de Memoria Conversacional (`memory.py`)

Redis como store:
- **Key:** `eia-rag:conversation:{conversation_id}:messages`
- **Estructura:** Lista de JSONs `{"role": "user|assistant", "content": "..."}`
- **Max mensajes:** 12 (FIFO trim)
- **TTL:** 86400 segundos (24 horas)
- **Degradacion graceful:** Si Redis falla, retorna historial vacio

### 3.6 Capa de Generacion (`llm_generator.py` + `templates/prompts.py`)

**Prompt del sistema** (en espanol):
- Persona: "Simetria", asistente de ventas
- 10 reglas: solo espanol, sin markdown, sin alucinaciones, max 150 tokens, URLs solo del contexto
- Placeholders: `{context_text}`, `{intent}` (NOTA: `{store_prompt}` se pasa a `build_prompt()` pero el template lo ignora — ver Known Issues)

**Formato de contexto:**
- Catalogo: `>> Producto: {name} Link: {url} Info: {attributes}`
- Documento: `>> Documento: {text}`
- URL: `>> Link: {url}`

**Historial:** Ultimos 8 mensajes de Redis se anteponen antes de la query actual.

### 3.7 Capa API (`main.py`)

4 endpoints:

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/chat` | POST | Endpoint principal RAG |
| `/health` | GET | Health check |
| `/stores` | GET | Lista todas las tiendas |
| `/stores/reload` | POST | Recarga stores.json en caliente |

---

## 4. Flujo de Datos — End to End

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. CLIENTE envia mensaje via WhatsApp/Instagram/etc                │
│    "Quiero comprar unos audifonos bluetooth, cual es el precio?"   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. CHATWOOT recibe el mensaje y hace POST a /chat                  │
│    {query: "...", conversation_id: "abc123", inbox_id: 3}          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. RESOLVER: inbox_id=3 → StoreConfig(ecommer, channel=whatsapp)   │
│    is_global=true, audience=CLIENTE                                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. CLASIFICAR: Query → LLM (Llama 3.1 8B) + Keywords              │
│    Resultado: ["CATALOGO"]                                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. EMBEDDING: Query → Azure text-embedding-3-small → [1536 dims]   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. BUSQUEDA VECTORIAL: Qdrant top-10 con filtros:                  │
│    - content_type = CATALOGO                                        │
│    - audience = CLIENTE                                             │
│    - channel_tokens includes tienda tokens                          │
│    Resultado: 5 productos de audifonos bluetooth con URLs y precios │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 7. HISTORIAL: Redis LRANGE → ultimos 8 mensajes                    │
│    ["Hola", "Buenos dias", "Que productos tienen?", "..."]         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 8. CONSTRUIR PROMPT:                                                │
│    [System Prompt de Simetria]                                      │
│    [Contexto de 5 productos encontrados]                            │
│    [Intento detectado: CATALOGO]                                    │
│    [Historial de 8 mensajes]                                        │
│    [Query del cliente]                                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 9. GENERAR: Groq Llama 3.3 70B (temp=0.2, max_tokens=150)         │
│    "Tenemos audifonos bluetooth desde $29.99... [con links]"       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 10. GUARDAR: user_message + assistant_response → Redis             │
│     RESPUESTA al cliente via Chatwoot                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Analisis de Cada Componente

### 5.1 `config.py` — Configuracion Centralizada

**Que hace:** Carga todas las variables de entorno en un objeto `Settings` con validacion de Pydantic.

**Problemas:**
- Clientes globales mutables (`_groq_client`, `_qdrant`, `_azure`, `_client`) — no thread-safe en principio
- No usa `Depends()` de FastAPI para inyeccion de dependencias
- Variables sensibles en el mismo nivel que configuracion no sensible

### 5.2 `store_loader.py` + `stores.json` — Configuracion Multi-Tenant

**Que hace:** Carga definiciones de tiendas desde un JSON estatico. Permite recarga en caliente via `/stores/reload`.

**Problemas:**
- `stores.json` es un archivo estatico — no se puede modificar sin reiniciar o llamar al endpoint de reload
- No hay validacion del contenido del JSON durante la carga
- Las tiendas sin `inbox_map` (ziru-acoustics, legaltech, cos-store, phybuch) estan configuradas pero no conectadas

### 5.3 `store_resolver.py` — Resolucion de Tenant

**Que hace:** Mapea `inbox_id` de Chatwoot a un `StoreConfig`. Si no encuentra, crea una configuración con token `__unmapped_{id}__` que no matchea con ningún producto real (fallback fail-closed).

**Problemas:**
- El fallback crea una tienda desconocida que no ve productos reales — esto es intencional (fail-closed) pero puede confundir al usuario
- No hay logging de cuando ocurre un fallback
- El mapa se carga una vez en startup; si se modifica `stores.json` via reload, el mapa se actualiza pero puede haber ventana de inconsistencia

### 5.4 `intent_classifier.py` — Clasificacion de Intenciones

**Que hace:** Clasifica la query usando LLM (rapido) + keywords (fallback). Fusiona ambos resultados.

**Fortalezas:**
- Enfoque dual robusto — si el LLM falla, keywords cubren
- Temperature=0.0 para clasificacion determinista
- Fusion inteligente: elimina CONVERSACIONAL si hay otras intenciones

**Problemas:**
- Llama 3.1 8B es util solo para clasificacion simple — queries ambiguas pueden fallar
- Los regex de keywords estan hardcodeados en espanol — no escalan a otros idiomas
- Max 10 tokens de respuesta del LLM puede cortar clasificaciones complejas

### 5.5 `retriever.py` — Busqueda Vectorial

**Que hace:** Convierte la query a embedding, busca en Qdrant con filtros de tenant/intencion, retorna top-10 resultados.

**Fortalezas:**
- Filtros inteligentes por `content_type`, `audience`, y `channel_tokens`
- Fallback automatico si el filtro `audience` causa error 400
- Degradacion graceful si Qdrant falla

**Problemas:**
- Solo 10 resultados maximos — puede ser insuficiente para queries complejas
- No hay ranking ni re-ranking de resultados
- No hay cache de embeddings — cada query genera un embedding nuevo
- Los campos de payload son consultados con multiples nombres posibles (text/content/body/policy) — fragil

### 5.6 `memory.py` — Memoria Conversacional

**Que hace:** Almacena historial de conversaciones en Redis con TTL de 24h y maximo 12 mensajes.

**Fortalezas:**
- TTL automatico limpia conversaciones viejas
- Degradacion graceful si Redis falla
- Trim FIFO mantiene el historial conciso

**Problemas:**
- `conversation_id` es controlado por el cliente — puede leer/escribir historial de otros usuarios
- No hay autenticacion del conversation_id
- Max 12 mensajes en la lista pero solo 8 se usan en el prompt — los 4 ultimos se pierden del contexto
- No hay persistencia a largo plazo — despues de 24h se borra todo

### 5.7 `llm_generator.py` + `templates/prompts.py` — Generacion

**Que hace:** Construye el prompt con sistema + contexto + historial + query, genera respuesta con Llama 3.3 70B.

**Fortalezas:**
- Prompt bien estructurado con reglas claras
- Formato de contexto diferenciado (catalogo vs documento vs URL)
- Max tokens bajo (150) para respuestas concisas

**Problemas:**
- Funcion `set_store_prompt()` declarada pero nunca usada — codigo muerto
- Variable global `_store_prompt` sin uso
- No hay streaming de respuesta — el cliente espera la respuesta completa
- Max 150 tokens puede ser insuficiente para explicaciones complejas
- No hay mecanismo de retry si Groq falla

### 5.8 `main.py` — Orquestacion API

**Que hace:** Define la aplicacion FastAPI, middleware CORS, y los 4 endpoints.

**Problemas criticos:**
- CORS abierto: `allow_origins=["*"]` con `allow_credentials=True`
- Endpoint `/stores` expone toda la configuracion de tiendas (tokens, prompts, etc.)
- Endpoint `/stores/reload` es publico — cualquiera puede recargar la configuracion
- Raw JSON parsing (`await http_request.json()`) en vez de usar Pydantic directamente
- `@app.on_event("startup")` esta deprecado en FastAPI moderno
- No hay versionado de API (no `/v1/`)

---

## 6. Por que se tomo esta Arquitectura

### 6.1 Decisiones Tecnicas Comprensibles

1. **FastAPI + Python:** Ecosistema natural para integraciones de IA. Rapido de desarrollar, buenas librerias para LLMs.

2. **Groq para clasificacion y generacion:** Groq ofrece inference ultrarapida (LPU). Usar Llama 3.1 8B para clasificacion es correcto — es barato y rapido. Llama 3.3 70B para generacion ofrece buen balance calidad/costo.

3. **Qdrant como vector DB:** Alternativa open-source a Pinecone/Weaviate. Buena para filtros por payload, despliegue gestionado en la nube.

4. **Redis para historial:** Solucion probada, baja latencia, TTL nativo para limpieza automatica. Ideal para sesiones temporales.

5. **`stores.json` como configuracion:** Facil de entender, hot-reloadable, sin necesidad de base de datos relacional para configuracion.

6. **Diseño multi-tenant via inbox_id:** Aprovecha la estructura existente de Chatwoot — cada inbox ya es un canal/tenant.

7. **Degradacion graceful:** Decision sensata — si un servicio falla (Redis, Qdrant), el sistema sigue funcionando con funcionalidad reducida.

### 6.2 Contexto del Proyecto

EIA-RAG es un **MVP (Minimum Viable Product)** o fase temprana:
- 600 lineas de codigo de aplicacion — muy poco codigo
- Enfoque en un solo proposito: responder preguntas de clientes via Chatwoot
- Mercado objetivo: Espanol, Latinoamerica
- Deploy en Railway (rapido, bajo costo, sin infraestructura compleja)

La arquitectura refleja **velocidad de iteracion** sobre robustez empresarial.

---

## 7. Fallas y Problemas Detectados

### 7.1 Problemas Criticos

| # | Problema | Ubicacion | Impacto |
|---|---------|-----------|---------|
| 1 | **Secrets en .env en el directorio de trabajo** | `.env` | Si se comparte el repositorio, todas las API keys quedan expuestas |
| 2 | **CORS abierto con credenciales** | `main.py` | Cualquier sitio web puede hacer requests con cookies |
| 3 | **Sin autenticacion/autorizacion** | Todos los endpoints | Cualquiera puede consumir LLM credits y enumerar tiendas |
| 4 | **Sin rate limiting** | `main.py` | Vulnerable a abuso y consumo excesivo |
| 5 | **conversation_id controlado por cliente** | `memory.py` | Suplantacion de identidad, lectura de historial ajeno |

### 7.2 Problemas Moderados

| # | Problema | Ubicacion | Impacto |
|---|---------|-----------|---------|
| 6 | **Endpoint `/stores` expone configuracion interna** | `main.py` | Filtra channel_tokens, system prompts, estructura de tiendas |
| 7 | **Endpoint `/stores/reload` publico** | `main.py` | Cualquiera puede manipular la configuracion en caliente |
| 8 | **Sin versionado de API** | `main.py` | Dificil hacer cambios sin romper clientes existentes |
| 9 | **Global mutable state** | `config.py` | Dificulta testing y puede causar condiciones de carrera |
| 10 | **Raw JSON parsing en `/chat`** | `main.py` | No valida con Pydantic antes de parsear, code smell |
| 11 | **Health check mentiroso** | `main.py` | Siempre retorna "ok" sin verificar Redis/Qdrant/Groq/Azure |
| 12 | **Sin metricas de ningun tipo** | Proyecto completo | Imposible medir latencia, throughput, errores, costos |
| 13 | **Sin request logging** | `main.py` | No se registra metodo, path, status code, duracion, IP |
| 14 | **Sin error tracking centralizado** | Proyecto completo | Errores se loggean localmente y se pierden |
| 15 | **Logs plain text, no estructurados** | `main.py` | Imposible parsear, buscar o analizar logs automaticamente |
| 16 | **Sin tracing distribuido** | Proyecto completo | No se puede identificar que paso del pipeline es lento |

### 7.3 Problemas Menores

| # | Problema | Ubicacion | Impacto |
|---|---------|-----------|---------|
| 17 | **`@app.on_event("startup")` deprecado** | `main.py` | FastAPI recomienda `lifespan` context manager |
| 18 | **Codigo muerto** | `llm_generator.py` | `set_store_prompt()` y `_store_prompt` sin uso |
| 19 | **Max 8 de 12 mensajes usados** | `memory.py` | 4 mensajes se guardan pero nunca se leen |
| 20 | **Fallback silencioso a tienda global** | `store_resolver.py` | inbox_id invalido obtiene datos de ecommer sin aviso |
| 21 | **Sin cache de embeddings** | `regenera el mismo embedding` | Queries similares generan embeddings redundantes |
| 22 | **Payload fragil** | `retriever.py` | Busca campos con multiples nombres (text/content/body/policy) |

---

## 8. Seguridad

### 8.1 Estado Actual de Seguridad

```
┌─────────────────────────────────────────────────────────────────┐
│                    NIVEL DE SEGURIDAD: BAJO                     │
│                                                                 │
│  Autenticacion:        ✗ NO EXISTE                             │
│  Autorizacion:         ✗ NO EXISTE                             │
│  Rate Limiting:        ✗ NO EXISTE                             │
│  CORS:                 ✗ ABIERTO (*)                           │
│  Input Validation:     ~ BASICA (Pydantic)                     │
│  Secrets Management:   ✗ EN REPOSITORIO                        │
│  HTTPS:                ~ DEPENDE DEL PROXY                     │
│  Audit Logging:        ✗ NO EXISTE                             │
│  API Versioning:       ✗ NO EXISTE                             │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Detalle de Problemas de Seguridad

#### 8.2.1 Secretos Expuestos
```python
# .env contiene (en el directorio de trabajo):
QDRANT_URL=https://xxx.qdrant.io:6333
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiI...
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_API_KEY=abc123def456...
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
REDIS_URL=redistogo:xxxxxxxx@xxx.railway.app:xxxxx
```
- Si el repositorio se comparte o hace fork, todas las credenciales quedan expuestas
- `.gitignore` excluye `.env` pero no hay verificacion de que nunca se commiteo
- No se usan vaults ni secrets managers

#### 8.2.2 CORS Abierto
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # CUALQUIER dominio
    allow_credentials=True,     # CON credenciales
    allow_methods=["*"],        # CUALQUIER metodo
    allow_headers=["*"],        # CUALQUIER header
)
```
Cualquier sitio web puede hacer requests al API incluyendo cookies y headers de autenticacion si existieran.

#### 8.2.3 Sin Autenticacion
Los endpoints son completamente publicos:
- `/chat` — consume LLM credits ($$$)
- `/stores` — expone configuracion interna
- `/stores/reload` — manipula configuracion
- `/health` — innocuo pero revela que el servicio existe

#### 8.2.4 Suplantacion de conversation_id
```python
# El cliente envia conversation_id — no se valida ni genera server-side
conversation_id = request.conversation_id or str(uuid.uuid4())
# Un atacante puede enviar cualquier ID y leer historial ajeno
```

#### 8.2.5 Sin Rate Limiting
No hay limite de requests por IP, usuario o tenant. Un atacante puede:
- Consumir todas las credits de Groq
- Sobrecargar Qdrant con queries
- Llenar Redis con conversaciones

### 8.3 Recomendaciones de Seguridad

1. **Mover secrets a un vault** (AWS Secrets Manager, HashiCorp Vault, o al menos variables de entorno del deploy)
2. **Eliminar .env del historial de git** si alguna vez se commiteo: `git filter-branch` o BFG Repo-Cleaner
3. **Agregar autenticacion:** API key basica o JWT tokens por tenant
4. **Restringir CORS** a dominios especificos de Chatwoot
5. **Rate limiting:** Usar `slowapi` o similar (100 req/min por IP)
6. **Generar conversation_id server-side** y firmarlo con HMAC
7. **Autenticar `/stores/reload`** con secret o IP whitelist
8. **No exponer `/stores`** en produccion — usarlo solo para debug
9. **Audit logging** de todas las requests con tenant_id

---

## 9. Observabilidad

### 9.1 Estado Actual de Observabilidad

```
┌─────────────────────────────────────────────────────────────────┐
│               NIVEL DE OBSERVABILIDAD: MUY BAJO                 │
│                                                                 │
│  Logging basico:       ~ 29 calls via stdlib logging            │
│  Logging estructurado: ✗ NO EXISTE (plain text)                │
│  Metricas:             ✗ NO EXISTE                             │
│  Tracing distribuido:  ✗ NO EXISTE                             │
│  Error Tracking:       ✗ NO EXISTE                             │
│  Health Checks:        ✗ SOLO STATUS ESTATICO                  │
│  Request Logging:      ✗ NO EXISTE                             │
│  Performance Monitor:  ✗ NO EXISTE                             │
│  Alertas:              ✗ NO EXISTE                             │
│  Dashboard:            ✗ NO EXISTE                             │
└─────────────────────────────────────────────────────────────────┘
```

**Resumen:** El proyecto tiene exactamente **29 llamadas a logger** repartidas en 6 archivos (15 info, 9 error, 5 warning). No hay ninguna otra forma de observabilidad. No hay metricas, no hay tracing, no hay error tracking, no hay health checks reales, no hay dashboard.

### 9.2 Desglose por Categoria

#### 9.2.1 Logging — Existe pero es Insuficiente

**Donde esta:**
- `app/main.py` — 1 `logger.info` al inicio de `/chat`
- `app/retriever.py` — logs de busqueda y errores de Qdrant
- `app/memory.py` — logs de conexion Redis y advertencias
- `app/intent_classifier.py` — logs de clasificacion y fallback
- `app/store_loader.py` — logs de carga de tiendas
- `app/store_resolver.py` — logs de resolucion de inbox

**Configuracion actual (unica en todo el proyecto):**
```python
# app/main.py lineas 17-20
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
```

**Que falta:**
- **Sin formato estructurado (JSON):** Los logs son plain text, imposibles de parsear automaticamente por herramientas como Datadog, ELK, Loki, CloudWatch
- **Sin nivel configurable:** `INFO` hardcodeado — no se puede bajar a `DEBUG` en produccion sin redeploy
- **Sin request ID:** No hay forma de correlacionar todos los logs de una misma request
- **Sin tenant ID en logs:** No se sabe que tienda genero cada log
- **Sin logs de acceso HTTP:** No se registra: metodo, path, status code, tiempo de respuesta, IP del cliente
- **Sin rotating file handler:** Los logs solo van a stdout — se pierden si el contenedor se reinicia
- **Sin logs de auditoria:** No se registra quien llamo a `/stores/reload` o `/chat`

#### 9.2.2 Metricas — No Existe

No hay ninguna libreria de metricas en las dependencias. Cero contadores, gauges, histograms.

**Metricas que deberian existir:**

| Metrica | Tipo | Por que |
|---------|------|---------|
| `requests_total` | Counter | Volumen total de requests por endpoint |
| `request_duration_seconds` | Histogram | Latencia de cada endpoint |
| `rag_queries_total` | Counter | Queries RAG por tenant, canal, intencion |
| `rag_query_duration_seconds` | Histogram | Tiempo total del pipeline RAG |
| `llm_calls_total` | Counter | Llamadas a Groq por modelo |
| `llm_call_duration_seconds` | Histogram | Latencia de Groq |
| `llm_tokens_used` | Counter | Tokens consumidos (costo) |
| `embedding_calls_total` | Counter | Llamadas a Azure OpenAI |
| `qdrant_queries_total` | Counter | Queries a Qdrant por tenant |
| `qdrant_query_duration_seconds` | Histogram | Latencia de Qdrant |
| `redis_operations_total` | Counter | Operaciones Redis |
| `redis_errors_total` | Counter | Errores de conexion Redis |
| `active_conversations` | Gauge | Conversaciones activas en Redis |
| `classification_fallback_total` | Counter | Veces que fallo el LLM y uso keywords |
| `intent_distribution` | Counter | Distribucion de intenciones detectadas |
| `error_total` | Counter | Errores por tipo y endpoint |
| `store_reload_total` | Counter | Recargas de configuracion |

**Sin estas metricas es imposible:**
- Saber si el servicio esta funcionando correctamente
- Detectar degradacion de latencia
- Entender patrones de uso por tenant
- Calcular costos de LLM por tenant
- Crear alertas automaticas
- Hacer capacity planning

#### 9.2.3 Tracing Distribuido — No Existe

No hay OpenTelemetry, Jaeger, Zipkin, ni ningun sistema de tracing.

**Que es:** Un trace unico que sigue una request desde que llega hasta que termina, cruzando todos los servicios (FastAPI → Groq → Azure → Qdrant → Redis).

**Por que importa en este caso:**
El pipeline RAG tiene 5 llamadas externas secuenciales:
```
Request → Clasificar(Groq) → Embedding(Azure) → Qdrant → Redis → Generar(Groq) → Redis(write)
```
Sin tracing, si una request tarda 5 segundos, **no sabes en que paso se perdio el tiempo**.

**Sin tracing no puedes:**
- Identificar que paso del pipeline es el mas lento
- Detectar si Groq esta lento vs Qdrant vs Redis
- Correlacionar errores entre servicios
- Hacer root cause analysis de requests lentas

#### 9.2.4 Error Tracking — No Existe

No hay Sentry, Rollbar, Bugsnag, ni ningun servicio de error tracking.

**Que pasa actualmente con los errores:**
```python
# patron comun en el codigo:
try:
    resultado = await algo()
except Exception as e:
    logger.error(f"Error: {e}")  # Se loggeo pero...
    return None  # ...se traga el error y continua
```

**Problemas:**
- Los errores se loggean pero no se agregan — no hay forma de ver "cuantos errores de Qdrant hubo hoy"
- No hay stack traces completos en un servicio centralizado
- No hay alertas cuando los errores superan un umbral
- No hay agrupacion de errores (el mismo error aparece como diferente cada vez)
- No hay contexto adicional (que tenant, que inbox_id, que query)

#### 9.2.5 Health Checks — Solo Status Estatico

```python
# app/main.py lineas 125-131
@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "service": "EIA RAG Gateway",
        "collection": settings.COLLECTION_NAME,
    }
```

**Problemas:**
- Siempre retorna `{"status": "ok"}` sin importar el estado real
- **No verifica Redis** — puede estar caido y el health check dice "ok"
- **No verifica Qdrant** — puede estar caido y el health check dice "ok"
- **No verifica Groq** — puede estar caido y el health check dice "ok"
- **No verifica Azure** — puede estar caido y el health check dice "ok"
- **No retorna version** del deploy
- **No retorna uptime**
- Railway usa este endpoint — si Redis cae, Railway sigue mandando trafico a un servicio degradado

**Health check real deberia:**
```python
{
    "status": "degraded",  # o "healthy" / "unhealthy"
    "version": "0.1.0",
    "uptime_seconds": 3600,
    "dependencies": {
        "redis": {"status": "up", "latency_ms": 2},
        "qdrant": {"status": "up", "latency_ms": 15},
        "groq": {"status": "up", "latency_ms": 120},
        "azure_openai": {"status": "up", "latency_ms": 80}
    }
}
```

#### 9.2.6 Request/Response Logging — No Existe

No hay middleware de acceso. No se registra:

| Campo | Presente? |
|-------|----------|
| Timestamp | ✗ |
| HTTP Method | ✗ |
| Path | ✗ |
| Status Code | ✗ |
| Response Time | ✗ |
| Client IP | ✗ |
| User Agent | ✗ |
| Tenant ID | ✗ |
| Request ID | ✗ |
| Body Size | ✗ |

**Sin esto es imposible:**
- Depurar requests problematicas
- Entender que clientes estan usando la API
- Detectar patrones de abuso
- Hacer auditoria de acceso

#### 9.2.7 Monitoreo de Performance — No Existe

No hay ningun mecanismo para medir:
- Tiempo total de cada request `/chat`
- Tiempo de cada paso del pipeline (clasificacion, embedding, busqueda, generacion)
- Uso de memoria del proceso
- Numero de conexiones activas
- Cola de requests

### 9.3 Impacto de la Falta de Observabilidad

Sin observabilidad, el equipo opera a ciegas:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ESCENARIOS SIN SOLUCION                      │
│                                                                 │
│  "El servicio esta lento"                                        │
│   → No sabes en que paso se tarda                               │
│                                                                 │
│  "Los clientes se quejan de respuestas malas"                   │
│   → No sabes si es clasificacion, retrieval o generacion        │
│                                                                 │
│  "Se acabo el presupuesto de Groq"                              │
│   → No sabes quien gasto ni cuanto                              │
│                                                                 │
│  "Redis se cayo y no nos dimos cuenta"                          │
│   → No hay alertas, no hay health check real                    │
│                                                                 │
│  "Una request falla intermitentemente"                          │
│   → No hay tracing, no hay request ID, no hay correlacion       │
│                                                                 │
│  "Queremos agregar una tienda nueva"                            │
│   → No sabes que volumen de trafico soportar                    │
│                                                                 │
│  "Hubo un pico de trafico inusual"                              │
│   → No hay metricas historicas para comparar                    │
└─────────────────────────────────────────────────────────────────┘
```

### 9.4 Stack de Observabilidad Recomendado

Para un proyecto de este tamano, la recomendacion minima:

```
┌─────────────────────────────────────────────────────────────────┐
│                  STACK DE OBSERVABILIDAD                        │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │  ESTRUCTURADO│   │   METRICAS   │   │   ERROR TRACKING    │  │
│  │    LOGS      │   │              │   │                     │  │
│  │              │   │  Prometheus  │   │      Sentry         │  │
│  │  structlog   │   │  + Grafana   │   │  (o equivalentes)   │  │
│  │  o loguru    │   │              │   │                     │  │
│  └──────┬───────┘   └──────┬───────┘   └──────────┬──────────┘  │
│         │                  │                      │              │
│         ▼                  ▼                      ▼              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LOKI / CLOUDWATCH / ELK                    │    │
│  │           (Almacenamiento y busqueda de logs)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            OPENTELEMETRY (Tracing)                       │    │
│  │     → Exportar traces a Jaeger / Tempo / Zipkin         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Alternativa simplificada para MVP:** Solo Sentry (errors + performance monitoring) cubre el 80% de lo necesario.

### 9.5 Que Deberia Logging Cada Request

```json
{
    "timestamp": "2025-01-15T10:30:45.123Z",
    "level": "info",
    "message": "chat_request_completed",
    "request_id": "req_abc123",
    "tenant": "ecommer",
    "inbox_id": 3,
    "channel": "whatsapp",
    "user_id": "user_456",
    "http_method": "POST",
    "path": "/chat",
    "status_code": 200,
    "response_time_ms": 2340,
    "intent_detected": ["CATALOGO"],
    "sources_used": 5,
    "conversation_id": "conv_xyz",
    "llm_model": "llama-3.3-70b",
    "llm_tokens_prompt": 850,
    "llm_tokens_completion": 120,
    "qdrant_results": 5,
    "redis_history_messages": 6
}
```

---

## 10. Escalabilidad

### 10.1 Estado Actual de Escalabilidad

```
┌─────────────────────────────────────────────────────────────────┐
│                 NIVEL DE ESCALABILIDAD: MEDIO                   │
│                                                                 │
│  Horizontal (workers):   ~ Uvicorn multi-worker posible        │
│  Multi-tenant:           ✓ Funciona con inbox_id               │
│  Base de datos:          ✓ Qdrant y Redis gestionados          │
│  Caching:                ✗ NO EXISTE                           │
│  Async:                  ✓ FastAPI nativo                       │
│  Load Balancing:         ~ Railway lo maneja                    │
│  State:                  ✗ GLOBALES MUTABLES                   │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Cuellos de Botella Identificados

1. **Groq API como dependencia critica:**
   - Toda la clasificacion y generacion depende de Groq
   - Si Groq cae, el servicio completo falla
   - No hay fallback a otro proveedor LLM

2. **Sin caching de embeddings:**
   - Cada query genera un embedding nuevo via Azure OpenAI
   - Queries similares (o identicas) repiten el proceso
   - Cuesta dinero y tiempo

3. **Sin caching de respuestas:**
   - La misma pregunta hace todo el pipeline cada vez
   - No hay cache de respuestas frecuentes

4. **Estado global mutable:**
   - `_groq_client`, `_qdrant`, `_azure` son globales
   - En multiples workers de Uvicorn, cada worker tiene su propia instancia
   - No hay conexion compartida — cada worker abre sus propias conexiones

5. **Redis como bottleneck:**
   - Lectura + escritura en cada request
   - Si Redis es lento, toda la request se retrasa
   - No hay ConnectionPool configurado explicitamente

6. **Qdrant como bottleneck:**
   - Una coleccion (`COMPLETA`) para todas las tiendas
   - Filtros por `channel_tokens` en cada query
   - Sin indices compuestos optimizados

### 10.3 Limites de Escalabilidad

| Factor | Limite Actual | Solucion |
|--------|--------------|----------|
| Tiendas | ~6 configuradas | Funciona bien, el `stores.json` escala |
| Requests/segundo | Limitado por Groq rate limits | Rate limiting + caching |
| Conversaciones simultaneas | Limitado por Redis memoria | Redis cluster o paginacion |
| Tamaño del catalogo | Limitado por Qdrant | Shardin por tenant |
| Latencia por request | ~2-4 segundos (3 llamadas LLM/Embedding) | Caching + streaming |
| Costo por query | ~$0.001-0.003 (Groq + Azure) | Caching de respuestas frecuentes |

### 10.4 Recomendaciones de Escalabilidad

1. **Agregar cache Redis para embeddings:** Hash de la query → embedding vector. Ahorra llamadas a Azure.
2. **Cache de respuestas frecuentes:** Para queries identicas, servir respuesta cacheada.
3. **Connection pooling:** Configurar pools explicitos para Redis, Qdrant, y HTTP clients.
4. **Circuit breakers:** Si Groq/Qdrant fallan, fallback rapido sin esperar timeout.
5. **Separar clasificacion de generacion:** Podrian ser servicios independientes.
6. **Multi-region:** Qdrant ya esta en `sa-east-1` — agregar replicas en otras regiones si se expande.
7. **Streaming:** Enviar tokens al cliente progresivamente para reducir percepcion de latencia.

---

## 11. Gestion del Historial Conversacional

### 11.1 Como funciona actualmente

```
┌─────────────────────────────────────────────────────────────────┐
│                    HISTORIAL EN REDIS                           │
│                                                                 │
│  Key: eia-rag:conversation:{conversation_id}:messages           │
│  Type: Redis List                                               │
│  TTL: 86400 segundos (24 horas)                                 │
│  Max elementos: 12 (FIFO trim)                                  │
│  Elementos usados: 8 (los mas recientes)                        │
│                                                                 │
│  Estructura de cada elemento:                                    │
│  {                                                              │
│    "role": "user" | "assistant",                                │
│    "content": "texto del mensaje"                               │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Flujo del Historial

```
┌──────────────────┐
│ Request llega    │
│ conversation_id  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    NO    ┌─────────────────┐
│ Redis UP?        │─────────→│ Historial vacio  │
└────────┬─────────┘          │ (degradacion)    │
         │ SI                  └─────────────────┘
         ▼
┌──────────────────┐
│ LRANGE key 0 -1  │
│ (todos los msgs)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Tomar ultimos 8  │
│ mensajes          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Agregar query     │
│ actual del usuario│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Incluir en prompt │
│ para el LLM       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ DESPUES de        │
│ generar respuesta:│
│                   │
│ RPUSH user msg    │
│ RPUSH assistant   │
│ LTRIM if > 12     │
└──────────────────┘
```

### 11.3 Problemas del Historial

1. **Sin persistencia a largo plazo:** Despues de 24 horas, toda la conversacion se borra. No hay forma de hacer analisis posterior o auditoria.

2. **conversation_id controlado por el cliente:** Un atacante puede:
   - Enviar un `conversation_id` existente y leer historial ajeno
   - Inyectar mensajes en la conversacion de otro usuario
   - No hay validacion de pertenencia (owner verification)

3. **Max 12 mensajes, solo 8 usados:** Se guardan 12 pero el prompt solo incluye 8. Los 4 mensajes mas viejos se pierden del contexto sin razon aparente.

4. **Sin compresion ni resumen:** Cada mensaje se guarda completo. Conversaciones largas consumen memoria Redis sin beneficio (se truncan de todas formas).

5. **Sin indexacion:** No hay forma de buscar en historiales anteriores ni hacer analytics.

6. **Degradacion sin aviso:** Si Redis falla, el servicio funciona pero "olvida" todo el historial. El cliente no recibe indicio de que el contexto se perdio.

### 11.4 Que se Necesita para un Historial Robusto

| Requerimiento | Estado Actual | Solucion |
|---------------|--------------|----------|
| Persistencia 24h | ✓ Redis TTL | OK para MVP |
| Persistencia largo plazo | ✗ No existe | PostgreSQL + Redis cache |
| Integridad (owner) | ✗ No validado | HMAC firmado del conversation_id |
| Resumen de conversacion | ✗ No existe | LLM summarization periodico |
| Busqueda en historial | ✗ No existe | Indexacion en BD relacional |
| Auditoria | ✗ No existe | Log de todas las interacciones |
| Multi-escenario | ✗ Solo una conversacion por usuario | Soporte para múltiples threads |

---

## 12. Recomendaciones

### 12.1 Prioridad Alta (Hacer Ahora)

1. **Eliminar .env del historial de git** si alguna vez se commiteo
2. **Agregar autenticacion basica** — API key por tenant en header
3. **Restringir CORS** a dominios de Chatwoot
4. **Rate limiting** — 100 req/min por IP
5. **Generar conversation_id server-side** con HMAC
6. **No exponer `/stores` y `/stores/reload`** en produccion
7. **Health check real** — verificar Redis, Qdrant, Groq, Azure y retornar estado real
8. **Logs estructurados (JSON)** — con request_id, tenant, timing, status code
9. **Request logging middleware** — registrar cada HTTP request con metodo, path, status, duracion

### 12.2 Prioridad Media (Sprint Siguiente)

10. **Cache de embeddings** en Redis (hash query → vector)
11. **Cache de respuestas frecuentes** para queries comunes
12. **Circuit breakers** para Groq/Qdrant
13. **Connection pooling** explicito para Redis/HTTP
14. **Error tracking centralizado** — Sentry o equivalente
15. **Metricas basicas** — Prometheus/Grafana con requests, latencia, errores, LLM tokens
16. **Migrar `@app.on_event` a `lifespan`**
17. **Nivel de log configurable** via variable de entorno `LOG_LEVEL`

### 12.3 Prioridad Baja (Roadmap)

18. **Tracing distribuido** — OpenTelemetry + Jaeger/Tempo
19. **Streaming de respuestas** via SSE
20. **Persistencia de historial** en PostgreSQL
21. **Resumenes automaticos** de conversaciones largas
22. **Multi-idioma** — internationalization del classifier
23. **Dashboard de monitoreo** — metricas de uso, latencia, costos por tenant
24. **A/B testing** de prompts y modelos
25. **Alertas automaticas** — latencia P95 > 5s, error rate > 5%, Redis down

---

## Resumen Ejecutivo

**EIA-RAG es un MVP funcional y bien pensado para su proposito.** La arquitectura RAG clasico (clasificar → recuperar → generar) es correcta y probada. El uso de Groq + Qdrant + Redis es una combinacion solida y de bajo costo.

**Los problemas principales son de seguridad, observabilidad y operaciones**, no de arquitectura:
- Sin autenticacion, sin rate limiting, secrets en el repositorio
- CORS abierto, endpoints de administracion publicos
- conversation_id controlado por el cliente
- **Cero observabilidad real** — 29 logs basicos, sin metricas, sin tracing, sin error tracking, health check mentiroso

**Para escalar a produccion estable**, las prioridades son:
1. Seguridad basica (auth, CORS, rate limiting)
2. **Observabilidad minima** (logs estructurados, metricas, error tracking, health checks reales)
3. Cache de embeddings y respuestas
4. Persistencia del historial

La arquitectura es un buen punto de partida. Los problemas identificados son corregibles sin reescribir el sistema.
