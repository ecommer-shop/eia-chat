# EIA Chat - Gateway Bot API 🤖

Este proyecto es la API de entrada (Gateway) para un sistema de **RAG (Retrieval-Augmented Generation)** enfocado en E-commerce. Se encarga de recibir las consultas de los usuarios, orquestar la obtención de contexto desde un servicio externo y generar respuestas inteligentes utilizando LLMs.

## 🚀 Tecnologías Principales

- **Lenguaje:** [Python 3.13](https://www.python.org/)
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Gestor de Paquetes:** [uv](https://docs.astral.sh/uv/) (Extremadamente rápido, reemplaza a pip)
- **LLM:** [Groq](https://groq.com/) (Llama 3 / Mixtral)
- **Contenerización:** [Docker](https://www.docker.com/)

## 📂 Estructura del Proyecto

```text
Api_RAG/
├── app/                    # Código fuente de la aplicación
│   ├── main.py             # Punto de entrada de FastAPI
│   ├── config.py           # Configuración y variables de entorno
│   ├── llm_generator.py    # Lógica de generación con LLMs (Groq)
│   ├── rag_client.py       # Cliente para consultar el servicio de contexto (RAG)
│   ├── schemas.py          # Modelos de datos (Pydantic)
│   └── dev.py              # Scripts de utilidad para desarrollo
├── Dockerfile              # Configuración de imagen optimizada con uv
├── pyproject.toml          # Definición de dependencias y proyecto (estándar PEP 621)
├── uv.lock                 # Archivo de bloqueo para instalaciones deterministas
└── .env                    # Variables sensibles (no incluido en git)
```

## 🛠️ Configuración Local

### Requisitos Previos
1. Tener instalado [uv](https://docs.astral.sh/uv/):
   ```bash
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

### Instalación y Ejecución
1. **Clonar el repositorio** e ir a la carpeta:
   ```bash
   cd Api_RAG
   ```
2. **Sincronizar dependencias**:
   ```bash
   uv sync
   ```
3. **Configurar el entorno**: Crea un archivo `.env` basado en `.env.example` (si existe) o añade:
   ```env
   GROQ_API_KEY=tu_api_key_aqui
   RAG_API_URL=https://tu-servicio-rag.com
   ```
4. **Ejecutar en modo desarrollo**:
   ```bash
   uv run dev
   ```

## 🐳 Docker

Para desplegar usando Docker (optimizada para producción):

```bash
docker build -t eia-chat-gateway .
docker run -p 8000:8000 --env-file .env eia-chat-gateway
```

## 📡 Endpoints Principales

- `POST /chat`: Recibe una pregunta, consulta el contexto y devuelve la respuesta del bot.
- `GET /health`: Estado de salud del servicio.
- `GET /docs`: Documentación interactiva Swagger.

---
Desarrollado con ❤️ para el ecosistema E-commerce.
