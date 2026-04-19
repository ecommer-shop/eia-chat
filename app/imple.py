import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from app.config import settings

# 1. Configurar el "Juez" (Usamos tu Azure OpenAI por su estabilidad)
# Nota: RAGAS internamente usa Langchain, así que inicializamos los wrappers de Langchain
azure_evaluator_llm = AzureChatOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment="TU_DEPLOYMENT_DE_GPT4_O_GPT35" # El modelo que evalúa
)

azure_evaluator_embeddings = AzureOpenAIEmbeddings(
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT
)

# 2. Armar el Dataset de Prueba (Ejemplo con tu log anterior)
data_samples = {
    "question": ["¿Qué es Ecommer?"],
    # La respuesta real que generó tu Llama-3 en Groq
    "answer": ["Ecommer es una plataforma que permite a negocios del Cauca vender por internet de forma profesional y legal..."],
    # Los textos crudos que devolvió Qdrant (los payloads que vimos en tu log)
    "contexts": [[
        "Ecommer es la plataforma que le permite a cualquier negocio del Cauca vender por internet...",
        "Ecommer was born from a simple observation: while large companies thrive online...",
        "At Ecommer, our mission is simple: Empower PYMEs with accessible, scalable e-commerce..."
    ]],
    # (Opcional) La respuesta ideal redactada por un humano (Ground Truth) para medir Context Recall
    "ground_truth": ["Ecommer es una infraestructura digital para microempresas del Cauca que permite vender sin saber tecnología."]
}

dataset = Dataset.from_dict(data_samples)

# 3. Ejecutar la Evaluación
print("🧠 Iniciando evaluación RAGAS...")
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,       # ¿Alucinó el LLM?
        answer_relevance,   # ¿Respondió lo que era?
        context_precision   # ¿Qdrant hizo bien su trabajo?
    ],
    llm=azure_evaluator_llm,
    embeddings=azure_evaluator_embeddings
)

print("\n📊 Resultados de Calidad:")
print(result)