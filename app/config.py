import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.RAG_API_URL = os.getenv("RAG_API_URL", "https://ragquery-production-7fe7.up.railway.app")
        
        # Validación inmediata en el constructor
        if not self.GROQ_API_KEY:
            raise ValueError("❌ Falta GROQ_API_KEY en el archivo .env")

# Instancia global
settings = Settings()