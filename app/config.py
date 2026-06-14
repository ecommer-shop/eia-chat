import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        # API Keys Externas
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.RAG_API_URL = os.getenv("RAG_API_URL")
        self.GROQ_MODEL = "llama-3.3-70b-versatile"
        
        # JWT Config
        self.SECRET_KEY = os.getenv("SECRET_KEY")
        self.ALGORITHM = os.getenv("ALGORITHM")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")) # 24h por defecto
        
        # Credenciales de acceso
        self.ADMIN_USER = os.getenv("ADMIN_USER")
        self.ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
        
        # PostgreSQL Configuration
        self.PG_HOST = os.getenv("PG_HOST")
        self.PG_PORT = int(os.getenv("PG_PORT", "5432"))
        self.PG_DB = os.getenv("PG_DB")
        self.PG_USER = os.getenv("PG_USER")
        self.PG_PASSWORD = os.getenv("PG_PASSWORD")

        # Validación inmediata
        required_vars = [
            (self.GROQ_API_KEY, "GROQ_API_KEY"),
            (self.RAG_API_URL, "RAG_API_URL"),
            (self.SECRET_KEY, "SECRET_KEY"),
            (self.ADMIN_USER, "ADMIN_USER"),
            (self.ADMIN_PASSWORD, "ADMIN_PASSWORD"),
            (self.PG_HOST, "PG_HOST"),
            (self.PG_DB, "PG_DB"),
            (self.PG_USER, "PG_USER"),
            (self.PG_PASSWORD, "PG_PASSWORD"),
        ]
        
        for var, name in required_vars:
            if not var:
                raise ValueError(f"❌ Falta {name} en el archivo .env o variables de entorno")

settings = Settings()