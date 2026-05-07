import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        # API Keys Externas
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.RAG_API_URL = os.getenv("RAG_API_URL")
        
        # JWT Config[cite: 1]
        self.SECRET_KEY = os.getenv("SECRET_KEY")
        self.ALGORITHM = os.getenv("ALGORITHM")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")) # 24h por defecto
        
        # --- CREDENCIALES DE ACCESO (Lo nuevo) ---
        self.ADMIN_USER = os.getenv("ADMIN_USER")
        self.ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

        # Validación inmediata[cite: 1]
        required_vars = [
            (self.GROQ_API_KEY, "GROQ_API_KEY"),
            (self.SECRET_KEY, "SECRET_KEY"),
            (self.ADMIN_USER, "ADMIN_USER"),
            (self.ADMIN_PASSWORD, "ADMIN_PASSWORD")
        ]
        
        for var, name in required_vars:
            if not var:
                raise ValueError(f"❌ Falta {name} en el archivo .env o variables de entorno")

settings = Settings()