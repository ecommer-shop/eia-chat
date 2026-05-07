import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.config import settings  # Importamos tu configuración

# OAuth2PasswordBearer le dice a FastAPI dónde está el endpoint para obtener el token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    """Crea el JWT empaquetando el tenant_id y el usuario."""
    to_encode = data.copy()
    
    # Calculamos la fecha de expiración usando tu configuración
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    # Firmamos con tu SECRET_KEY y ALGORITHM
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user_tenant(token: str = Depends(oauth2_scheme)):
    """Desencripta el JWT y extrae el tenant para usarlo en Qdrant/PostgreSQL."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Desencriptamos usando tu llave
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        tenant_id: str = payload.get("tenant_id")
        username: str = payload.get("sub")
        
        if username is None or tenant_id is None:
            raise credentials_exception
            
        return {"username": username, "tenant_id": tenant_id}
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="El token ha expirado")
    except jwt.PyJWTError:
        raise credentials_exception