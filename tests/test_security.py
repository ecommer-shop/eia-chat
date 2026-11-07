"""La clave de API protege los endpoints sensibles solo cuando EIA_API_KEY está definida."""
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

CLAVE = "clave-de-prueba-no-real"
PROTEGIDOS = [
    ("post", "/agent/chat", {"json": {}}),
    ("get", "/stores", {}),
    ("post", "/stores/reload", {}),
]


@pytest.fixture
def client():
    # Sin `with`: no ejecuta el lifespan (que exige variables reales de Qdrant, Azure, etc.).
    return TestClient(app)


@pytest.mark.parametrize("metodo,ruta,extra", PROTEGIDOS)
def test_sin_clave_configurada_no_exige_autenticacion(client, monkeypatch, metodo, ruta, extra):
    monkeypatch.setattr(settings, "EIA_API_KEY", "")
    r = getattr(client, metodo)(ruta, **extra)
    assert r.status_code != 401


@pytest.mark.parametrize("metodo,ruta,extra", PROTEGIDOS)
@pytest.mark.parametrize(
    "cabeceras",
    [
        {},
        {"X-API-Key": ""},
        {"X-API-Key": "incorrecta"},
        {"Authorization": "Bearer incorrecta"},
        {"Authorization": "Basic " + CLAVE},  # esquema que no es Bearer
    ],
)
def test_con_clave_configurada_rechaza_ausente_o_incorrecta(client, monkeypatch, metodo, ruta, extra, cabeceras):
    monkeypatch.setattr(settings, "EIA_API_KEY", CLAVE)
    r = getattr(client, metodo)(ruta, headers=cabeceras, **extra)
    assert r.status_code == 401
    assert r.json() == {"detail": "No autorizado"}  # no revela si falló por ausente o por incorrecta


@pytest.mark.parametrize("cabeceras", [{"X-API-Key": CLAVE}, {"Authorization": f"Bearer {CLAVE}"}])
def test_con_clave_correcta_pasa_la_autenticacion(client, monkeypatch, cabeceras):
    monkeypatch.setattr(settings, "EIA_API_KEY", CLAVE)
    # Cuerpo vacío: pasa la autenticación y falla después en la validación (422), sin ejecutar la lógica.
    r = client.post("/agent/chat", json={}, headers=cabeceras)
    assert r.status_code == 422


def test_health_sigue_abierto_aunque_haya_clave(client, monkeypatch):
    monkeypatch.setattr(settings, "EIA_API_KEY", CLAVE)
    assert client.get("/health").status_code == 200
