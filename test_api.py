#!/usr/bin/env python
"""
Script de testing rápido para verificar que el API funciona correctamente.
Requiere que el servidor esté corriendo en http://localhost:8001
"""

import asyncio
import httpx
import json
from uuid import uuid4

BASE_URL = "http://localhost:8001"

# Tenant de prueba (debe existir en tu BD o usar cualquier UUID)
TENANT_ID = str(uuid4())

async def test_health():
    """Prueba el endpoint de health"""
    print("\n🔍 Testing /health...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200

async def test_login():
    """Prueba el endpoint de login"""
    print("\n🔍 Testing /token...")
    async with httpx.AsyncClient() as client:
        data = {
            "username": "admin",
            "password": "12345"
        }
        response = await client.post(
            f"{BASE_URL}/token",
            data=data  # OAuth2PasswordRequestForm usa form data
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        assert response.status_code == 200
        assert "access_token" in result
        return result["access_token"]

async def test_chat_whatsapp(token: str):
    """Prueba endpoint /chat desde WhatsApp"""
    print("\n🔍 Testing /chat (WhatsApp)...")
    
    payload = {
        "query": "¿Tienen tenis Nike talla 42?",
        "tenant_id": TENANT_ID,
        "channel": "whatsapp",
        "external_id": "+573001234567",
        "contact_name": "Kevin"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/chat",
            json=payload,
            headers=headers
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            assert "answer" in result
            assert "conversation_id" in result
            assert "contact_id" in result
            return result["conversation_id"]
        else:
            print(f"Error: {response.text}")
            raise Exception(f"Chat request failed: {response.status_code}")

async def test_chat_instagram(token: str):
    """Prueba endpoint /chat desde Instagram"""
    print("\n🔍 Testing /chat (Instagram)...")
    
    payload = {
        "query": "¿Hacen envíos a Bogotá?",
        "tenant_id": TENANT_ID,
        "channel": "instagram",
        "external_id": "kevin.ig.user123",
        "contact_name": "Kevin IG"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/chat",
            json=payload,
            headers=headers
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")

async def test_chat_web(token: str):
    """Prueba endpoint /chat desde Web"""
    print("\n🔍 Testing /chat (Web)...")
    
    payload = {
        "query": "¿Cuál es la política de devoluciones?",
        "tenant_id": TENANT_ID,
        "channel": "web",
        "external_id": f"session-{uuid4()}",
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/chat",
            json=payload,
            headers=headers
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")

async def main():
    """Ejecuta todos los tests"""
    print("=" * 70)
    print("🚀 EIA Chat Gateway API - Test Suite")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print(f"Tenant ID: {TENANT_ID}")
    
    try:
        # 1. Health check
        await test_health()
        
        # 2. Login
        token = await test_login()
        
        # 3. Chat tests
        conv_id = await test_chat_whatsapp(token)
        await test_chat_instagram(token)
        await test_chat_web(token)
        
        print("\n" + "=" * 70)
        print("✅ TODOS LOS TESTS PASARON!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
