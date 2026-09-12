"""Fase 3 — Agente conversacional a mano (sin framework).

Provee el núcleo del agente (`run_agent`) y su registro de herramientas.

El endpoint `/chat` queda intacto como pipeline de referencia de la fase 1;
el agente reemplazará el flujo actual en la fase 6 vía un endpoint nuevo
(`/agent/chat`), solo cuando supere el baseline del golden set.
"""

from app.agent.core import AgentResult, run_agent

__all__ = ["AgentResult", "run_agent"]