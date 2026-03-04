"""Tests for the Financial QA Agent API."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.financial_qa_agent.main import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ask_success(client):
    resp = await client.post("/api/ask", json={"question": "What is inflation?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["data"]["question"] == "What is inflation?"
    assert isinstance(body["data"]["answer"], str)
    assert len(body["data"]["answer"]) > 0


@pytest.mark.asyncio
async def test_ask_empty_question(client):
    resp = await client.post("/api/ask", json={"question": "   "})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["data"] is None


@pytest.mark.asyncio
async def test_ask_missing_question(client):
    resp = await client.post("/api/ask", json={})
    assert resp.status_code == 422  # Pydantic validation error
