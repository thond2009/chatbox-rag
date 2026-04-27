import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "version" in data


def test_info():
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "RAG Chatbot System"


def test_chat_without_documents():
    response = client.post(
        "/api/v1/chat",
        json={"query": "Hello, what can you do?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "session_id" in data


def test_chat_empty_query():
    response = client.post(
        "/api/v1/chat",
        json={"query": ""},
    )
    assert response.status_code == 422


def test_list_documents():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data


def test_chat_history():
    session_id = "test-session-123"
    response = client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "query": "Hello"},
    )
    assert response.status_code == 200

    response = client.get(f"/api/v1/history/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert len(data["history"]) == 2


def test_clear_chat_history():
    session_id = "test-clear-session"
    client.post("/api/v1/chat", json={"session_id": session_id, "query": "Test"})

    response = client.delete(f"/api/v1/history/{session_id}")
    assert response.status_code == 200

    response = client.get(f"/api/v1/history/{session_id}")
    data = response.json()
    assert len(data["history"]) == 0
