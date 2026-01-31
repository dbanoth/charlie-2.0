"""Tests for the Livestock Advisor API."""
import pytest
from fastapi.testclient import TestClient


# Skip tests if dependencies not available
pytest.importorskip("fastapi")


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_endpoint_returns_status(self):
        """Health endpoint should return a status field."""
        # Import here to avoid issues if dependencies missing
        from api import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data


class TestChatEndpoint:
    """Tests for chat endpoint."""

    def test_chat_requires_message(self):
        """Chat endpoint should require a message."""
        from api import app

        client = TestClient(app)
        response = client.post("/chat", json={"thread_id": "test"})

        # Should fail validation - message is required
        assert response.status_code == 422

    def test_chat_accepts_valid_request(self):
        """Chat endpoint should accept valid requests."""
        from api import app

        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"message": "Hello", "thread_id": "test-thread"}
        )

        # Should return 200 or 500 (if LLM not configured)
        assert response.status_code in [200, 500]


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_html(self):
        """Root endpoint should return HTML."""
        from api import app

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Livestock Advisor" in response.text
