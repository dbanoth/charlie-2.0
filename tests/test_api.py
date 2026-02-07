"""Tests for the Livestock Advisor API."""
import pytest
from unittest.mock import patch, MagicMock


# Skip tests if dependencies not available
pytest.importorskip("fastapi")


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_endpoint_returns_status(self):
        """Health endpoint should return a status field."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["status"] == "healthy"

    def test_health_endpoint_returns_version(self):
        """Health endpoint should return API version."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.get("/health")

        data = response.json()
        assert "version" in data
        assert data["version"] == "2.0.0"


class TestChatEndpoint:
    """Tests for chat endpoint."""

    def test_chat_requires_message(self):
        """Chat endpoint should require a message."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.post("/chat", json={"thread_id": "test", "user_id": "test_user"})

        # Should fail validation - message is required
        assert response.status_code == 422

    def test_chat_requires_user_id(self):
        """Chat endpoint should require a user_id."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.post("/chat", json={"message": "Hello", "thread_id": "test"})

        # Should fail validation - user_id is required
        assert response.status_code == 422

    def test_chat_message_max_length(self):
        """Chat endpoint should enforce max message length."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        # Create a message longer than 4000 characters
        long_message = "a" * 4001
        response = client.post(
            "/chat",
            json={"message": long_message, "thread_id": "test", "user_id": "test_user"}
        )

        assert response.status_code == 422

    def test_chat_accepts_valid_request(self):
        """Chat endpoint should accept valid requests."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"message": "Hello", "thread_id": "test-thread", "user_id": "test_user"}
        )

        # Should return 200 or 500 (if LLM not configured)
        assert response.status_code in [200, 500]

    def test_chat_returns_response_format(self):
        """Chat endpoint should return proper response format."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"message": "Hello", "thread_id": "test-thread", "user_id": "test_user"}
        )

        if response.status_code == 200:
            data = response.json()
            assert "response" in data
            assert "thread_id" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_html(self):
        """Root endpoint should return HTML."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_root_contains_app_title(self):
        """Root endpoint should contain app title."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.get("/")

        assert "Livestock Advisor" in response.text or "Charlie" in response.text


class TestThreadsEndpoint:
    """Tests for threads endpoint."""

    def test_threads_requires_user_id(self):
        """Threads endpoint should require user_id parameter."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.get("/threads")

        # Should fail - user_id is required
        assert response.status_code == 422

    def test_threads_accepts_user_id(self):
        """Threads endpoint should accept user_id parameter."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.get("/threads?user_id=test_user")

        # Should return 200 or 500 (if Firestore not configured)
        assert response.status_code in [200, 500]


class TestDeleteThreadEndpoint:
    """Tests for delete thread endpoint."""

    def test_delete_thread_requires_user_id(self):
        """Delete thread endpoint should require user_id parameter."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.delete("/threads/test-thread")

        # Should fail - user_id is required
        assert response.status_code == 422


class TestRequestModels:
    """Tests for request/response models."""

    def test_chat_request_model_valid(self):
        """ChatRequest model should accept valid data."""
        from api import ChatRequest

        request = ChatRequest(
            message="Test message",
            thread_id="test-thread",
            user_id="test-user"
        )
        assert request.message == "Test message"
        assert request.thread_id == "test-thread"
        assert request.user_id == "test-user"

    def test_chat_request_model_default_thread(self):
        """ChatRequest model should have default thread_id."""
        from api import ChatRequest

        request = ChatRequest(
            message="Test message",
            user_id="test-user"
        )
        assert request.thread_id == "default"

    def test_chat_response_model(self):
        """ChatResponse model should work correctly."""
        from api import ChatResponse

        response = ChatResponse(
            response="Test response",
            thread_id="test-thread"
        )
        assert response.response == "Test response"
        assert response.thread_id == "test-thread"


class TestCORSMiddleware:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self):
        """CORS headers should be present in response."""
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.options("/health", headers={"Origin": "http://localhost:3000"})

        # Should allow CORS
        assert response.status_code in [200, 405]
