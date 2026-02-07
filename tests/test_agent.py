"""Tests for the LangGraph agent module."""
import pytest
from unittest.mock import patch, MagicMock


class TestChatState:
    """Tests for ChatState type definition."""

    def test_chat_state_structure(self):
        """ChatState should have expected fields."""
        from agent import ChatState

        state: ChatState = {
            "messages": [],
            "user_input": "test input",
            "context": "test context",
            "response": "test response",
            "query_type": "livestock"
        }

        assert state["messages"] == []
        assert state["user_input"] == "test input"
        assert state["context"] == "test context"
        assert state["response"] == "test response"
        assert state["query_type"] == "livestock"


class TestPrompts:
    """Tests for prompt templates."""

    def test_classification_prompt_exists(self):
        """Classification prompt should be defined."""
        from agent import CLASSIFICATION_PROMPT

        assert CLASSIFICATION_PROMPT is not None
        assert "{question}" in CLASSIFICATION_PROMPT
        assert "livestock" in CLASSIFICATION_PROMPT.lower()

    def test_system_prompt_exists(self):
        """System prompt should be defined."""
        from agent import SYSTEM_PROMPT

        assert SYSTEM_PROMPT is not None
        assert "livestock" in SYSTEM_PROMPT.lower()

    def test_general_prompt_exists(self):
        """General prompt should be defined."""
        from agent import GENERAL_PROMPT

        assert GENERAL_PROMPT is not None


class TestClassifyQuery:
    """Tests for query classification function."""

    def test_classify_query_empty_input(self):
        """Classify query should handle empty input."""
        from agent import classify_query

        state = {"user_input": ""}
        result = classify_query(state)

        assert "query_type" in result
        assert result["query_type"] == "general"

    def test_classify_query_missing_input(self):
        """Classify query should handle missing input."""
        from agent import classify_query

        state = {}
        result = classify_query(state)

        assert "query_type" in result
        assert result["query_type"] == "general"


class TestRouteQuery:
    """Tests for query routing function."""

    def test_route_query_livestock(self):
        """Route query should return 'retrieve' for livestock queries."""
        from agent import route_query

        state = {"query_type": "livestock"}
        result = route_query(state)

        assert result == "retrieve"

    def test_route_query_general(self):
        """Route query should return 'generate_direct' for general queries."""
        from agent import route_query

        state = {"query_type": "general"}
        result = route_query(state)

        assert result == "generate_direct"

    def test_route_query_default(self):
        """Route query should default to 'general' for unknown types."""
        from agent import route_query

        state = {}
        result = route_query(state)

        assert result == "generate_direct"


class TestRetrieveContext:
    """Tests for context retrieval function."""

    def test_retrieve_context_empty_input(self):
        """Retrieve context should handle empty input."""
        from agent import retrieve_context

        state = {"user_input": ""}
        result = retrieve_context(state)

        assert "context" in result
        assert result["context"] == ""

    def test_retrieve_context_missing_input(self):
        """Retrieve context should handle missing input."""
        from agent import retrieve_context

        state = {}
        result = retrieve_context(state)

        assert "context" in result
        assert result["context"] == ""


class TestBuildGraph:
    """Tests for graph building function."""

    def test_build_graph_returns_compiled_graph(self):
        """Build graph should return a compiled graph."""
        from agent import build_graph

        graph = build_graph()

        assert graph is not None
        assert hasattr(graph, "invoke")


class TestAgentSingleton:
    """Tests for agent singleton."""

    def test_agent_exists(self):
        """Agent singleton should be created."""
        from agent import agent

        assert agent is not None
        assert hasattr(agent, "invoke")


class TestChatFunction:
    """Tests for the main chat function."""

    def test_chat_function_exists(self):
        """Chat function should be defined."""
        from agent import chat

        assert callable(chat)

    def test_chat_function_signature(self):
        """Chat function should have correct signature."""
        import inspect
        from agent import chat

        sig = inspect.signature(chat)
        params = list(sig.parameters.keys())

        assert "user_input" in params
        assert "thread_id" in params
        assert "user_id" in params


class TestInitializeFunction:
    """Tests for the initialize function."""

    def test_initialize_function_exists(self):
        """Initialize function should be defined."""
        from agent import initialize

        assert callable(initialize)
