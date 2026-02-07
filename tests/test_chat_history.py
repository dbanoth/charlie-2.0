"""Tests for the chat history storage module."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestChatHistoryStore:
    """Tests for ChatHistoryStore class."""

    def test_chat_history_store_init(self):
        """ChatHistoryStore should initialize with None db."""
        from chat_history import ChatHistoryStore

        store = ChatHistoryStore()
        assert store._db is None

    def test_get_doc_id(self):
        """_get_doc_id should generate correct document ID."""
        from chat_history import ChatHistoryStore

        store = ChatHistoryStore()
        doc_id = store._get_doc_id("user123", "thread456")

        assert doc_id == "user123_thread456"

    def test_get_doc_id_with_special_chars(self):
        """_get_doc_id should handle special characters."""
        from chat_history import ChatHistoryStore

        store = ChatHistoryStore()
        doc_id = store._get_doc_id("user_123", "thread-456")

        assert doc_id == "user_123_thread-456"


class TestChatStoreSingleton:
    """Tests for chat_store singleton."""

    def test_chat_store_singleton_exists(self):
        """chat_store singleton should be created."""
        from chat_history import chat_store

        assert chat_store is not None

    def test_chat_store_is_chat_history_store(self):
        """chat_store should be instance of ChatHistoryStore."""
        from chat_history import chat_store, ChatHistoryStore

        assert isinstance(chat_store, ChatHistoryStore)


class TestChatHistoryStoreMethods:
    """Tests for ChatHistoryStore methods with mocked Firestore."""

    @patch('chat_history.firestore')
    def test_get_messages_empty(self, mock_firestore):
        """get_messages should return empty list for non-existent doc."""
        from chat_history import ChatHistoryStore

        mock_doc = MagicMock()
        mock_doc.exists = False

        mock_collection = MagicMock()
        mock_collection.document.return_value.get.return_value = mock_doc

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        store = ChatHistoryStore()
        store._db = mock_db

        messages = store.get_messages("user1", "thread1")

        assert messages == []

    @patch('chat_history.firestore')
    def test_get_messages_with_data(self, mock_firestore):
        """get_messages should return formatted messages."""
        from chat_history import ChatHistoryStore

        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {
            "messages": [
                {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00Z"},
                {"role": "assistant", "content": "Hi there!", "timestamp": "2024-01-01T00:00:01Z"}
            ]
        }

        mock_collection = MagicMock()
        mock_collection.document.return_value.get.return_value = mock_doc

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        store = ChatHistoryStore()
        store._db = mock_db

        messages = store.get_messages("user1", "thread1")

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}

    @patch('chat_history.firestore')
    def test_save_messages_new_doc(self, mock_firestore):
        """save_messages should create new document."""
        from chat_history import ChatHistoryStore

        mock_doc_ref = MagicMock()
        mock_existing_doc = MagicMock()
        mock_existing_doc.exists = False
        mock_doc_ref.get.return_value = mock_existing_doc

        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        store = ChatHistoryStore()
        store._db = mock_db

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]

        store.save_messages("user1", "thread1", messages)

        mock_doc_ref.set.assert_called_once()

    @patch('chat_history.firestore')
    def test_save_messages_existing_doc(self, mock_firestore):
        """save_messages should update existing document."""
        from chat_history import ChatHistoryStore

        mock_doc_ref = MagicMock()
        mock_existing_doc = MagicMock()
        mock_existing_doc.exists = True
        mock_doc_ref.get.return_value = mock_existing_doc

        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        store = ChatHistoryStore()
        store._db = mock_db

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]

        store.save_messages("user1", "thread1", messages)

        mock_doc_ref.update.assert_called_once()

    @patch('chat_history.firestore')
    def test_delete_thread_exists(self, mock_firestore):
        """delete_thread should return True for existing thread."""
        from chat_history import ChatHistoryStore

        mock_doc_ref = MagicMock()
        mock_existing_doc = MagicMock()
        mock_existing_doc.exists = True
        mock_doc_ref.get.return_value = mock_existing_doc

        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        store = ChatHistoryStore()
        store._db = mock_db

        result = store.delete_thread("user1", "thread1")

        assert result is True
        mock_doc_ref.delete.assert_called_once()

    @patch('chat_history.firestore')
    def test_delete_thread_not_exists(self, mock_firestore):
        """delete_thread should return False for non-existent thread."""
        from chat_history import ChatHistoryStore

        mock_doc_ref = MagicMock()
        mock_existing_doc = MagicMock()
        mock_existing_doc.exists = False
        mock_doc_ref.get.return_value = mock_existing_doc

        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        store = ChatHistoryStore()
        store._db = mock_db

        result = store.delete_thread("user1", "thread1")

        assert result is False
        mock_doc_ref.delete.assert_not_called()

    @patch('chat_history.firestore')
    def test_get_user_threads_empty(self, mock_firestore):
        """get_user_threads should return empty list when no threads."""
        from chat_history import ChatHistoryStore

        mock_query = MagicMock()
        mock_query.stream.return_value = []

        mock_collection = MagicMock()
        mock_collection.where.return_value = mock_query

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        store = ChatHistoryStore()
        store._db = mock_db

        threads = store.get_user_threads("user1")

        assert threads == []

    @patch('chat_history.firestore')
    def test_get_user_threads_with_data(self, mock_firestore):
        """get_user_threads should return formatted thread list."""
        from chat_history import ChatHistoryStore

        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "thread_id": "thread1",
            "user_id": "user1",
            "messages": [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Hi!"}
            ],
            "message_count": 2,
            "created_at": datetime(2024, 1, 1),
            "updated_at": datetime(2024, 1, 2)
        }

        mock_query = MagicMock()
        mock_query.stream.return_value = [mock_doc]

        mock_collection = MagicMock()
        mock_collection.where.return_value = mock_query

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        store = ChatHistoryStore()
        store._db = mock_db

        threads = store.get_user_threads("user1")

        assert len(threads) == 1
        assert threads[0]["thread_id"] == "thread1"
        assert threads[0]["message_count"] == 2
        assert threads[0]["preview"] == "Hello world"


class TestMessageTimestamps:
    """Tests for message timestamp handling."""

    @patch('chat_history.firestore')
    def test_messages_get_timestamps(self, mock_firestore):
        """save_messages should add timestamps to messages."""
        from chat_history import ChatHistoryStore

        mock_doc_ref = MagicMock()
        mock_existing_doc = MagicMock()
        mock_existing_doc.exists = False
        mock_doc_ref.get.return_value = mock_existing_doc

        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        store = ChatHistoryStore()
        store._db = mock_db

        messages = [{"role": "user", "content": "Hello"}]
        store.save_messages("user1", "thread1", messages)

        call_args = mock_doc_ref.set.call_args[0][0]
        saved_messages = call_args["messages"]

        assert "timestamp" in saved_messages[0]
        assert saved_messages[0]["role"] == "user"
        assert saved_messages[0]["content"] == "Hello"
