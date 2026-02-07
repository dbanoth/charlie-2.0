"""Chat history storage using Google Cloud Firestore."""
from datetime import datetime
from typing import List, Dict, Any, Optional
from google.cloud import firestore

from config import GCP_PROJECT, FIRESTORE_DATABASE, CHAT_HISTORY_COLLECTION


class ChatHistoryStore:
    """Manages persistent chat history storage in Firestore."""

    def __init__(self):
        self._db = None

    @property
    def firestore_db(self):
        """Lazy initialization of Firestore client."""
        if self._db is None:
            self._db = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)
            print(f"[ChatHistory] Connected to Firestore (Project: {GCP_PROJECT})")
        return self._db

    @property
    def collection(self):
        """Get the chat history collection."""
        return self.firestore_db.collection(CHAT_HISTORY_COLLECTION)

    def _get_doc_id(self, user_id: str, thread_id: str) -> str:
        """Generate document ID from user_id and thread_id."""
        return f"{user_id}_{thread_id}"

    def get_messages(self, user_id: str, thread_id: str) -> List[Dict[str, Any]]:
        """
        Load chat history for a user's thread.

        Args:
            user_id: The user identifier
            thread_id: The conversation thread identifier

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        doc_id = self._get_doc_id(user_id, thread_id)
        doc = self.collection.document(doc_id).get()

        if doc.exists:
            data = doc.to_dict()
            messages = data.get("messages", [])
            # Strip timestamps for agent compatibility
            return [{"role": m["role"], "content": m["content"]} for m in messages]

        return []

    def save_messages(self, user_id: str, thread_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Save chat history for a user's thread.

        Args:
            user_id: The user identifier
            thread_id: The conversation thread identifier
            messages: List of message dictionaries with 'role' and 'content' keys
        """
        doc_id = self._get_doc_id(user_id, thread_id)
        doc_ref = self.collection.document(doc_id)

        now = datetime.utcnow()

        # Add timestamps to messages if not present
        timestamped_messages = []
        for msg in messages:
            timestamped_msg = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp", now.isoformat() + "Z")
            }
            timestamped_messages.append(timestamped_msg)

        # Check if document exists to set created_at appropriately
        existing_doc = doc_ref.get()

        if existing_doc.exists:
            # Update existing document
            doc_ref.update({
                "messages": timestamped_messages,
                "updated_at": now,
                "message_count": len(timestamped_messages)
            })
        else:
            # Create new document
            doc_ref.set({
                "user_id": user_id,
                "thread_id": thread_id,
                "messages": timestamped_messages,
                "created_at": now,
                "updated_at": now,
                "message_count": len(timestamped_messages)
            })

    def get_user_threads(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get list of conversation threads for a user.

        Args:
            user_id: The user identifier
            limit: Maximum number of threads to return

        Returns:
            List of thread summaries with thread_id, message_count, and timestamps
        """
        try:
            # Query by user_id only (no composite index needed)
            query = self.collection.where("user_id", "==", user_id)

            threads = []
            for doc in query.stream():
                data = doc.to_dict()
                messages = data.get("messages", [])
                # Get first user message as preview
                preview = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        preview = msg.get("content", "")[:100]
                        break
                if not preview and messages:
                    preview = messages[0].get("content", "")[:100]

                # Handle Firestore timestamp conversion
                updated_at = data.get("updated_at")
                if updated_at and hasattr(updated_at, 'isoformat'):
                    updated_at = updated_at
                elif updated_at:
                    # Convert Firestore timestamp to datetime if needed
                    try:
                        updated_at = updated_at.datetime() if hasattr(updated_at, 'datetime') else updated_at
                    except Exception:
                        updated_at = None

                threads.append({
                    "thread_id": data.get("thread_id"),
                    "message_count": data.get("message_count", 0),
                    "created_at": data.get("created_at"),
                    "updated_at": updated_at,
                    "preview": preview
                })

            # Sort by updated_at in Python (avoids needing composite index)
            def get_sort_key(x):
                updated = x.get("updated_at")
                if updated is None:
                    return datetime.min
                if hasattr(updated, 'timestamp'):
                    return updated
                return updated if isinstance(updated, datetime) else datetime.min

            threads.sort(key=get_sort_key, reverse=True)

            return threads[:limit]
        except Exception as e:
            print(f"[ChatHistory] Error getting user threads: {e}")
            return []

    def delete_thread(self, user_id: str, thread_id: str) -> bool:
        """
        Delete a conversation thread.

        Args:
            user_id: The user identifier
            thread_id: The conversation thread identifier

        Returns:
            True if deleted, False if not found
        """
        doc_id = self._get_doc_id(user_id, thread_id)
        doc_ref = self.collection.document(doc_id)

        if doc_ref.get().exists:
            doc_ref.delete()
            return True
        return False


# Singleton instance
chat_store = ChatHistoryStore()
