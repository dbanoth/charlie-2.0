"""FastAPI server for the Agriculture RAG Chatbot - Production Ready."""
import os
import logging
import time
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from typing import List
from agent import chat, initialize
from database import db
from config import GCP_PROJECT
from chat_history import chat_store

# Configure structured logging for GCP Cloud Logging
class StructuredLogFormatter(logging.Formatter):
    """Format logs as JSON for Cloud Logging."""
    def format(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "logging.googleapis.com/sourceLocation": {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

# Setup logging
logger = logging.getLogger("livestock-advisor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

# Use structured logging in production (GCP), simple format locally
if GCP_PROJECT:
    handler.setFormatter(StructuredLogFormatter())
else:
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger.addHandler(handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG system on startup."""
    logger.info("Starting Livestock Advisor API...")
    try:
        doc_count = initialize()
        logger.info(f"RAG system initialized with {doc_count} documents")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    yield
    logger.info("Shutting down Livestock Advisor API...")


app = FastAPI(
    title="Livestock Advisor API",
    description="RAG-powered chatbot for livestock breed information",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not GCP_PROJECT else None,  # Disable docs in production
    redoc_url="/redoc" if not GCP_PROJECT else None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(json.dumps({
        "httpRequest": {
            "requestMethod": request.method,
            "requestUrl": str(request.url),
            "status": response.status_code,
            "latency": f"{duration:.3f}s",
            "userAgent": request.headers.get("user-agent", ""),
            "remoteIp": request.client.host if request.client else ""
        }
    }))

    return response


# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = Field(default="default", max_length=100)
    user_id: str = Field(..., min_length=1, max_length=100)


class ChatResponse(BaseModel):
    response: str
    thread_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    database: Optional[dict] = None
    error: Optional[str] = None


class ThreadSummary(BaseModel):
    thread_id: str
    message_count: int
    preview: str
    updated_at: Optional[str] = None


class ThreadListResponse(BaseModel):
    threads: List[ThreadSummary]


class MessageItem(BaseModel):
    role: str
    content: str


class ThreadMessagesResponse(BaseModel):
    thread_id: str
    messages: List[MessageItem]


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."}
    )


# Chat UI HTML
CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Charlie</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --accent: #22c55e;
            --accent-hover: #16a34a;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: #475569;
            --sidebar-width: 280px;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
        }
        /* Sidebar */
        .sidebar {
            width: var(--sidebar-width);
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            height: 100vh;
            position: fixed;
            left: 0;
            top: 0;
            z-index: 100;
            transition: transform 0.3s ease;
        }
        .sidebar.collapsed {
            transform: translateX(calc(-1 * var(--sidebar-width)));
        }
        .sidebar-header {
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .sidebar-header h2 {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-secondary);
        }
        .new-chat-btn {
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 600;
            transition: background 0.2s;
        }
        .new-chat-btn:hover { background: var(--accent-hover); }
        .thread-list {
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem;
        }
        .thread-item {
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            margin-bottom: 0.25rem;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .thread-item:hover { background: var(--bg-tertiary); }
        .thread-item.active { background: var(--bg-tertiary); border-left: 3px solid var(--accent); }
        .thread-content {
            flex: 1;
            min-width: 0;
        }
        .thread-preview {
            font-size: 0.875rem;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .thread-meta {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }
        .thread-delete {
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 0.25rem;
            border-radius: 0.25rem;
            opacity: 0;
            transition: all 0.2s;
        }
        .thread-item:hover .thread-delete { opacity: 1; }
        .thread-delete:hover { color: #ef4444; background: rgba(239, 68, 68, 0.1); }
        .no-threads {
            text-align: center;
            padding: 2rem 1rem;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
        /* Main content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            margin-left: var(--sidebar-width);
            height: 100vh;
            transition: margin-left 0.3s ease;
        }
        .main-content.expanded {
            margin-left: 0;
        }
        .header {
            background: var(--bg-secondary);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .toggle-sidebar {
            background: none;
            border: none;
            color: var(--text-primary);
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .toggle-sidebar:hover { background: var(--bg-tertiary); }
        .toggle-sidebar svg { width: 1.25rem; height: 1.25rem; }
        .header h1 {
            font-size: 1.25rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .header h1::before { content: "🐄"; }
        .chat-container {
            flex: 1;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            padding: 1.5rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .welcome { text-align: center; padding: 3rem 1rem; color: var(--text-secondary); }
        .welcome h2 { color: var(--text-primary); margin-bottom: 0.5rem; font-size: 1.5rem; }
        .welcome p { margin-bottom: 1.5rem; }
        .suggestions { display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; }
        .suggestion {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.2s;
        }
        .suggestion:hover { border-color: var(--accent); color: var(--accent); }
        .message {
            max-width: 85%;
            padding: 1rem 1.25rem;
            border-radius: 1rem;
            line-height: 1.6;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            background: var(--accent);
            color: var(--bg-primary);
            margin-left: auto;
            border-bottom-right-radius: 0.25rem;
        }
        .message.assistant {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            margin-right: auto;
            border-bottom-left-radius: 0.25rem;
        }
        .message.assistant ul, .message.assistant ol { margin: 0.5rem 0; padding-left: 1.5rem; }
        .message.assistant li { margin: 0.25rem 0; }
        .message.assistant strong { color: var(--accent); }
        .typing { color: var(--text-secondary); font-style: italic; }
        .typing::after { content: ''; animation: dots 1.5s infinite; }
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }
        .input-area { background: var(--bg-secondary); padding: 1rem 1.5rem; border-top: 1px solid var(--border); }
        .input-wrapper { max-width: 900px; margin: 0 auto; display: flex; gap: 0.75rem; }
        #messageInput {
            flex: 1;
            padding: 0.875rem 1.25rem;
            border: 1px solid var(--border);
            border-radius: 1.5rem;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }
        #messageInput:focus { border-color: var(--accent); }
        #messageInput::placeholder { color: var(--text-secondary); }
        #sendBtn {
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            padding: 0.875rem 1.5rem;
            border-radius: 1.5rem;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        #sendBtn:hover { background: var(--accent-hover); }
        #sendBtn:disabled { background: var(--bg-tertiary); cursor: not-allowed; }
        #sendBtn svg { width: 1.25rem; height: 1.25rem; }
        /* Responsive */
        @media (max-width: 768px) {
            .sidebar { transform: translateX(calc(-1 * var(--sidebar-width))); }
            .sidebar.open { transform: translateX(0); }
            .main-content { margin-left: 0; }
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <aside class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <h2>Chat History</h2>
            <button class="new-chat-btn" onclick="newChat()">+ New</button>
        </div>
        <div class="thread-list" id="threadList">
            <div class="no-threads">No conversations yet</div>
        </div>
    </aside>

    <!-- Main Content -->
    <main class="main-content" id="mainContent">
        <div class="header">
            <button class="toggle-sidebar" onclick="toggleSidebar()">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/>
                </svg>
            </button>
            <h1>Charlie</h1>
        </div>
        <div class="chat-container" id="chatContainer">
            <div class="welcome" id="welcome">
                <h2>Welcome to Livestock Advisor</h2>
                <p>Ask me anything about livestock breeds, species, colors, and more!</p>
                <div class="suggestions">
                    <button class="suggestion" onclick="askSuggestion(this)">What cattle breeds are available?</button>
                    <button class="suggestion" onclick="askSuggestion(this)">Tell me about dairy goat breeds</button>
                    <button class="suggestion" onclick="askSuggestion(this)">What colors do sheep come in?</button>
                    <button class="suggestion" onclick="askSuggestion(this)">List poultry breeds for eggs</button>
                </div>
            </div>
        </div>
        <div class="input-area">
            <div class="input-wrapper">
                <input type="text" id="messageInput" placeholder="Ask about livestock breeds, species, colors..."
                       onkeydown="if(event.key==='Enter' && !event.shiftKey) sendMessage()">
                <button id="sendBtn" onclick="sendMessage()">
                    Send
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                    </svg>
                </button>
            </div>
        </div>
    </main>

    <script>
        // Get or create persistent user ID
        let userId = localStorage.getItem('charlie_user_id');
        if (!userId) {
            userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('charlie_user_id', userId);
        }
        let threadId = 'thread_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        let currentThreadId = null;

        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const threadList = document.getElementById('threadList');
        const sidebar = document.getElementById('sidebar');
        const mainContent = document.getElementById('mainContent');

        // Toggle sidebar
        function toggleSidebar() {
            sidebar.classList.toggle('collapsed');
            mainContent.classList.toggle('expanded');
        }

        // Load chat history threads
        async function loadThreads() {
            try {
                console.log('Loading threads for user:', userId);
                const response = await fetch(`/threads?user_id=${encodeURIComponent(userId)}`);
                if (!response.ok) {
                    const errorText = await response.text();
                    console.error('Failed to load threads:', response.status, errorText);
                    throw new Error('Failed to load threads');
                }
                const data = await response.json();
                console.log('Loaded threads:', data.threads);
                renderThreads(data.threads);
            } catch (error) {
                console.error('Error loading threads:', error);
            }
        }

        // Render threads in sidebar
        function renderThreads(threads) {
            if (threads.length === 0) {
                threadList.innerHTML = '<div class="no-threads">No conversations yet</div>';
                return;
            }
            threadList.innerHTML = threads.map(thread => `
                <div class="thread-item ${thread.thread_id === currentThreadId ? 'active' : ''}"
                     onclick="loadThread('${thread.thread_id}')">
                    <div class="thread-content">
                        <div class="thread-preview">${escapeHtml(thread.preview) || 'New conversation'}</div>
                        <div class="thread-meta">${thread.message_count} messages</div>
                    </div>
                    <button class="thread-delete" onclick="event.stopPropagation(); deleteThread('${thread.thread_id}')" title="Delete">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                        </svg>
                    </button>
                </div>
            `).join('');
        }

        // Load a specific thread
        async function loadThread(tid) {
            try {
                const response = await fetch(`/threads/${tid}/messages?user_id=${encodeURIComponent(userId)}`);
                if (!response.ok) throw new Error('Failed to load messages');
                const data = await response.json();

                threadId = tid;
                currentThreadId = tid;
                chatContainer.innerHTML = '';

                if (data.messages.length === 0) {
                    showWelcome();
                } else {
                    data.messages.forEach(msg => addMessage(msg.content, msg.role, false));
                }

                loadThreads(); // Refresh to update active state
                messageInput.focus();
            } catch (error) {
                console.error('Error loading thread:', error);
            }
        }

        // Delete a thread
        async function deleteThread(tid) {
            if (!confirm('Delete this conversation?')) return;
            try {
                const response = await fetch(`/threads/${tid}?user_id=${encodeURIComponent(userId)}`, {
                    method: 'DELETE'
                });
                if (!response.ok) throw new Error('Failed to delete thread');

                if (tid === currentThreadId) {
                    newChat();
                }
                loadThreads();
            } catch (error) {
                console.error('Error deleting thread:', error);
            }
        }

        // Start new chat
        function newChat() {
            threadId = 'thread_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            currentThreadId = null;
            showWelcome();
            loadThreads();
        }

        // Show welcome screen
        function showWelcome() {
            chatContainer.innerHTML = `
                <div class="welcome" id="welcome">
                    <h2>Welcome to Livestock Advisor</h2>
                    <p>Ask me anything about livestock breeds, species, colors, and more!</p>
                    <div class="suggestions">
                        <button class="suggestion" onclick="askSuggestion(this)">What cattle breeds are available?</button>
                        <button class="suggestion" onclick="askSuggestion(this)">Tell me about dairy goat breeds</button>
                        <button class="suggestion" onclick="askSuggestion(this)">What colors do sheep come in?</button>
                        <button class="suggestion" onclick="askSuggestion(this)">List poultry breeds for eggs</button>
                    </div>
                </div>`;
        }

        function askSuggestion(el) {
            messageInput.value = el.textContent;
            sendMessage();
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function addMessage(content, role, animate = true) {
            const welcomeEl = document.getElementById('welcome');
            if (welcomeEl) welcomeEl.remove();
            const div = document.createElement('div');
            div.className = 'message ' + role;
            if (!animate) div.style.animation = 'none';
            let formatted = content
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
                .replace(/^- (.+)$/gm, '<li>$1</li>')
                .replace(/^(\\d+)\\. (.+)$/gm, '<li>$2</li>')
                .replace(/\\n/g, '<br>');
            if (formatted.includes('<li>')) {
                formatted = formatted.replace(/(<li>.*<\\/li>)+/g, '<ul>$&</ul>');
            }
            div.innerHTML = formatted;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return div;
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            messageInput.value = '';
            sendBtn.disabled = true;
            addMessage(message, 'user');

            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant typing';
            typingDiv.textContent = 'Thinking';
            chatContainer.appendChild(typingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, thread_id: threadId, user_id: userId })
                });
                typingDiv.remove();
                if (!response.ok) throw new Error('Failed to get response');
                const data = await response.json();
                addMessage(data.response, 'assistant');
                currentThreadId = threadId;
                loadThreads(); // Refresh thread list
            } catch (error) {
                typingDiv.remove();
                addMessage('Sorry, something went wrong. Please try again.', 'assistant');
                console.error('Error:', error);
            }
            sendBtn.disabled = false;
            messageInput.focus();
        }

        // Initialize
        loadThreads();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI."""
    return CHAT_HTML


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a chat message."""
    logger.info(f"Chat request: user={request.user_id}, thread={request.thread_id}, message_length={len(request.message)}")

    try:
        response = chat(request.message, request.thread_id, request.user_id)
        logger.info(f"Chat response: user={request.user_id}, thread={request.thread_id}, response_length={len(response)}")
        return ChatResponse(response=response, thread_id=request.thread_id)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process message")


@app.get("/threads", response_model=ThreadListResponse)
async def get_threads(user_id: str):
    """Get list of chat threads for a user."""
    try:
        logger.info(f"Getting threads for user: {user_id}")
        threads = chat_store.get_user_threads(user_id)
        logger.info(f"Found {len(threads)} threads for user: {user_id}")

        result_threads = []
        for t in threads:
            updated_at = t.get("updated_at")
            # Handle different timestamp formats
            if updated_at:
                if hasattr(updated_at, 'isoformat'):
                    updated_at = updated_at.isoformat()
                elif hasattr(updated_at, 'timestamp'):
                    # Firestore Timestamp
                    updated_at = updated_at.isoformat() if hasattr(updated_at, 'isoformat') else str(updated_at)
                else:
                    updated_at = str(updated_at)

            result_threads.append(ThreadSummary(
                thread_id=t["thread_id"],
                message_count=t["message_count"],
                preview=t["preview"] or "",
                updated_at=updated_at
            ))

        return ThreadListResponse(threads=result_threads)
    except Exception as e:
        logger.error(f"Failed to get threads: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get threads")


@app.get("/threads/{thread_id}/messages", response_model=ThreadMessagesResponse)
async def get_thread_messages(thread_id: str, user_id: str):
    """Get messages for a specific thread."""
    try:
        messages = chat_store.get_messages(user_id, thread_id)
        return ThreadMessagesResponse(
            thread_id=thread_id,
            messages=[MessageItem(role=m["role"], content=m["content"]) for m in messages]
        )
    except Exception as e:
        logger.error(f"Failed to get thread messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get messages")


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str, user_id: str):
    """Delete a chat thread."""
    try:
        deleted = chat_store.delete_thread(user_id, thread_id)
        if deleted:
            return {"status": "deleted", "thread_id": thread_id}
        else:
            raise HTTPException(status_code=404, detail="Thread not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete thread: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete thread")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for Cloud Run."""
    try:
        summary = db.get_database_summary()
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            database=summary
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="2.0.0",
            error=str(e)
        )


@app.get("/ready")
async def readiness():
    """Readiness probe for Kubernetes/Cloud Run."""
    try:
        # Check database connection
        db.get_database_summary()
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
