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

from agent import chat, initialize
from database import db
from config import GCP_PROJECT

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


class ChatResponse(BaseModel):
    response: str
    thread_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    database: Optional[dict] = None
    error: Optional[str] = None


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
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: var(--bg-secondary);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 {
            font-size: 1.25rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .header h1::before { content: "🐄"; }
        .new-chat {
            background: transparent;
            border: 1px solid var(--accent);
            color: var(--accent);
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.2s;
        }
        .new-chat:hover { background: var(--accent); color: var(--bg-primary); }
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
    </style>
</head>
<body>
    <div class="header">
        <h1>Charlie</h1>
        <button class="new-chat" onclick="newChat()">+ New Chat</button>
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
    <script>
        let threadId = 'thread_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');

        function newChat() {
            threadId = 'thread_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
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

        function addMessage(content, role) {
            const welcomeEl = document.getElementById('welcome');
            if (welcomeEl) welcomeEl.remove();
            const div = document.createElement('div');
            div.className = 'message ' + role;
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
                    body: JSON.stringify({ message, thread_id: threadId })
                });
                typingDiv.remove();
                if (!response.ok) throw new Error('Failed to get response');
                const data = await response.json();
                addMessage(data.response, 'assistant');
            } catch (error) {
                typingDiv.remove();
                addMessage('Sorry, something went wrong. Please try again.', 'assistant');
                console.error('Error:', error);
            }
            sendBtn.disabled = false;
            messageInput.focus();
        }
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
    logger.info(f"Chat request: thread={request.thread_id}, message_length={len(request.message)}")

    try:
        response = chat(request.message, request.thread_id)
        logger.info(f"Chat response: thread={request.thread_id}, response_length={len(response)}")
        return ChatResponse(response=response, thread_id=request.thread_id)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process message")


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
