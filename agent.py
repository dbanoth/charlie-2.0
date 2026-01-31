"""LangGraph agent for the Agriculture RAG Chatbot."""
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from config import USE_VERTEX_AI, LLM_MODEL, GCP_PROJECT, GCP_LOCATION
from database import db
from rag import rag

# Initialize LLM based on configuration
if USE_VERTEX_AI:
    from langchain_google_vertexai import ChatVertexAI
    llm = ChatVertexAI(
        model=LLM_MODEL,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        temperature=0.3,
    )
    print(f"[Agent] Using Vertex AI LLM ({LLM_MODEL})")
else:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from config import GOOGLE_API_KEY
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    print(f"[Agent] Using Google AI LLM ({LLM_MODEL})")


class ChatState(TypedDict, total=False):
    """State for the chat agent."""
    messages: List[dict]
    user_input: str
    context: str
    response: str
    query_type: str  # "livestock" or "general"


# Classification prompt
CLASSIFICATION_PROMPT = """Classify if the following user question is related to livestock, agriculture, animal breeds, or farming.

Livestock-related topics include:
- Animal breeds (cattle, sheep, goats, pigs, poultry, horses, etc.)
- Species information, characteristics, terminology
- Colors, patterns, categories of animals
- Farming, breeding, animal husbandry
- Agricultural practices related to animals

Respond with ONLY one word: "livestock" if related, or "general" if not related.

User question: {question}

Classification:"""


# System prompt
SYSTEM_PROMPT = """You are an expert livestock advisor with access to a comprehensive database of animal breeds, species, colors, patterns, and categories.

Your knowledge includes:
- Detailed information about various livestock species (cattle, sheep, goats, pigs, poultry, horses, etc.)
- Breed characteristics, purposes (meat, milk, wool, eggs, working), and descriptions
- Available colors and patterns for each species
- Terminology (male/female terms, baby terms, etc.)

Guidelines:
1. Use the provided context from the database to answer questions accurately
2. Be helpful and informative about livestock breeds and species
3. If asked about something not in the context, say so honestly
4. Format responses clearly with bullet points or numbered lists when appropriate
5. Be conversational but professional

If the user asks about:
- Breeds: Provide breed names, species, purpose, and descriptions
- Species: Provide terminology, characteristics, and available breeds
- Colors/Patterns: List available options for the species
- General questions: Use your knowledge plus the database context
"""

# General assistant prompt (for non-livestock questions)
GENERAL_PROMPT = """You are a helpful, friendly assistant. Answer the user's question to the best of your ability.
Be conversational but concise. If you don't know something, say so honestly.
"""


def classify_query(state: ChatState) -> ChatState:
    """Classify if the query is livestock-related or general."""
    user_input = state.get("user_input", "")

    if not user_input:
        return {"query_type": "general"}

    prompt = CLASSIFICATION_PROMPT.format(question=user_input)
    response = llm.invoke(prompt)
    classification = response.content.strip().lower()

    # Default to livestock if classification is unclear
    if "livestock" in classification:
        query_type = "livestock"
    else:
        query_type = "general"

    print(f"[Agent] Query classified as: {query_type}")
    return {"query_type": query_type}


def route_query(state: ChatState) -> str:
    """Route to RAG or direct response based on classification."""
    query_type = state.get("query_type", "general")
    if query_type == "livestock":
        return "retrieve"
    else:
        return "generate_direct"


def retrieve_context(state: ChatState) -> ChatState:
    """Retrieve relevant context from RAG system."""
    user_input = state.get("user_input", "")

    if not user_input:
        return {"context": ""}

    context = rag.get_context_for_query(user_input)
    return {"context": context}


def generate_response(state: ChatState) -> ChatState:
    """Generate response using LLM with RAG context."""
    user_input = state.get("user_input", "")
    context = state.get("context", "")
    messages = state.get("messages", [])

    # Build the prompt
    prompt_parts = [SYSTEM_PROMPT]

    if context:
        prompt_parts.append(f"\n--- DATABASE CONTEXT ---\n{context}\n--- END CONTEXT ---\n")

    # Add chat history (last 10 messages)
    if messages:
        prompt_parts.append("\nRecent conversation:")
        for msg in messages[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}")

    prompt_parts.append(f"\nUser: {user_input}\n\nAssistant:")

    full_prompt = "\n".join(prompt_parts)

    # Generate response
    response = llm.invoke(full_prompt)
    response_text = response.content

    # Update messages
    new_messages = list(messages)
    new_messages.append({"role": "user", "content": user_input})
    new_messages.append({"role": "assistant", "content": response_text})

    return {
        "response": response_text,
        "messages": new_messages
    }


def generate_direct(state: ChatState) -> ChatState:
    """Generate response using LLM directly without RAG context."""
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    # Build the prompt
    prompt_parts = [GENERAL_PROMPT]

    # Add chat history (last 10 messages)
    if messages:
        prompt_parts.append("\nRecent conversation:")
        for msg in messages[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}")

    prompt_parts.append(f"\nUser: {user_input}\n\nAssistant:")

    full_prompt = "\n".join(prompt_parts)

    # Generate response
    response = llm.invoke(full_prompt)
    response_text = response.content

    # Update messages
    new_messages = list(messages)
    new_messages.append({"role": "user", "content": user_input})
    new_messages.append({"role": "assistant", "content": response_text})

    return {
        "response": response_text,
        "messages": new_messages
    }


def build_graph():
    """Build the LangGraph workflow with query classification and routing."""
    builder = StateGraph(ChatState)

    # Add nodes
    builder.add_node("classify", classify_query)
    builder.add_node("retrieve", retrieve_context)
    builder.add_node("generate", generate_response)
    builder.add_node("generate_direct", generate_direct)

    # Add edges with conditional routing
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_query,
        {
            "retrieve": "retrieve",
            "generate_direct": "generate_direct"
        }
    )
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("generate_direct", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Create the agent graph
agent = build_graph()


def chat(user_input: str, thread_id: str = "default") -> str:
    """
    Send a message to the chatbot and get a response.

    Args:
        user_input: The user's message
        thread_id: Session/thread identifier for conversation memory

    Returns:
        The agent's response
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Get current state to preserve message history
    current_state = agent.get_state(config)
    messages = current_state.values.get("messages", []) if current_state.values else []

    # Run the agent
    result = agent.invoke(
        {"user_input": user_input, "messages": messages},
        config
    )

    return result.get("response", "I apologize, but I couldn't generate a response.")


def initialize():
    """Initialize the RAG system by indexing the database."""
    print("[Agent] Initializing RAG system...")
    count = rag.index_database()
    try:
        summary = db.get_database_summary()
        print(f"[Agent] Ready! Database: {summary['total_species']} species, {summary['total_breeds']} breeds")
    except Exception:
        print("[Agent] Ready! (SQL database unavailable)")
    return count
