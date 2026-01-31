"""Configuration settings for the Agriculture RAG Chatbot."""
import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "").strip(),
    "port": int(os.getenv("DB_PORT", "1433").strip()),
    "user": os.getenv("DB_USER", "").strip(),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "").strip(),
}

# Allowed tables for security
ALLOWED_TABLES = [
    "Speciesavailable",
    "Speciesbreedlookuptable",
    "Speciescategory",
    "Speciescolorlookuptable",
    "Speciespatternlookuptable",
    "Speciesregistrationtypelookuptable",
]

# GCP Configuration
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
GCP_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

# Vertex AI Vector Search Configuration
VECTOR_SEARCH_INDEX_ENDPOINT = os.getenv("VECTOR_SEARCH_INDEX_ENDPOINT", "").strip()
VECTOR_SEARCH_INDEX_ID = os.getenv("VECTOR_SEARCH_INDEX_ID", "").strip()
VECTOR_SEARCH_DEPLOYED_INDEX_ID = os.getenv("VECTOR_SEARCH_DEPLOYED_INDEX_ID", "").strip()
GCS_BUCKET = os.getenv("GCS_BUCKET", "").strip()

# Determine if using Vertex AI
USE_VERTEX_AI = bool(GCP_PROJECT)

# LLM Configuration
if USE_VERTEX_AI:
    LLM_MODEL = os.getenv("VERTEX_AI_MODEL", "gemini-2.0-flash-001")
else:
    LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# RAG Configuration
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSIONS = 768
TOP_K_RESULTS = 10

# Firestore Configuration
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "(default)").strip()
FIRESTORE_COLLECTION = "livestock_knowledge"

# Print configuration on import
if USE_VERTEX_AI:
    print(f"[Config] Using Vertex AI (Project: {GCP_PROJECT}, Location: {GCP_LOCATION})")
else:
    print("[Config] Using Google AI API")
