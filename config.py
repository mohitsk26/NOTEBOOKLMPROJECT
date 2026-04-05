import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API KEYS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# MODEL CONFIG
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192"

# CHUNKING CONFIG

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# RETRIEVAL CONFIG

TOP_K = 3

# PATH CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")
