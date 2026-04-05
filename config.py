import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API KEYS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# MODEL CONFIG
# ========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192"


# CHUNKING CONFIG
# ========================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# RETRIEVAL CONFIG

TOP_K = 3

# PATHS
# ========================
UPLOAD_DIR = "data/uploads"
FAISS_INDEX_PATH = "vectorstore/faiss_index"
