import os
from dotenv import load_dotenv

# ========================
# LOAD ENV BASED ON ENV TYPE
# ========================
ENV = os.getenv("ENV", "dev")  # default = dev

if ENV == "prod":
    load_dotenv(".env.prod")
else:
    load_dotenv(".env")  # default dev

# ========================
# API KEYS
# ========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# ========================
# MODEL CONFIG
# ========================
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3-8b-8192"
)

# ========================
# CHUNKING CONFIG (with validation)
# ========================
def get_int_env(key, default):
    value = os.getenv(key, default)
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{key} must be an integer")


CHUNK_SIZE = get_int_env("CHUNK_SIZE", 500)
CHUNK_OVERLAP = get_int_env("CHUNK_OVERLAP", 100)

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

# ========================
# RETRIEVAL CONFIG
# ========================
TOP_K = get_int_env("TOP_K", 3)

if TOP_K <= 0:
    raise ValueError("TOP_K must be greater than 0")

# ========================
# PATH CONFIG
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

# ========================
# DEBUG FUNCTION
# ========================
def print_config():
    print("=== CONFIGURATION ===")
    print(f"ENV: {ENV}")
    print(f"MODEL: {LLM_MODEL}")
    print(f"EMBEDDING: {EMBEDDING_MODEL}")
    print(f"CHUNK_SIZE: {CHUNK_SIZE}, OVERLAP: {CHUNK_OVERLAP}")
    print(f"TOP_K: {TOP_K}")
    print(f"UPLOAD_DIR: {UPLOAD_DIR}")
























#  Dependencies (IMPORTANT — Tell Interviewer)
# pip install python-dotenv
# Example FOR .env File
# GROQ_API_KEY=your_key_here
# CHUNK_SIZE=500
# CHUNK_OVERLAP=100
# TOP_K=3
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# LLM_MODEL=llama3-8b-8192







#  Fixed getenv bug
#  Added env-based config (dev/prod)
#  Added type validation
#  Added logical validation
#  Made models dynamic
#  Added directory safety
#  Kept code simple (no over-engineering)

# Alternatives:
#  Pydantic BaseSettings
#  YAML config files
#  Config server (AWS/GCP)

# Trade-off:
#  Manual validation instead of schema-based




