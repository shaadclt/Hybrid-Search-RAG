# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # 👈 loads .env automatically

# Paths
DATA_DIR = "data/documents"

# Retrieval
TOP_K = 5
HYBRID_ALPHA = 0.6

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM (Groq)
LLM_MODEL = "llama3-70b-8192"
TEMPERATURE = 0.2

# Secrets (optional explicit access)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
