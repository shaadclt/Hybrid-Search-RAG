# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = "data/documents"

# Retrieval
TOP_K = 5
HYBRID_ALPHA = 0.6

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# LLM (Groq)
LLM_MODEL = "llama3-70b-8192"
TEMPERATURE = 0.2
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hybrid-search-rag"
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
