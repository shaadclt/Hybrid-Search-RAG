# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel

from config import DATA_DIR, TOP_K, HYBRID_ALPHA
from data import load_documents, prepare_chunks
from retrieval import BM25Retriever, PineconeDenseRetriever, HybridRetriever
from pipeline import RAGPipeline

app = FastAPI(title="Hybrid Search RAG API")

print("🔹 Initializing system...")
raw_docs = load_documents(DATA_DIR)
documents = prepare_chunks(raw_docs)

bm25 = BM25Retriever(documents)
dense = PineconeDenseRetriever()
hybrid = HybridRetriever(bm25, dense, HYBRID_ALPHA)
pipeline = RAGPipeline(hybrid)
print("✅ System ready")


class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


@app.post("/query")
def query_rag(req: QueryRequest):
    answer = pipeline.run(req.query, req.top_k)
    return {
        "query": req.query,
        "answer": answer
    }
