# app/main.py
from config import DATA_DIR, TOP_K, HYBRID_ALPHA, EMBEDDING_MODEL
from data import load_documents, prepare_chunks
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from pipeline import RAGPipeline


def main():
    print("🔹 Loading documents...")
    raw_docs = load_documents(DATA_DIR)
    documents = prepare_chunks(raw_docs)

    print("🔹 Building retrievers...")
    bm25 = BM25Retriever(documents)
    dense = DenseRetriever(documents, EMBEDDING_MODEL)
    hybrid = HybridRetriever(bm25, dense, HYBRID_ALPHA)

    pipeline = RAGPipeline(hybrid)

    print("✅ Hybrid Search RAG ready!")
    while True:
        query = input("\nQuery (type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = pipeline.run(query, TOP_K)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
