# scripts/build_pinecone_index.py
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from app.config import (
    DATA_DIR, EMBEDDING_MODEL, EMBEDDING_DIM,
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENV
)
from app.data import load_documents, prepare_chunks

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(PINECONE_INDEX_NAME)
model = SentenceTransformer(EMBEDDING_MODEL)

raw_docs = load_documents(DATA_DIR)
documents = prepare_chunks(raw_docs)

vectors = []
for doc in documents:
    emb = model.encode(doc["text"]).tolist()
    vectors.append((doc["id"], emb, {"text": doc["text"]}))

index.upsert(vectors=vectors)
print("✅ Pinecone index built successfully")
