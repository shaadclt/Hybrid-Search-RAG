# app/data.py
from pathlib import Path

def load_documents(data_dir: str):
    documents = []
    for file in Path(data_dir).glob("*.txt"):
        documents.append({
            "id": file.stem,
            "text": file.read_text(encoding="utf-8")
        })
    return documents


def chunk_text(text: str, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap

    return chunks


def prepare_chunks(raw_documents):
    documents = []
    for doc in raw_documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{doc['id']}_{i}",
                "text": chunk
            })
    return documents
