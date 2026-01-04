# app/retrieval.py
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized = [doc["text"].split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, k: int):
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:k]


class DenseRetriever:
    def __init__(self, documents, model_name: str):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [doc["text"] for doc in documents],
            normalize_embeddings=True
        )

    def retrieve(self, query: str, k: int):
        query_emb = self.model.encode(query, normalize_embeddings=True)
        scores = np.dot(self.embeddings, query_emb)

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:k]


class HybridRetriever:
    def __init__(self, sparse, dense, alpha: float):
        self.sparse = sparse
        self.dense = dense
        self.alpha = alpha
        self.doc_map = {doc["id"]: doc for doc in sparse.documents}

    def retrieve(self, query: str, k: int):
        scores = {}

        for doc, score in self.sparse.retrieve(query, k):
            scores[doc["id"]] = scores.get(doc["id"], 0) + (1 - self.alpha) * score

        for doc, score in self.dense.retrieve(query, k):
            scores[doc["id"]] = scores.get(doc["id"], 0) + self.alpha * score

        ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.doc_map[doc_id] for doc_id, _ in ranked_ids]
