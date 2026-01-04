# app/retrieval.py
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from config import (
    EMBEDDING_MODEL, PINECONE_API_KEY, PINECONE_INDEX_NAME
)


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized = [doc["text"].split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query, k):
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]


class PineconeDenseRetriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX_NAME)

    def retrieve(self, query, k):
        q_emb = self.model.encode(query).tolist()
        res = self.index.query(vector=q_emb, top_k=k, include_metadata=True)

        return [
            ({"id": match["id"], "text": match["metadata"]["text"]}, match["score"])
            for match in res["matches"]
        ]


class HybridRetriever:
    def __init__(self, sparse, dense, alpha):
        self.sparse = sparse
        self.dense = dense
        self.alpha = alpha
        self.doc_map = {doc["id"]: doc for doc in sparse.documents}

    def retrieve(self, query, k):
        scores = {}

        for doc, s in self.sparse.retrieve(query, k):
            scores[doc["id"]] = scores.get(doc["id"], 0) + (1 - self.alpha) * s

        for doc, s in self.dense.retrieve(query, k):
            scores[doc["id"]] = scores.get(doc["id"], 0) + self.alpha * s
            self.doc_map.setdefault(doc["id"], doc)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.doc_map[i] for i, _ in ranked]
