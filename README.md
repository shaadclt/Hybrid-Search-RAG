# Hybrid Search RAG (Production-Ready)

This repository contains a **production-ready Hybrid Search Retrieval-Augmented Generation (RAG) system** built using Python, FastAPI, Pinecone, and Llama 3 (served via Groq).

The system combines **sparse keyword-based retrieval (BM25)** with **dense vector retrieval (Pinecone)**to deliver accurate, low-latency, and scalable question answering.

The project is intentionally structured as a **modular Python codebase**, avoiding notebook-centric designs.


## 🚀 Key Features

- **Hybrid Retrieval**
  - Sparse retrieval using BM25 for exact keyword matching
  - Dense semantic retrieval using sentence embeddings stored in Pinecone
  - Weighted score fusion for improved recall and robustness

- **Production-Grade Architecture**
  - Modular Python files with clear separation of concerns
  - Config-driven design with environment-based secrets
  - Designed for extensibility (reranking, evaluation, APIs)

- **Low-Latency LLM Inference**
  - Llama 3 (`8B / 70B`) served via **Groq**
  - Fast response times suitable for real-time applications

- **API-First Design**
  - FastAPI-based REST service
  - Interactive Swagger documentation
  - CLI and API interfaces supported

## 📁 Project Structure

```text
hybrid-search-rag/
│
├── app/
│   ├── config.py        # Central configuration & environment loading
│   ├── api.py           # FastAPI application
│   ├── data.py          # Document loading and chunking
│   ├── retrieval.py     # BM25, dense, and hybrid retrievers
│   ├── generation.py    # Prompting and Llama 3 (Groq) integration
│   ├── pipeline.py      # End-to-end RAG orchestration
│   └── main.py          # CLI entry point
│
├── data/
│   └── documents/       # Input text documents
│
├── scripts/
│   └── build_pinecone_index.py  # One-time vector index builder
│
├── requirements.txt
├── LICENSE.txt
├── .env.example
├── .gitignore
└── README.md
```
