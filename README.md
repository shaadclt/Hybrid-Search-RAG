# Hybrid Search RAG (Production-Ready)

This repository contains a **modular, production-ready Hybrid Search Retrieval-Augmented Generation (RAG) system** built using Python.  
The system combines **sparse retrieval (BM25)** and **dense retrieval (embeddings)** with **Llama 3 hosted on Groq** to deliver low-latency, high-relevance question answering.


## 🚀 Key Features

- **Hybrid Retrieval**
  - Sparse keyword-based retrieval using BM25
  - Dense semantic retrieval using sentence embeddings
  - Weighted fusion strategy for improved recall and robustness

- **Production-Grade Architecture**
  - Modular Python files with clear separation of concerns
  - Config-driven system design
  - Easy extensibility for reranking, evaluation, or APIs

- **Low-Latency Generation**
  - Llama 3 (`8B / 70B`) served via **Groq**
  - Fast inference suitable for interactive applications

- **Evaluation Ready**
  - Offline metrics such as Recall@K and MRR
  - Simple hooks to compare retrieval strategies


## 📁 Project Structure

```text
hybrid-search-rag/
│
├── app/
│   ├── config.py        # Central configuration & environment loading
│   ├── data.py          # Document loading and chunking
│   ├── retrieval.py     # BM25, dense, and hybrid retrievers
│   ├── generation.py    # Prompting and Llama 3 (Groq) integration
│   ├── pipeline.py      # End-to-end RAG orchestration
│   ├── evaluation.py   # Retrieval evaluation metrics
│   └── main.py          # CLI entry point
│
├── data/
│   └── documents/       # Input text documents
│
├── scripts/
│   └── run_eval.py      # Offline evaluation runner
│
├── requirements.txt
├── LICENSE.txt
├── .env.example
├── .gitignore
└── README.md
```
