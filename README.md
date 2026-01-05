# Hybrid Search RAG (Production-Ready)

This repository contains a **production-ready Hybrid Search Retrieval-Augmented Generation (RAG) system** built using Python, FastAPI, Pinecone, and Llama 3 (served via Groq).

The system combines **sparse keyword-based retrieval (BM25)** with **dense vector retrieval (Pinecone)** to deliver accurate, low-latency, and scalable question answering.

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

## 🧠 System Overview

1. **Documents** are loaded and chunked into passages

2. **BM25** retrieves keyword-relevant passages in-memory

3. **Pinecone** retrieves semantically relevant passages using embeddings

4. Retrieval scores are **fused via a hybrid strategy**

5. Top-ranked context is injected into a prompt

6. Llama 3 on Groq generates a grounded answer

This separation of **retrieval**, **ranking**, and **generation** makes the system easier to tune, debug, and scale.

## ⚙️ Setup
### 1. Clone the repository
```bash
git clone https://github.com/shaadclt/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

⚠️ Do not commit `.env` to version control.

### 🗄️ Build the Pinecone Index (One-Time)

Before running the system, build the vector index:
```bash
python scripts/build_pinecone_index.py
```

This step:

- Embeds document chunks

- Uploads vectors to Pinecone

- Stores text as metadata for retrieval

## ▶️ Running the Application
### Option 1: Run as an API (Recommended)
```bash
uvicorn app.api:app --reload
```

- Swagger UI: http://127.0.0.1:8000/docs

- Endpoint: `POST /query`

Example request:
```json
{
  "query": "What is hybrid search?",
  "top_k": 5
}
```

### Option 2: Run as a CLI
```bash
python app/main.py
```

Example:
```text
Query: Explain hybrid retrieval
Answer: ...
```

## 🔮 Extensibility

The system is designed to be easily extended with:

- Cross-encoder or LLM-based rerankers

- Streaming responses

- Authentication and rate limiting

- Vector store alternatives (FAISS, Chroma)

- Monitoring and feedback loops


## 🏆 Why This Project

This project demonstrates:

- Practical understanding of information retrieval systems

- Trade-offs between sparse and dense search

- Real-world RAG system design beyond notebooks

- Clean, production-style Python engineering

- API-first ML system deployment


## 📜 License

This project is licensed under the MIT License.
See the `LICENSE.txt` file for details.

