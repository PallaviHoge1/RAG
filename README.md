## 🧠 RAG-QA on AI Research Papers (Local & Offline)

**Description:**
This project implements a **Retrieval-Augmented Generation (RAG)** system for **question answering on AI research papers**, fully offline and privacy-friendly.
It allows users to ask natural-language questions and get **accurate, context-aware answers** directly sourced from a small collection of research PDFs such as *Transformers*, *RAG*, and *GPT-3* papers.

The system is designed to run entirely **on local hardware (CPU/iGPU)** with **no API keys or paid LLM access** — powered by **Ollama**, **FAISS**, and **Sentence-Transformers**.

---

### 🚀 Features

* 📄 **PDF ingestion & preprocessing:** Extracts and cleans text from AI research papers.
* 🧩 **Text chunking & vectorization:** Splits documents into semantically meaningful segments and embeds them locally.
* 🔍 **Efficient retrieval:** Uses FAISS for vector similarity search and optional reranking via Sentence-Transformers.
* 🤖 **Local LLM generation:** Generates grounded answers using `llama3.2:3b-instruct` (Ollama).
* 📚 **Source attribution:** Each answer includes citations from the original papers for transparency.
* 💻 **Runs 100% offline:** No OpenAI, API keys, or external services required.

---

### 🧩 Tech Stack

* **LLM:** `llama3.2:3b-instruct` (via Ollama)
* **Embeddings:** `nomic-embed-text` or `bge-small-en-v1.5`
* **Retriever:** FAISS (vector index)
* **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU-friendly)
* **Frameworks:** Python, LangChain, PyMuPDF, Sentence-Transformers

---

### 📊 Example Questions

1. What are the main components of a RAG model and how do they interact?
2. What are the two sub-layers in each encoder layer of the Transformer model?
3. Explain how positional encoding is implemented in Transformers and why it is necessary.
4. Describe the concept of multi-head attention and its benefits.
5. What is few-shot learning and how does GPT-3 implement it during inference?

---

### 🏗️ Project Structure

```
rag_qa/
├── data/                # PDF files (research papers)
├── index/               # Vector index storage
├── src/
│   ├── ingest.py        # PDF loading, cleaning, chunking
│   ├── embed_index.py   # Embedding + FAISS index creation
│   ├── retrieve.py      # Search & rerank
│   ├── generate.py      # Answer generation using Ollama
│   └── app_cli.py       # CLI interface for asking questions
├── config.yaml          # Model paths, chunk size, and top-k settings
└── README.md
```

---

### ⚡ How It Works

1. Load and preprocess 3–4 research papers (e.g., Transformer, RAG, GPT-3).
2. Split text into overlapping chunks and embed them using local models.
3. Store embeddings in a FAISS vector index.
4. Retrieve top-k relevant passages for each user query.
5. Generate grounded answers via Llama-3.2 and show citation references.

---

### 🧰 Installation

```bash
pip install pymupdf faiss-cpu sentence-transformers langchain-core
ollama pull llama3.2:3b-instruct
ollama pull nomic-embed-text
```

---

### 💡 Future Enhancements

* Web UI with Streamlit or Gradio
* Evaluation metrics for retrieval and answer quality
* Multi-document reranking
* Citation highlighting in generated answers

---