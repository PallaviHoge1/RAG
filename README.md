# RAG-QA on AI Research Papers (Offline)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for question answering on AI research papers. It allows users to ask questions and receive accurate, context-based answers directly sourced from a collection of academic PDFs such as "Attention Is All You Need", "Retrieval-Augmented Generation", and "Language Models are Few-Shot Learners (GPT-3)".

The entire system runs fully offline, without using any API keys or paid LLM services, making it suitable for local machines and privacy-preserving research workflows.

## Key Features

* PDF preprocessing: automatically extracts and cleans text from research papers.
* Chunking & embedding: splits documents into overlapping chunks and generates local embeddings.
* Efficient retrieval: uses FAISS for semantic vector search and a cross-encoder for reranking.
* Local answer generation: employs the `llama3.2:3b` model from Ollama to generate concise, grounded answers.
* Source attribution: each generated answer includes references to the original paper sections or pages.
* Runs 100% offline: no cloud dependency or external API usage.

## Tech Stack

| Component       | Tool / Library                         | Purpose                                                |
| --------------- | -------------------------------------- | ------------------------------------------------------ |
| LLM             | `llama3.2:3b` (Ollama)                 | Generates final grounded answers                       |
| Embeddings      | `nomic-embed-text` (Ollama)            | Converts text chunks into dense vectors                |
| Retriever       | FAISS                                  | Finds semantically similar document chunks             |
| Reranker        | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranks top retrieved results based on query relevance |
| Text extraction | PyMuPDF / PyPDF                        | Reads and cleans text from PDFs                        |
| Environment     | Python 3.11 (16 GB RAM, CPU/iGPU)      | Offline environment setup                              |

## Project Structure

```
rag_qa/
├── data/                # Folder for PDF research papers
├── index/               # Stores vector index and metadata
├── src/
│   ├── ingest.py        # Extracts and chunks PDF text
│   ├── embed_index.py   # Generates embeddings and builds FAISS index
│   ├── retrieve.py      # Retrieves and reranks relevant passages
│   ├── generate.py      # Generates answers using local LLM
│   └── app_cli.py       # CLI interface for querying the system
├── config.yaml          # Configuration file for models and parameters
├── requirements.txt     # Dependencies
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag_qa.git
cd rag_qa
```

### 2. Create virtual environment

```bash
python -m venv .venv
# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Activate on Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull required Ollama models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Configuration

Edit the `config.yaml` file to adjust parameters such as chunk size, overlap, top_k, and rerank model.

Example:

```yaml
models:
  llm: "llama3.2:3b"
  embedding: "nomic-embed-text"
  reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"

chunking:
  chunk_tokens: 500
  chunk_overlap: 100

retrieval:
  top_k: 8
  rerank_top_k: 4
```

## Usage

### Step 1: Add research papers

Place your PDF research papers inside the `data/` folder.

Example files:

```
data/
├── 1706.03762v7.pdf       # Attention Is All You Need
├── 2005.11401v4.pdf       # Retrieval-Augmented Generation
├── 2005.14165v4.pdf       # GPT-3 Few-Shot Learning
```

### Step 2: Preprocess PDFs

Extract and chunk text:

```bash
python src/ingest.py
```

This will generate `index/chunks.jsonl`.

### Step 3: Embed and build index

Generate embeddings and build a FAISS vector index:

```bash
python src/embed_index.py
```

The embeddings and index files will be stored in `index/`.

### Step 4: Run the question answering CLI

Launch the interactive CLI:

```bash
python src/app_cli.py
```

Example interaction:

```
RAG QA (Local & Offline). Type your question, or 'exit' to quit.
> What are the two sub-layers in each encoder layer of the Transformer model?
=== Answer ===
Each encoder layer consists of a multi-head self-attention mechanism followed by a feed-forward neural network. [Transformer p.5]
```

## Example Questions

1. What are the main components of a RAG model and how do they interact?
2. What are the two sub-layers in each encoder layer of the Transformer model?
3. Explain how positional encoding is implemented in Transformers and why it is necessary.
4. Describe the concept of multi-head attention and its benefits.
5. What is few-shot learning and how does GPT-3 implement it during inference?

## Evaluation

Evaluation can be done by:

* Comparing generated answers with the original paper text.
* Checking citation accuracy (paper name and page reference).
* Measuring retrieval quality using similarity scores.

## Future Enhancements

* Add Streamlit/Gradio web interface for better interactivity.
* Integrate OCR (Tesseract) for scanned PDFs.
* Include answer confidence scoring.
* Optimize retrieval pipeline with hybrid search (semantic + keyword).
* Support more embedding models for experimentation.

## System Requirements

| Component      | Minimum Requirement          |
| -------------- | ---------------------------- |
| RAM            | 8–16 GB                      |
| Disk Space     | ~4 GB for models and indexes |
| CPU            | Intel i5 or equivalent       |
| GPU (optional) | Integrated GPU works fine    |
| OS             | Windows / Linux / macOS      |
| Ollama         | v0.3.0 or later              |

## Credits

* Meta (Llama 3.2) for open-source LLMs.
* Research papers used as data sources.
* Sentence-Transformers team for reranker models.
* PyMuPDF & FAISS contributors for PDF processing and vector indexing.