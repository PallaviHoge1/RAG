#!/usr/bin/env bash
set -euo pipefail

# Root folder (change if you want a different name)
PROJECT_NAME="rag_qa"

echo "▶ Creating project structure: ${PROJECT_NAME}"
mkdir -p "${PROJECT_NAME}"/{data,index,src,tests}
touch "${PROJECT_NAME}/data/.gitkeep" "${PROJECT_NAME}/index/.gitkeep"

# .gitignore
cat > "${PROJECT_NAME}/.gitignore" <<'EOF'
# Python
__pycache__/
*.pyc
.env
.venv/
venv/
# Indexes & cache
index/
.cache/
# OS
.DS_Store
Thumbs.db
EOF

# Config
cat > "${PROJECT_NAME}/config.yaml" <<'EOF'
# ==== RAG Project Config (offline, local) ====
paths:
  data_dir: "./data"
  index_dir: "./index"

models:
  llm: "llama3.2:3b"             # via Ollama (already installed)
  embedding: "nomic-embed-text"  # via Ollama (run: ollama pull nomic-embed-text)
  reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"  # sentence-transformers (CPU)

chunking:
  chunk_tokens: 500
  chunk_overlap: 100

retrieval:
  top_k: 8           # initial vector search
  rerank_top_k: 4    # keep best N after rerank

generation:
  max_context_tokens: 3000
  answer_instructions: |
    You are a careful research assistant. Answer ONLY from the provided context.
    If the answer is not present, say "Not found in the provided papers."
    Include concise citations like [Paper, Section/Page].

EOF

# src/ingest.py
cat > "${PROJECT_NAME}/src/ingest.py" <<'EOF'
import os, re, json
from pathlib import Path
import fitz  # PyMuPDF
import yaml
from typing import List, Dict

def load_config():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_text_with_metadata(pdf_path: Path) -> List[Dict]:
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        # basic cleanup
        text = re.sub(r'\s+\n', '\n', text).strip()
        chunks.append({
            "paper_id": pdf_path.stem,
            "page": page_num + 1,
            "section": "",  # optional: infer from headings if needed
            "text": text
        })
    doc.close()
    return chunks

def token_estimate(s: str) -> int:
    # rough token estimate for chunking without tokenizer deps
    return max(1, len(s.split()) // 0.75)

def split_into_chunks(pages: List[Dict], chunk_tokens=500, overlap=100) -> List[Dict]:
    out = []
    for p in pages:
        words = p["text"].split()
        step = chunk_tokens - overlap
        i = 0
        while i < len(words):
            piece = " ".join(words[i:i+chunk_tokens])
            out.append({
                "paper_id": p["paper_id"],
                "page": p["page"],
                "section": p.get("section",""),
                "text": piece
            })
            i += step
    return out

def main():
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    assert data_dir.exists(), f"Data dir not found: {data_dir}"
    all_chunks = []
    for pdf in data_dir.glob("*.pdf"):
        pages = extract_text_with_metadata(pdf)
        chunks = split_into_chunks(pages, cfg["chunking"]["chunk_tokens"], cfg["chunking"]["chunk_overlap"])
        all_chunks.extend(chunks)
    out_path = Path(__file__).resolve().parents[1] / "index" / "chunks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks to {out_path}")

if __name__ == "__main__":
    main()
EOF

# src/embed_index.py
cat > "${PROJECT_NAME}/src/embed_index.py" <<'EOF'
import os, json
from pathlib import Path
import faiss
import numpy as np
import yaml
from tqdm import tqdm
from ollama import Client as OllamaClient

def load_config():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_jsonl(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def embed_texts_ollama(texts, model="nomic-embed-text", host="http://localhost:11434"):
    client = OllamaClient(host=host)
    vecs = []
    for t in tqdm(texts, desc="Embedding"):
        res = client.embeddings(model=model, prompt=t)
        vecs.append(np.array(res["embedding"], dtype="float32"))
    return np.vstack(vecs)

def main():
    cfg = load_config()
    chunks_path = Path(__file__).resolve().parents[1] / "index" / "chunks.jsonl"
    assert chunks_path.exists(), "Run ingest.py first to create chunks.jsonl"

    meta = []
    texts = []
    for rec in read_jsonl(chunks_path):
        meta.append(rec)
        texts.append(rec["text"])

    print(f"Total chunks: {len(texts)}")
    embs = embed_texts_ollama(texts, model=cfg["models"]["embedding"])

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via inner product on normalized vectors
    # normalize
    faiss.normalize_L2(embs)
    index.add(embs)

    index_dir = Path(cfg["paths"]["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Saved index to {index_dir}")

if __name__ == "__main__":
    main()
EOF

# src/retrieve.py
cat > "${PROJECT_NAME}/src/retrieve.py" <<'EOF'
import faiss, json
from pathlib import Path
import numpy as np
import yaml
from sentence_transformers import CrossEncoder
from ollama import Client as OllamaClient

def load_config():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_index(index_path: Path):
    index = faiss.read_index(str(index_path))
    return index

def load_meta(meta_path: Path):
    return [json.loads(l) for l in open(meta_path, "r", encoding="utf-8")]

def embed_query_ollama(q: str, model="nomic-embed-text"):
    client = OllamaClient(host="http://localhost:11434")
    emb = np.array(client.embeddings(model=model, prompt=q)["embedding"], dtype="float32")
    emb = emb / np.linalg.norm(emb)
    return emb

def retrieve(query: str, top_k: int, cfg):
    index = load_index(Path(cfg["paths"]["index_dir"]) / "faiss.index")
    meta = load_meta(Path(cfg["paths"]["index_dir"]) / "meta.jsonl")
    qv = embed_query_ollama(query, cfg["models"]["embedding"]).reshape(1, -1)
    D, I = index.search(qv, top_k)
    results = [(meta[i], float(D[0][rank])) for rank, i in enumerate(I[0])]
    return results

def rerank(query: str, results, keep_n: int, model_name: str):
    # results: list of (meta, score)
    passages = [r[0]["text"] for r in results]
    pairs = [(query, p) for p in passages]
    ce = CrossEncoder(model_name)
    scores = ce.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:keep_n]
    return [r for (r, s) in ranked]

if __name__ == "__main__":
    cfg = load_config()
    q = "What are the two sub-layers in each encoder layer of the Transformer model?"
    initial = retrieve(q, cfg["retrieval"]["top_k"], cfg)
    final = rerank(q, initial, cfg["retrieval"]["rerank_top_k"], cfg["models"]["reranker"])
    for rec, score in final:
        m = rec
        print(f"[{m['paper_id']} p.{m['page']}] score={score:.3f}\n{m['text'][:300]}...\n")
EOF

# src/generate.py
cat > "${PROJECT_NAME}/src/generate.py" <<'EOF'
from pathlib import Path
import yaml
from ollama import Client as OllamaClient

SYS_PROMPT = """You are a precise research assistant.
Answer ONLY from the given context. If not in context, say "Not found in the provided papers."
Always include citation tags like [Paper, Page] where appropriate.
Be concise and factual.
"""

def load_config():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def format_context(chunks):
    out = []
    for m, _ in chunks:
        tag = f"[{m['paper_id']} p.{m['page']}]"
        out.append(f"{tag}\n{m['text']}")
    return "\n\n---\n\n".join(out)

def call_llm(prompt, model="llama3.2:3b"):
    client = OllamaClient(host="http://localhost:11434")
    res = client.chat(model=model, messages=[
        {"role":"system","content": SYS_PROMPT},
        {"role":"user","content": prompt}
    ])
    return res["message"]["content"]

def generate_answer(question: str, top_chunks):
    context = format_context(top_chunks)
    full_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    cfg = load_config()
    return call_llm(full_prompt, cfg["models"]["llm"])

if __name__ == "__main__":
    print("This module is used by app_cli.py")
EOF

# src/app_cli.py
cat > "${PROJECT_NAME}/src/app_cli.py" <<'EOF'
import yaml
from pathlib import Path
from retrieve import retrieve, rerank, load_config
from generate import generate_answer

def main():
    cfg = load_config()
    print("RAG QA (Local & Offline). Type your question, or 'exit' to quit.")
    while True:
        q = input("\n> ")
        if not q or q.lower().strip() in {"exit","quit"}:
            break
        hits = retrieve(q, cfg["retrieval"]["top_k"], cfg)
        hits = rerank(q, hits, cfg["retrieval"]["rerank_top_k"], cfg["models"]["reranker"])
        answer = generate_answer(q, hits)
        print("\n=== Answer ===")
        print(answer)

if __name__ == "__main__":
    main()
EOF

echo "✅ Done. Next steps:
1) cd ${PROJECT_NAME}
2) python -m venv .venv && source .venv/bin/activate    # Windows (Git Bash): source .venv/Scripts/activate
3) pip install -r requirements.txt
4) ollama pull nomic-embed-text
5) Put your PDFs in ./data
6) python src/ingest.py && python src/embed_index.py
7) python src/app_cli.py
"
