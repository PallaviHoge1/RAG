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
