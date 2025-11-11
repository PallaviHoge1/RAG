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
