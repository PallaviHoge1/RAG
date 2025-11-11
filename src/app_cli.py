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
