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
