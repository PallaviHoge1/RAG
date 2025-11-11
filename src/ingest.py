import os, re, json
from pathlib import Path
import yaml
from typing import List, Dict

# Optional PyMuPDF (fast, best layout) -> fallback to pypdf (pure Python)
_USE_PYMUPDF = True
try:
    import fitz  # PyMuPDF
except Exception:
    _USE_PYMUPDF = False
    from pypdf import PdfReader

def load_config():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _extract_with_pymupdf(pdf_path: Path) -> List[Dict]:
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = re.sub(r"\s+\n", "\n", text).strip()
        chunks.append({
            "paper_id": pdf_path.stem,
            "page": page_num + 1,
            "section": "",
            "text": text
        })
    doc.close()
    return chunks

def _extract_with_pypdf(pdf_path: Path) -> List[Dict]:
    reader = PdfReader(str(pdf_path))
    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+\n", "\n", text).strip()
        chunks.append({
            "paper_id": pdf_path.stem,
            "page": i,
            "section": "",
            "text": text
        })
    return chunks

def extract_text_with_metadata(pdf_path: Path) -> List[Dict]:
    if _USE_PYMUPDF:
        try:
            return _extract_with_pymupdf(pdf_path)
        except Exception:
            # fallback if runtime DLLs are missing
            return _extract_with_pypdf(pdf_path)
    else:
        return _extract_with_pypdf(pdf_path)  # this calls pypdf version above

def split_into_chunks(pages: List[Dict], chunk_tokens=500, overlap=100) -> List[Dict]:
    out = []
    for p in pages:
        words = p["text"].split()
        step = max(1, chunk_tokens - overlap)
        i = 0
        while i < len(words):
            piece = " ".join(words[i:i+chunk_tokens])
            if piece.strip():
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
