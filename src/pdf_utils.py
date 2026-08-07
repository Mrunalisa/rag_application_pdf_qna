from dataclasses import dataclass
from pathlib import Path
from typing import List
import fitz

@dataclass
class Chunk:
    doc_name: str
    page_number: int
    text: str
    chunk_index: int

def extract_pages(pdf_path: Path) -> List[str]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return pages

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    words = text.split(" ")
    avg_word_len = 6
    words_per_chunk = max(20, chunk_size // avg_word_len)
    overlap_words = max(0, overlap // avg_word_len)

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end == len(words):
            break
        start = end - overlap_words if end - overlap_words > start else end
    return chunks

def process_pdf(pdf_path: Path, chunk_size: int, overlap: int) -> List[Chunk]:
    doc_name = pdf_path.name
    result: List[Chunk] = []
    pages = extract_pages(pdf_path)
    for page_number, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text, chunk_size, overlap)
        for i, ch in enumerate(page_chunks):
            result.append(
                Chunk(doc_name=doc_name, page_number=page_number, text=ch, chunk_index=i)
            )
    return result

def discover_pdfs(data_dir: Path) -> List[Path]:
    return sorted(p for p in data_dir.glob("*.pdf") if p.is_file())
