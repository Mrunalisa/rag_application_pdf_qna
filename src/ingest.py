import argparse
import sys
import time
from pathlib import Path
from . import config
from .embeddings import embed_texts, get_embedding_dim
from .pdf_utils import discover_pdfs, process_pdf
from .vector_store import ensure_collection, get_client, upsert_chunks

BATCH_SIZE = 16
MAX_RETRIES = 4
PAUSE_BETWEEN_BATCHES = 0.5

def run(recreate: bool = False) -> None:
    data_dir = Path(config.DATA_DIR)
    pdf_paths = discover_pdfs(data_dir)
    if not pdf_paths:
        print(f"No PDF files found in '{data_dir}/'. Add PDFs there and re-run.")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s): {[p.name for p in pdf_paths]}")

    all_chunks = []
    for pdf_path in pdf_paths:
        try:
            chunks = process_pdf(pdf_path, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        except Exception as exc:  # malformed / unreadable PDF
            print(f"  [WARN] Failed to parse {pdf_path.name}: {exc}")
            continue
        print(f"  {pdf_path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No text could be extracted from any PDF. Aborting.")
        sys.exit(1)

    print(f"Total chunks to index: {len(all_chunks)}")

    client = get_client()
    dim = get_embedding_dim()
    ensure_collection(client, dim, recreate=recreate)

    total = len(all_chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        vectors = embed_texts([c.text for c in batch])

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                upsert_chunks(client, batch, vectors)
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    print(f"  [FAIL] Batch at offset {i} failed after {MAX_RETRIES} attempts: {exc}")
                    raise
                wait = 2 ** attempt
                print(f"  [WARN] Batch at offset {i} failed ({exc}); retrying in {wait}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)

        print(f"  Indexed {min(i + BATCH_SIZE, total)}/{total}")
        time.sleep(PAUSE_BETWEEN_BATCHES)

    print("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant")
    parser.add_argument(
        "--recreate", action="store_true", help="Drop and rebuild the collection from scratch"
    )
    args = parser.parse_args()
    run(recreate=args.recreate)
