# PDF RAG Q&A

A simple Retrieval-Augmented Generation app that answers questions from a set
of PDF documents, with mandatory citations (document, page, snippet).

## Architecture

```
PDFs (data/) --> PyMuPDF page extraction --> word-window chunking (per page)
             --> sentence-transformers embeddings --> Qdrant (cosine, upsert)

Question --> embed query --> Qdrant top-k search --> similarity floor check
         --> if evidence weak: "not available" (no LLM call)
         --> else: OpenRouter free LLM (context-only prompt) --> answer
         --> print answer + sources (doc name, page, snippet) from the same
             chunks that were fed to the LLM
```

Chunks never span two pages, so every citation's page number is exact. A
minimum cosine-similarity threshold (`MIN_SCORE`) is checked before calling
the LLM; if nothing relevant is found, the app returns the "not available"
message directly instead of asking the model to guess. The LLM is also
instructed via system prompt to output that same message if the retrieved
context is insufficient, as a second safety net against fabrication.

## Libraries used

- **PyMuPDF (fitz)** — PDF text extraction, page by page
- **sentence-transformers** — local embedding model (free, no API key)
- **qdrant-client** — vector storage/search (local Docker or Qdrant Cloud free tier)
- **requests** — calls the OpenRouter chat-completions API directly
- **python-dotenv** — loads config from `.env`

## Embedding model

`sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs locally on CPU).
OpenRouter only routes LLM chat completions, not embeddings, so a local
free model is used to keep the whole pipeline cost-free and keep document
content from being sent to a third-party embedding API.

## LLM

Any **free** OpenRouter model works; default is
`meta-llama/llama-3.1-8b-instruct:free`. Change via `OPENROUTER_MODEL` in `.env`.

## Assumptions

- PDFs are text-based (not scanned images) — no OCR step is included.
- "Not available" is decided by a similarity-score floor **and** an explicit
  LLM instruction, whichever is stricter.
- Answers are generated strictly from retrieved chunks; the LLM is told not
  to use outside knowledge.
- Single flat `data/` folder of PDFs; no sub-folder recursion.

## How to run

1. **Start Qdrant** (local Docker):
   ```bash
   docker compose up -d
   ```
   (Or set `QDRANT_URL` / `QDRANT_API_KEY` in `.env` to a Qdrant Cloud free instance instead.)

2. **Create and activate a virtual environment**:

   **macOS / Linux (bash/zsh):**
```bash
   python3 -m venv .venv
   source .venv/bin/activate
```

   **Windows PowerShell:**
```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
```
   If PowerShell blocks the script with an execution-policy error, run this once first:
```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

   **Windows cmd.exe:**
```cmd
   python -m venv .venv
   .venv\Scripts\activate.bat
```

   **Git Bash on Windows:**
```bash
   python -m venv .venv
   source .venv/Scripts/activate
```

   Then install dependencies (same command for all shells, venv active):
```bash
   pip install -r requirements.txt
```

3. **Configure**:
   ```bash
   cp .env.example .env
   # edit .env and set OPENROUTER_API_KEY (get one free at https://openrouter.ai)
   ```

4. **Add PDFs**: place your PDF files into `data/`.

5. **Ingest**:
   ```bash
   python -m src.ingest          # add/update
   python -m src.ingest --recreate   # wipe and rebuild the collection
   ```

6. **Ask questions**:
   ```bash
   python -m src.app                     # interactive loop
   python -m src.app "What is the leave policy?"   # single question
   ```

Example output:

```
Answer: Employees are entitled to 24 days of annual leave per calendar year.

Sources:
  - employee_handbook.pdf | Page 17 (score 0.81)
    "All full-time employees receive 24 days of annual leave..."
```
