import requests
from . import config

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer the user's question using ONLY the "
    "information in the provided context excerpts. Each excerpt is labeled with a "
    "source number.\n"
    "Rules:\n"
    "- If the context does not contain enough information to answer, respond with "
    "exactly this sentence and nothing else: "
    f"\"{config.NOT_FOUND_MESSAGE}\"\n"
    "- Do not use outside knowledge and do not guess.\n"
    "- Keep the answer concise and directly grounded in the excerpts.\n"
    "- Do not fabricate facts, numbers, or policies not present in the context."
)

def build_context_block(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Source {i} | {c.payload['doc_name']} p.{c.payload['page_number']}]\n{c.payload['text']}")
    return "\n\n".join(parts)

def generate_answer(question: str, chunks) -> str:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    context_block = build_context_block(chunks)
    user_prompt = f"Context excerpts:\n\n{context_block}\n\nQuestion: {question}"

    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(config.OPENROUTER_BASE_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
