import sys
import textwrap

from . import config
from .llm import generate_answer
from .retriever import has_sufficient_evidence, retrieve


def format_sources(results) -> str:
    lines = []
    for r in results:
        snippet = r.payload["text"]
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "..."
        lines.append(
            f"  - {r.payload['doc_name']} | Page {r.payload['page_number']} "
            f"(score {r.score:.2f})\n"
            f"    \"{snippet}\""
        )
    return "\n".join(lines)


def answer_question(question: str) -> None:
    results = retrieve(question)

    if not has_sufficient_evidence(results):
        print(f"\nAnswer: {config.NOT_FOUND_MESSAGE}\n")
        return

    answer = generate_answer(question, results)

    if config.NOT_FOUND_MESSAGE.lower() in answer.lower():
        print(f"\nAnswer: {config.NOT_FOUND_MESSAGE}\n")
        return

    print("\nAnswer:")
    print(textwrap.fill(answer, width=100))
    print("\nSources:")
    print(format_sources(results))
    print()


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer_question(question)
        return

    print("RAG PDF Q&A — type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        try:
            answer_question(question)
        except Exception as exc:
            print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
