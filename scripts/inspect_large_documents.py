import json
from pathlib import Path

CORPUS_FILE = Path("data/raw/n8n_full_corpus.json")


def main():

    with CORPUS_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    # Sort documents from largest to smallest.
    largest_documents = sorted(
        documents,
        key=lambda doc: len(doc.get("content") or ""),
        reverse=True
    )

    print("\nLARGEST DOCUMENTS")
    print("=" * 75)

    for index, doc in enumerate(largest_documents[:10], start=1):

        content = doc.get("content") or ""

        print(f"\n{index}. {doc.get('title')}")
        print(f"URL: {doc.get('url')}")
        print(f"Characters: {len(content):,}")
        print(f"Words: {len(content.split()):,}")
        print(f"Headings: {sum(1 for line in content.splitlines() if line.startswith('#'))}")
        print(f"Code blocks: {content.count('```') // 2}")

        print("\nCONTENT PREVIEW:")
        print(content[:700])

        print("\n" + "-" * 75)


if __name__ == "__main__":
    main()
