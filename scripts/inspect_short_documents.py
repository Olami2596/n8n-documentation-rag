import json
from pathlib import Path

CORPUS_FILE = Path("data/raw/n8n_full_corpus.json")


def main():
    with CORPUS_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    short_documents = [
        doc for doc in documents
        if len(doc.get("content") or "") < 150
    ]

    print("\nSHORT DOCUMENT INSPECTION")
    print("=" * 75)
    print(f"Documents found: {len(short_documents)}")

    for index, doc in enumerate(short_documents, start=1):
        content = doc.get("content") or ""

        print(f"\n{index}. {doc.get('title')}")
        print(f"URL: {doc.get('url')}")
        print(f"Category: {doc.get('category')}")
        print(f"Status: {doc.get('status')}")
        print(f"Characters: {len(content)}")
        print(f"Content: {content!r}")

        print("-" * 75)


if __name__ == "__main__":
    main()
