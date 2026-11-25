import json
from pathlib import Path

CORPUS_FILE = Path("data/raw/n8n_full_corpus.json")


def main():
    with CORPUS_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    keywords = ["execute workflow", "execution", "workflow execution", "run workflow"]

    print("\nAPI EXECUTION DOCUMENTATION SEARCH")
    print("=" * 75)

    matches = []

    for document in documents:
        title = (document.get("title") or "").lower()
        url = (document.get("url") or "").lower()
        content = (document.get("content") or "").lower()

        if "n8n-api" in url and any(
            keyword in title or keyword in content for keyword in keywords
        ):
            matches.append(document)

    print(f"Matching API documents: {len(matches)}")

    for index, document in enumerate(matches, start=1):
        print(f"\n{index}. {document.get('title')}")
        print(f"URL: {document.get('url')}")
        print(f"Category: {document.get('category')}")

        content = document.get("content") or ""

        for keyword in keywords:
            position = content.lower().find(keyword)

            if position != -1:
                start = max(0, position - 200)
                end = min(len(content), position + 500)

                print(f"\nMatch: {keyword}")
                print(content[start:end])
                break

        print("-" * 75)


if __name__ == "__main__":
    main()
