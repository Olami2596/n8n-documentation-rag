import json
from pathlib import Path

from ingestion.chunker import chunk_documents


CORPUS_FILE = Path("data/raw/n8n_full_corpus.json")
OUTPUT_FILE = Path("data/processed/n8n_full_chunks.json")


def main():
    print("Loading full n8n documentation corpus...")

    with CORPUS_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    print(f"Loaded {len(documents):,} documents.")

    print("\nStarting full corpus chunking...")

    chunks = chunk_documents(
        documents,
        chunk_size=800,
        chunk_overlap=120
    )

    print("\nChunking completed.")

    print(f"Documents processed: {len(documents):,}")
    print(f"Total chunks generated: {len(chunks):,}")

    if chunks:
        lengths = [len(chunk["content"]) for chunk in chunks]

        print("\nCHUNK STATISTICS")
        print("=" * 60)
        print(f"Smallest chunk: {min(lengths):,} characters")
        print(f"Largest chunk: {max(lengths):,} characters")
        print(f"Average chunk: {sum(lengths) / len(lengths):,.2f} characters")

        print("\nSaving chunks...")

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT_FILE.open("w", encoding="utf-8") as file:
            json.dump(chunks, file, ensure_ascii=False, indent=2)

        print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
