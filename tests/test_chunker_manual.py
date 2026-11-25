
import json

from pathlib import Path

from ingestion.chunker import chunk_documents


# Project paths
INPUT_FILE = Path("data/raw/n8n_sample_documents.json")
OUTPUT_FILE = Path("data/processed/n8n_sample_chunks.json")


def main():

    print("Loading documentation...")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        documents = json.load(file)

    print(f"Documents loaded: {len(documents)}")

    print("\nStarting chunking...")

    chunks = chunk_documents(
        documents,
        chunk_size=800,
        chunk_overlap=120
    )

    print(f"Total chunks created: {len(chunks)}")

    # Save chunks
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nChunks saved to: {OUTPUT_FILE}")

    # Document-level statistics
    print("\n========== CHUNKING SUMMARY ==========")

    for document in documents:

        document_chunks = [
            chunk for chunk in chunks
            if chunk["document_id"] == document["document_id"]
        ]

        print(
            f"{document['title']}: "
            f"{len(document_chunks)} chunks"
        )

    # Chunk size statistics
    sizes = [
        len(chunk["content"])
        for chunk in chunks
    ]

    if sizes:
        print("\n========== CHUNK SIZE STATISTICS ==========")

        print(f"Smallest chunk: {min(sizes)} characters")
        print(f"Largest chunk: {max(sizes)} characters")
        print(f"Average chunk: {sum(sizes) / len(sizes):.2f} characters")

    print("\nChunking test completed.")


if __name__ == "__main__":
    main()
