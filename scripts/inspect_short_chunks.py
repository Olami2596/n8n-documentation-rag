import json
from collections import defaultdict
from pathlib import Path

CHUNKS_FILE = Path("data/processed/n8n_full_chunks.json")


def main():
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Group chunks by their original document.
    documents = defaultdict(list)

    for chunk in chunks:
        documents[chunk["document_id"]].append(chunk)

    short_chunks = [
        chunk for chunk in chunks
        if 0 < len(chunk.get("content") or "") < 100
    ]

    print(f"Short chunks found: {len(short_chunks)}")

    print("\nINSPECTING FIRST 15 SHORT CHUNKS")
    print("=" * 75)

    for index, chunk in enumerate(short_chunks[:15], start=1):
        doc_chunks = sorted(
            documents[chunk["document_id"]],
            key=lambda item: item["chunk_index"]
        )

        current_position = next(
            i for i, item in enumerate(doc_chunks)
            if item["chunk_id"] == chunk["chunk_id"]
        )

        previous_chunk = (
            doc_chunks[current_position - 1]
            if current_position > 0 else None
        )

        next_chunk = (
            doc_chunks[current_position + 1]
            if current_position + 1 < len(doc_chunks) else None
        )

        print(f"\n{index}. {chunk.get('title')}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print(f"Short content: {chunk['content']!r}")

        if previous_chunk:
            print(
                "PREVIOUS:",
                previous_chunk["content"][:300]
            )

        if next_chunk:
            print(
                "NEXT:",
                next_chunk["content"][:300]
            )

        print("-" * 75)


if __name__ == "__main__":
    main()
