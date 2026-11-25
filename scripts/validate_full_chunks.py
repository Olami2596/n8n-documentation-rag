import json
from collections import Counter
from pathlib import Path

CHUNKS_FILE = Path("data/processed/n8n_full_chunks.json")


def main():
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    lengths = [len(chunk.get("content") or "") for chunk in chunks]

    oversized = [
        chunk for chunk in chunks
        if len(chunk.get("content") or "") > 800
    ]

    very_short = [
        chunk for chunk in chunks
        if 0 < len(chunk.get("content") or "") < 100
    ]

    empty = [
        chunk for chunk in chunks
        if not (chunk.get("content") or "").strip()
    ]

    chunk_ids = [
        chunk.get("chunk_id")
        for chunk in chunks
        if chunk.get("chunk_id")
    ]

    duplicate_ids = [
        chunk_id
        for chunk_id, count in Counter(chunk_ids).items()
        if count > 1
    ]

    print("\nFULL CORPUS CHUNK VALIDATION")
    print("=" * 70)

    print(f"Total chunks: {len(chunks):,}")
    print(f"Empty chunks: {len(empty):,}")
    print(f"Chunks over 800 characters: {len(oversized):,}")
    print(f"Chunks under 100 characters: {len(very_short):,}")
    print(f"Duplicate chunk IDs: {len(duplicate_ids):,}")

    print("\nCHUNK LENGTH STATISTICS")
    print("-" * 70)
    print(f"Smallest: {min(lengths):,}")
    print(f"Largest: {max(lengths):,}")
    print(f"Average: {sum(lengths) / len(lengths):,.2f}")

    print("\nLARGEST CHUNKS")
    print("-" * 70)

    for chunk in sorted(
        oversized,
        key=lambda item: len(item.get("content") or ""),
        reverse=True
    )[:10]:
        print(
            f"\n{len(chunk['content']):,} characters | "
            f"{chunk.get('title')}"
        )
        print(f"Chunk ID: {chunk.get('chunk_id')}")
        print(f"URL: {chunk.get('url')}")
        print(f"Preview: {chunk['content'][:250]!r}")

    print("\nSHORTEST NON-EMPTY CHUNKS")
    print("-" * 70)

    for chunk in sorted(
        very_short,
        key=lambda item: len(item.get("content") or "")
    )[:10]:
        print(
            f"\n{len(chunk['content']):,} characters | "
            f"{chunk.get('title')}"
        )
        print(f"Chunk ID: {chunk.get('chunk_id')}")
        print(f"Preview: {chunk['content']!r}")

    if duplicate_ids:
        print("\nWARNING: Duplicate chunk IDs detected.")

    if empty:
        print("\nWARNING: Empty chunks detected.")

    print("\nValidation inspection completed.")


if __name__ == "__main__":
    main()
