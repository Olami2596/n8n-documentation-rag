import json
from pathlib import Path

from ingestion.embedder import embed_chunks


INPUT_FILE = Path("data/processed/n8n_full_chunks.json")
OUTPUT_FILE = Path("data/processed/n8n_full_embedded_chunks.json")

BATCH_SIZE = 16
EXPECTED_DIMENSION = 384


def main():
    print("Loading full documentation chunks...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks):,} chunks.")

    print("\nStarting embedding generation...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Expected dimensions: {EXPECTED_DIMENSION}")

    embedded_chunks = embed_chunks(
        chunks,
        batch_size=BATCH_SIZE
    )

    print("\nValidating embeddings...")

    if len(embedded_chunks) != len(chunks):
        raise ValueError("Embedded chunk count does not match input.")

    for index, chunk in enumerate(embedded_chunks):
        embedding = chunk.get("embedding")

        if not embedding:
            raise ValueError(f"Missing embedding at chunk {index}.")

        if len(embedding) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Incorrect dimensions at chunk {index}: "
                f"{len(embedding)}"
            )

    print("Embedding validation passed.")

    print("\nSaving embedded chunks...")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(embedded_chunks, file, ensure_ascii=False)

    print("\nFULL CORPUS EMBEDDING COMPLETED")
    print("=" * 60)
    print(f"Original chunks: {len(chunks):,}")
    print(f"Embedded chunks: {len(embedded_chunks):,}")
    print(f"Vector dimensions: {EXPECTED_DIMENSION}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
