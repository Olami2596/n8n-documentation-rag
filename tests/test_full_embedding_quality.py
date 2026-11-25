import json
import math
from pathlib import Path

INPUT_FILE = Path(
    "data/processed/n8n_full_embedded_chunks.json"
)

EXPECTED_DIMENSION = 384
TOLERANCE = 0.001


def main():
    print("Loading embedded chunks...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks):,} chunks.")

    chunk_ids = set()
    invalid_vectors = 0
    invalid_norms = 0
    duplicate_ids = 0

    print("\nChecking vector quality...")

    for index, chunk in enumerate(chunks):
        embedding = chunk.get("embedding")

        if not embedding or len(embedding) != EXPECTED_DIMENSION:
            invalid_vectors += 1
            continue

        if not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in embedding
        ):
            invalid_vectors += 1
            continue

        magnitude = math.sqrt(
            sum(value * value for value in embedding)
        )

        if abs(magnitude - 1.0) > TOLERANCE:
            invalid_norms += 1

        chunk_id = chunk.get("chunk_id")

        if chunk_id in chunk_ids:
            duplicate_ids += 1

        chunk_ids.add(chunk_id)

        if (index + 1) % 2000 == 0:
            print(f"Checked {index + 1:,} vectors...")

    print("\nFULL EMBEDDING QUALITY REPORT")
    print("=" * 60)
    print(f"Total chunks: {len(chunks):,}")
    print(f"Invalid vectors: {invalid_vectors:,}")
    print(f"Invalid normalized vectors: {invalid_norms:,}")
    print(f"Duplicate chunk IDs: {duplicate_ids:,}")

    if invalid_vectors or invalid_norms or duplicate_ids:
        raise ValueError("Embedding quality validation failed.")

    print("\nAll embedding quality checks passed!")


if __name__ == "__main__":
    main()
