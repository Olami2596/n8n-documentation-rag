import json
import math


def main():

    with open(
        "data/processed/n8n_embedded_chunks.json",
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks)} embedded chunks.\n")

    for chunk in chunks:

        embedding = chunk["embedding"]

        assert len(embedding) == 384

        assert all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in embedding
        )

        # Calculate vector magnitude (L2 norm).
        magnitude = math.sqrt(
            sum(value ** 2 for value in embedding)
        )

        assert math.isclose(
            magnitude,
            1.0,
            rel_tol=1e-4,
            abs_tol=1e-4
        ), f"Vector is not normalized: {magnitude}"

    print("Embedding quality checks passed!")
    print("All vectors have 384 dimensions.")
    print("All values are finite numbers.")
    print("All vectors are normalized.")


if __name__ == "__main__":
    main()
