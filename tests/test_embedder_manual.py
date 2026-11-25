
import json

from ingestion.embedder import embed_chunks


def main():

    print("Loading documentation chunks...")

    with open(
        "data/processed/n8n_sample_chunks.json",
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks)} chunks.")

    embedded_chunks = embed_chunks(
        chunks=chunks,
        batch_size=16
    )

    output_path = "data/processed/n8n_embedded_chunks.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            embedded_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved embeddings to: {output_path}")

    # Basic validation
    assert len(embedded_chunks) == len(chunks)

    for chunk in embedded_chunks:

        assert "embedding" in chunk
        assert len(chunk["embedding"]) == 384
        assert len(chunk["chunk_id"]) > 0

    print("\nAll embedding validations passed!")


if __name__ == "__main__":
    main()
