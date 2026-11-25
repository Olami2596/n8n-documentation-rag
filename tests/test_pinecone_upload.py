import json

from ingestion.uploader import upload_chunks


def main():

    print("Loading embedded documentation...")

    with open(
        "data/processed/n8n_embedded_chunks.json",
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks)} embedded chunks.")

    uploaded = upload_chunks(
        chunks=chunks,
        namespace="n8n-docs",
        batch_size=100
    )

    print(f"\nTotal vectors uploaded: {uploaded}")

    assert uploaded == len(chunks)

    print("\nPinecone upload test passed!")


if __name__ == "__main__":
    main()
