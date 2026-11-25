import json
from pathlib import Path

from ingestion.uploader import upload_chunks


INPUT_FILE = Path(
    "data/processed/n8n_full_embedded_chunks.json"
)

NAMESPACE = "n8n-full-corpus"
BATCH_SIZE = 100


def main():
    print("Loading full embedded corpus...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks):,} embedded chunks.")

    print("\nPINECONE UPLOAD CONFIGURATION")
    print("=" * 60)
    print("Index: n8n-documentation")
    print(f"Namespace: {NAMESPACE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Vectors to upload: {len(chunks):,}")

    confirmation = input(
        "\nType UPLOAD to proceed: "
    )

    if confirmation != "UPLOAD":
        print("Upload cancelled.")
        return

    print("\nStarting Pinecone upload...")

    uploaded = upload_chunks(
        chunks,
        namespace=NAMESPACE,
        batch_size=BATCH_SIZE
    )

    print("\nUPLOAD SUMMARY")
    print("=" * 60)
    print(f"Expected vectors: {len(chunks):,}")
    print(f"Successfully uploaded: {uploaded:,}")

    if uploaded != len(chunks):
        raise RuntimeError(
            "Upload count does not match expected chunk count."
        )

    print("\nFull corpus upload completed successfully!")


if __name__ == "__main__":
    main()
