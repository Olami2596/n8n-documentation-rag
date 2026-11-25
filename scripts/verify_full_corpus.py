import json
from pathlib import Path

from vectorstore.pinecone_store import PineconeStore


INPUT_FILE = Path(
    "data/processed/n8n_full_embedded_chunks.json"
)

NAMESPACE = "n8n-full-corpus"
EXPECTED_COUNT = 12_287


def main():
    print("Loading embedded dataset...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    sample_id = chunks[0]["chunk_id"]

    print("Connecting to Pinecone...")

    store = PineconeStore()
    index = store.connect()

    print("\nChecking index statistics...")

    stats = index.describe_index_stats()

    namespace_stats = stats.namespaces or {}
    namespace_info = namespace_stats.get(NAMESPACE)

    if namespace_info is None:
        raise RuntimeError(
            f"Namespace '{NAMESPACE}' was not found."
        )

    namespace_count = namespace_info.vector_count

    print(f"Index total vectors: {stats.total_vector_count:,}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Namespace vector count: {namespace_count:,}")

    print("\nFetching a sample vector...")

    response = index.fetch(
        ids=[sample_id],
        namespace=NAMESPACE
    )

    vector = response.vectors.get(sample_id)

    if vector is None:
        raise RuntimeError("Sample vector was not found.")

    print("Sample vector found.")
    print(f"Vector ID: {sample_id}")
    print(f"Vector dimensions: {len(vector.values)}")

    metadata = vector.metadata or {}

    print(f"Document title: {metadata.get('title')}")
    print(f"Document URL: {metadata.get('url')}")
    print(f"Category: {metadata.get('category')}")
    print(f"Chunk index: {metadata.get('chunk_index')}")

    if namespace_count != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNT:,} vectors, "
            f"but found {namespace_count:,}."
        )

    if len(vector.values) != 384:
        raise RuntimeError("Unexpected vector dimensions.")

    if not metadata.get("content"):
        raise RuntimeError("Sample vector has no content metadata.")

    print("\nPINECONE VERIFICATION PASSED")
    print("All expected vectors are present.")
    print("Sample vector and metadata verified.")


if __name__ == "__main__":
    main()
