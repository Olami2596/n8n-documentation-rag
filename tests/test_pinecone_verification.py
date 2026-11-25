import json

from vectorstore.pinecone_store import PineconeStore


def main():

    store = PineconeStore()
    index = store.connect()

    namespace = "n8n-docs"

    print("Checking Pinecone index statistics...\n")

    stats = index.describe_index_stats()

    print("Total vectors:", stats.total_vector_count)

    namespace_stats = stats.namespaces.get(namespace)

    if namespace_stats is None:
        raise AssertionError("Expected namespace was not found.")

    print("Namespace:", namespace)
    print("Namespace vector count:", namespace_stats.vector_count)

    assert namespace_stats.vector_count == 29

    # Load one locally stored chunk to verify its actual record.
    with open(
        "data/processed/n8n_embedded_chunks.json",
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    sample_chunk = chunks[0]
    sample_id = sample_chunk["chunk_id"]

    print("\nFetching a sample vector:", sample_id)

    response = index.fetch(
        ids=[sample_id],
        namespace=namespace
    )

    vector = response.vectors.get(sample_id)

    assert vector is not None, "Sample vector was not found."

    metadata = vector.metadata

    print("Sample vector found!")
    print("Title:", metadata["title"])
    print("Category:", metadata["category"])
    print("URL:", metadata["url"])
    print("Content preview:", metadata["content"][:150])

    # Verify deprecated document metadata.
    deprecated_chunks = [
        chunk for chunk in chunks
        if chunk["title"] == "Activation Trigger"
    ]

    for chunk in deprecated_chunks:
        result = index.fetch(
            ids=[chunk["chunk_id"]],
            namespace="n8n-docs"
        )

        vector = result.vectors.get(chunk["chunk_id"])

        if vector is None:
            raise AssertionError(
                f"Vector not found: {chunk['chunk_id']}"
            )

        status = vector.metadata.get("status")

        print(
            f"Verified: {chunk['title']} -> {status}"
        )

        assert status == "deprecated", (
            f"Expected deprecated, got {status}"
        )

    assert metadata["title"] == sample_chunk["title"]
    assert metadata["content"] == sample_chunk["content"]

    print("\nPinecone verification passed!")


if __name__ == "__main__":
    main()
