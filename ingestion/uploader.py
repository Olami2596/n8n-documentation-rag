from vectorstore.pinecone_store import PineconeStore

def upload_chunks(
    chunks,
    namespace="n8n-docs",
    batch_size=100
):
    """
    Upload embedded documentation chunks to Pinecone.
    """

    if not chunks:
        raise ValueError("No chunks provided for upload.")

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    store = PineconeStore()
    index = store.connect()

    vectors = []

    for chunk in chunks:

        embedding = chunk["embedding"]

        if len(embedding) != store.dimension:
            raise ValueError(
                f"Invalid embedding dimension for {chunk['chunk_id']}"
            )

        metadata = {
            "document_id": chunk["document_id"],
            "title": chunk["title"],
            "url": chunk["url"],
            "category": chunk["category"],
            "source": chunk["source"],
            "content": chunk["content"],
            "chunk_index": chunk["chunk_index"],
            "start_index": chunk["start_index"]
        }

        # Pinecone metadata cannot contain null values.
        metadata = {
            key: value
            for key, value in metadata.items()
            if value is not None
        }

        if chunk.get("status") is not None:
            metadata["status"] = chunk["status"]

        vectors.append({
            "id": chunk["chunk_id"],
            "values": embedding,
            "metadata": metadata
        })

    total_uploaded = 0

    for start in range(0, len(vectors), batch_size):

        batch = vectors[start:start + batch_size]

        response = index.upsert(
            vectors=batch,
            namespace=namespace
        )

        total_uploaded += response.upserted_count

        print(
            f"Uploaded {total_uploaded}/{len(vectors)} vectors"
        )

    print("\nPinecone upload completed.")

    return total_uploaded
