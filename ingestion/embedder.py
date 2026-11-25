from vectorstore.embeddings import EmbeddingGenerator

def embed_chunks(chunks, batch_size=16):
    """
    Generate embeddings for documentation chunks.

    Each chunk retains its original metadata and receives
    an additional embedding field.
    """

    if not chunks:
        return []

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    embedder = EmbeddingGenerator()

    texts = [chunk["content"] for chunk in chunks]

    print(f"\nGenerating embeddings for {len(texts)} chunks...")

    embeddings = []

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        batch_embeddings = embedder.generate_embeddings(batch)

        embeddings.extend(batch_embeddings)

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):

        embedded_chunk = {
            **chunk,
            "embedding": embedding
        }

        embedded_chunks.append(embedded_chunk)

    print("\nEmbedding generation completed.")
    print(f"Chunks processed: {len(embedded_chunks)}")
    print(f"Embedding dimensions: {embedder.embedding_dimension}")

    return embedded_chunks
