
from vectorstore.embeddings import EmbeddingGenerator


def main():
    print("Initializing embedding generator...")

    embedder = EmbeddingGenerator()

    sample_text = (
        "n8n is a workflow automation platform "
        "that connects different applications."
    )

    print("\nGenerating embedding...")

    embedding = embedder.generate_embedding(sample_text)

    print("Embedding generated successfully.")
    print("Embedding dimensions:", len(embedding))
    print("Expected dimensions:", embedder.embedding_dimension)
    print("First 10 values:", embedding[:10])

    assert len(embedding) == 384
    assert embedder.embedding_dimension == 384

    print("\nEmbedding test passed!")


if __name__ == "__main__":
    main()
