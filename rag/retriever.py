from vectorstore.embeddings import EmbeddingGenerator
from vectorstore.pinecone_store import PineconeStore


class DocumentRetriever:
    """
    Retrieve relevant documentation from Pinecone
    and identify deprecated sources.
    """

    def __init__(
        self,
        namespace="n8n-docs",
        top_k=5
    ):
        self.namespace = namespace
        self.top_k = top_k

        self.embedder = EmbeddingGenerator()

        self.store = PineconeStore()
        self.index = self.store.connect()

    def search(self, question):
        """
        Embed a question, retrieve relevant chunks,
        and identify deprecated documentation.
        """

        if not question or not question.strip():
            raise ValueError("Search question cannot be empty.")

        print(f"\nSearching documentation for: {question}")

        # Step 1: Convert the question into an embedding.
        query_embedding = self.embedder.generate_embedding(
            question
        )

        # Step 2: Search Pinecone.
        results = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            namespace=self.namespace,
            include_metadata=True
        )

        # Step 3: Extract the results.
        retrieved_documents = []

        for match in results.matches:

            metadata = match.metadata or {}

            retrieved_documents.append({
                "chunk_id": match.id,
                "score": match.score,
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "category": metadata.get("category"),
                "status": metadata.get("status", "unspecified"),
                "content": metadata.get("content"),
                "document_id": metadata.get("document_id")
            })

        # Step 4: Identify deprecated results.
        deprecated_sources_by_id = {}
        for document in retrieved_documents:
            if document["status"] == "deprecated":
                document_id = document.get("document_id")

                if document_id not in deprecated_sources_by_id:
                    deprecated_sources_by_id[document_id] = document

        deprecated_results = list(deprecated_sources_by_id.values())

        # Step 5: Return results and retrieval information.
        return {
            "results": retrieved_documents,
            "has_deprecated_sources": bool(deprecated_results),
            "deprecated_sources": deprecated_results
        }
