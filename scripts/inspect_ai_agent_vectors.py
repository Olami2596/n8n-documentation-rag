from vectorstore.embeddings import EmbeddingGenerator
from vectorstore.pinecone_store import PineconeStore

TARGET_URL = (
    "https://docs.n8n.io/integrations/builtin/cluster-nodes/"
    "root-nodes/n8n-nodes-langchain.agent"
)

query = "How do I add and configure the AI Agent node inside an n8n workflow?"

embedder = EmbeddingGenerator()
store = PineconeStore()
index = store.connect()

query_vector = embedder.generate_embedding(query)

results = index.query(
    vector=query_vector, top_k=10, namespace="n8n-full-corpus", include_metadata=True
)

print(f"\nQuery: {query}\n")
print("=" * 80)

for rank, match in enumerate(results.matches, start=1):
    metadata = match.metadata or {}
    url = metadata.get("url", "")

    print(f"\nRank: {rank}")
    print(f"Score: {match.score:.4f}")
    print(f"Title: {metadata.get('title')}")
    print(f"URL: {url}")
    print(f"Chunk index: {metadata.get('chunk_index')}")
    print(f"Target AI Agent page: {url.rstrip('/') == TARGET_URL}")
    print(f"Content: {str(metadata.get('content', ''))[:400]}")
    print("-" * 80)
