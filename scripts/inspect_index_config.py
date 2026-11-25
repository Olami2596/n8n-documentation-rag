from vectorstore.pinecone_store import PineconeStore

store = PineconeStore()

description = store.pc.describe_index("n8n-documentation")

print("\n--- PINECONE INDEX CONFIGURATION ---")
print(description)

print("\n--- INDEX DIMENSION ---")
print(description.dimension)

print("\n--- INDEX METRIC ---")
print(description.metric)

print("\n--- INDEX SPEC ---")
print(description.spec)