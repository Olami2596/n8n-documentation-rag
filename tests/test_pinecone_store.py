from vectorstore.pinecone_store import PineconeStore

def main():

    store = PineconeStore(
        index_name="n8n-documentation",
        dimension=384,
        metric="cosine"
    )

    store.create_index()

    index = store.connect()

    description = store.pc.describe_index(
        "n8n-documentation"
    )

    print("\n========== INDEX DETAILS ==========")
    print("Name:", description.name)
    print("Dimension:", description.dimension)
    print("Metric:", description.metric)
    print("Host:", description.host)

    assert description.dimension == 384
    assert description.metric == "cosine"

    print("\nPinecone index test passed!")


if __name__ == "__main__":
    main()
