from rag.retriever import DocumentRetriever


def main():
    print("Initializing full-corpus retriever...")

    retriever = DocumentRetriever(namespace="n8n-full-corpus", top_k=5)

    questions = [
        "How do I create a scheduled workflow?",
        "How can I connect n8n to Telegram?",
        "How do I configure webhook authentication?",
        "How can I execute workflows using the API?",
        "How do I deploy n8n using Docker?",
    ]

    for question in questions:
        print("\n" + "=" * 75)
        print(f"QUESTION: {question}")
        print("=" * 75)

        response = retriever.search(question)

        results = response["results"]

        print(f"\nRetrieved documents: {len(results)}")

        for index, result in enumerate(results, start=1):
            print(f"\n{index}. {result['title']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Category: {result['category']}")
            print(f"Status: {result['status']}")
            print(f"URL: {result['url']}")
            print(f"Content: {(result['content'] or '')[:250]}")

        print(
            f"\nDeprecated sources detected: " f"{response['has_deprecated_sources']}"
        )


if __name__ == "__main__":
    main()
