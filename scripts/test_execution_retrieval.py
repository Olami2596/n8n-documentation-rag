from rag.retriever import DocumentRetriever


def main():
    retriever = DocumentRetriever(namespace="n8n-full-corpus", top_k=10)

    questions = [
        "How do I retry a failed n8n execution using the API?",
        "How do I retrieve all executions from n8n?",
        "How do I stop a running execution using the API?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        response = retriever.search(question)

        for index, result in enumerate(response["results"], start=1):
            print(f"\nResult {index}")
            print(f"Title: {result['title']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"URL: {result['url']}")
            print(f"Content:\n{result['content'][:700]}")
            print("-" * 80)


if __name__ == "__main__":
    main()
