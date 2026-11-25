from rag.retriever import DocumentRetriever
from rag.generator import AnswerGenerator


def main():
    # Initialise the retrieval and answer-generation components
    retriever = DocumentRetriever(top_k=5)
    generator = AnswerGenerator()

    # Ask a question about n8n
    question = "How can I create and execute an automation workflow?"

    print("\nQUESTION:")
    print(question)

    # Step 1: Retrieve relevant documentation from Pinecone
    print("\nSearching the n8n documentation...")
    retrieval_response = retriever.search(question)

    # Step 2: Generate an answer using Groq
    print("\nGenerating an answer with Groq...")
    response = generator.generate_answer(
        question,
        retrieval_response
    )

    # Step 3: Display the generated answer
    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(response["answer"])

    # Step 4: Display the sources
    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in response["sources"]:
        print(f"\n[{source['reference']}] {source['title']}")
        print("URL:", source["url"])
        print("Status:", source["status"])

    # Step 5: Display deprecated documentation warnings
    if response["has_deprecated_sources"]:
        print("\nWARNING: Retrieved documentation includes deprecated content.")

        for source in response["deprecated_sources"]:
            print("-", source["title"])

    print("\nRAG pipeline test completed.")


if __name__ == "__main__":
    main()
