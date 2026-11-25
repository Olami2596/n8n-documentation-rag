from rag.retriever import DocumentRetriever

def main():

    retriever = DocumentRetriever(top_k=5)

    questions = [
        "How do I create and execute an automation workflow?",
        "How can I start a workflow automatically when an event occurs?",
        "How can I interact with n8n programmatically using its API?"
    ]

    for question in questions:

        response = retriever.search(question)

        results = response["results"]

        print("\n" + "=" * 75)
        print("QUESTION:", question)
        print("=" * 75)

        for position, result in enumerate(results, start=1):

            print(f"\nResult {position}")
            print("Title:", result["title"])
            print("Score:", round(result["score"], 4))
            print("Status:", result["status"])
            print("URL:", result["url"])
            print("Content:", result["content"][:250])

        print("\nDeprecated sources found:",
              response["has_deprecated_sources"])

        if response["deprecated_sources"]:
            for source in response["deprecated_sources"]:
                print("Deprecated:", source["title"])


if __name__ == "__main__":
    main()
