from rag.retriever import DocumentRetriever

TEST_CASES = [
    {
        "question": "How do I create a scheduled workflow?",
        "expected_keywords": ["schedule", "trigger"],
    },
    {
        "question": "How can I connect n8n to Telegram?",
        "expected_keywords": ["telegram", "credential"],
    },
    {
        "question": "How do I configure webhook authentication?",
        "expected_keywords": ["webhook", "authentication"],
    },
    {
        "question": "How can I execute workflows using the API?",
        "expected_keywords": ["workflow", "execution"],
    },
    {
        "question": "How do I deploy n8n using Docker?",
        "expected_keywords": ["docker", "deploy"],
    },
    {
        "question": "How do I add and configure the AI Agent node inside an n8n workflow?",
        "expected_keywords": [
            "AI Agent",
            "model",
            "tools",
            "workflow",
        ],
        "expected_urls": [
            "https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.agent"
        ],
    },
]


def main():
    retriever = DocumentRetriever(namespace="n8n-full-corpus", top_k=5)

    total_matches = 0
    total_expected = 0

    for case in TEST_CASES:
        print("\n" + "=" * 75)
        print(f"QUESTION: {case['question']}")
        print("=" * 75)

        response = retriever.search(case["question"])

        results = response["results"]

        matched_keywords = set()
        expected_urls = set(case.get("expected_urls", []))
        retrieved_urls = {
            result.get("url", "").rstrip("/")
            for result in results
        }

        matched_urls = expected_urls.intersection(retrieved_urls)

        for result in results:
            content = (
                (result.get("title") or "") + " " + (result.get("content") or "")
            ).lower()

            for keyword in case["expected_keywords"]:
                if keyword.lower() in content:
                    matched_keywords.add(keyword)

            print(f"{result['title']} | " f"Score: {result['score']:.4f}")

        expected = set(case["expected_keywords"])
        matches = len(matched_keywords)

        total_matches += matches
        total_expected += len(expected)

        print(f"\nExpected keywords: {sorted(expected)}")
        print(f"Matched keywords: {sorted(matched_keywords)}")
        print(f"Keyword coverage: {matches}/{len(expected)}")
        if expected_urls:
            print(f"Expected URLs: {len(expected_urls)}")
            print(f"Retrieved URLs: {len(matched_urls)}")
            print(f"URL match: {'PASS' if matched_urls else 'FAIL'}")

    print("\n" + "=" * 75)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Keyword coverage: {total_matches}/{total_expected}")

    if total_expected:
        print(f"Coverage percentage: " f"{total_matches / total_expected * 100:.2f}%")


if __name__ == "__main__":
    main()