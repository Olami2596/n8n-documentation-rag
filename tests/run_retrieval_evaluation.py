import json
from pathlib import Path

from rag.retriever import DocumentRetriever

DATASET_FILE = Path("tests/evaluation_questions_v3.json")
OUTPUT_FILE = Path("data/processed/retrieval_evaluation_results_v3.json")


def main():
    # Load evaluation questions
    with DATASET_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    print("=" * 75)
    print("N8N DOCUMENTATION RAG - RETRIEVAL EVALUATION")
    print("=" * 75)
    print(f"Total questions: {len(questions)}")
    print("Namespace: n8n-full-corpus")
    print("Top K: 5")

    # Initialize the existing retriever
    retriever = DocumentRetriever(namespace="n8n-full-corpus", top_k=5)

    evaluation_results = []

    for position, question in enumerate(questions, start=1):
        question_id = question["id"]
        query = question["question"]

        print(f"\n[{position}/{len(questions)}] {query}")

        # Retrieve relevant documentation chunks
        response = retriever.search(query)

        retrieved_results = []

        for rank, result in enumerate(response["results"], start=1):
            retrieved_results.append(
                {
                    "rank": rank,
                    "chunk_id": result["chunk_id"],
                    "title": result["title"],
                    "url": result["url"],
                    "category": result["category"],
                    "score": result["score"],
                    "content": result["content"],
                }
            )

            print(f"  {rank}. {result['title']} " f"(score: {result['score']:.4f})")

        evaluation_results.append(
            {
                "question_id": question_id,
                "category": question["category"],
                "question": query,
                "expected_topic": question["expected_topic"],
                "expected_keywords": question["expected_keywords"],
                "retrieved_count": len(retrieved_results),
                "results": retrieved_results,
            }
        )

    # Save complete evaluation results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(evaluation_results, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETED")
    print("=" * 75)
    print(f"Questions evaluated: {len(evaluation_results)}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
