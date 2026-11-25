from rag.generator import AnswerGenerator


def main():

    generator = AnswerGenerator()

    conversation_history = [
        {
            "role": "user",
            "content": "How do I create and execute an n8n workflow?"
        },
        {
            "role": "assistant",
            "content": (
                "You can create a workflow, add nodes, "
                "connect them, save it and execute it."
            )
        }
    ]

    question = "What about triggering it automatically?"

    rewritten_question = generator.rewrite_question(
        question,
        conversation_history
    )

    print("\nOriginal question:")
    print(question)

    print("\nRewritten retrieval query:")
    print(rewritten_question)


if __name__ == "__main__":
    main()
