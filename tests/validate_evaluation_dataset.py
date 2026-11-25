import json
from pathlib import Path

DATASET = Path("tests/evaluation_questions.json")


def main():
    with DATASET.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    print(f"Total questions: {len(questions)}")

    categories = {}

    for question in questions:
        category = question["category"]
        categories[category] = categories.get(category, 0) + 1

    print("\nQuestions per category:")

    for category, count in categories.items():
        print(f"{category}: {count}")

    ids = [question["id"] for question in questions]

    assert len(ids) == len(set(ids)), "Duplicate question IDs found."
    assert len(questions) == 30, "Expected exactly 30 questions."

    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()
