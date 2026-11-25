import json
import hashlib

from pathlib import Path
from collections import Counter
from statistics import mean


CORPUS_FILE = Path("data/raw/n8n_full_corpus.json")
REPORT_FILE = Path("data/metadata/corpus_validation_report.json")


def main():

    with CORPUS_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    print("\nFULL CORPUS VALIDATION")
    print("=" * 60)

    # Basic statistics
    total = len(documents)

    urls = [
        doc.get("url", "").rstrip("/")
        for doc in documents
        if doc.get("url")
    ]

    document_ids = [
        doc.get("document_id")
        for doc in documents
        if doc.get("document_id")
    ]

    content_lengths = [
        len(doc.get("content") or "")
        for doc in documents
    ]

    # Check missing fields
    missing_titles = [
        doc.get("url")
        for doc in documents
        if not doc.get("title")
    ]

    missing_urls = [
        doc.get("title")
        for doc in documents
        if not doc.get("url")
    ]

    empty_content = [
        doc.get("url")
        for doc in documents
        if not (doc.get("content") or "").strip()
    ]

    short_content = [
        {
            "title": doc.get("title"),
            "url": doc.get("url"),
            "characters": len(doc.get("content") or "")
        }
        for doc in documents
        if 0 < len((doc.get("content") or "").strip()) < 150
    ]

    # Duplicate checks
    duplicate_urls = len(urls) - len(set(urls))

    duplicate_ids = (
        len(document_ids) - len(set(document_ids))
    )

    content_hashes = [
        hashlib.sha256(
            (doc.get("content") or "").encode("utf-8")
        ).hexdigest()
        for doc in documents
        if (doc.get("content") or "").strip()
    ]

    duplicate_content = (
        len(content_hashes) - len(set(content_hashes))
    )

    # Category statistics
    categories = Counter(
        doc.get("category") or "uncategorized"
        for doc in documents
    )

    # Document status statistics
    statuses = Counter(
        doc.get("status") or "unspecified"
        for doc in documents
    )

    # Content statistics
    non_empty_lengths = [
        length for length in content_lengths
        if length > 0
    ]

    average_length = (
        round(mean(non_empty_lengths), 2)
        if non_empty_lengths else 0
    )

    # Print summary
    print(f"Total documents: {total}")
    print(f"Unique URLs: {len(set(urls))}")
    print(f"Duplicate URLs: {duplicate_urls}")
    print(f"Duplicate document IDs: {duplicate_ids}")
    print(f"Duplicate content: {duplicate_content}")

    print("\nCONTENT QUALITY")
    print("-" * 40)
    print(f"Empty documents: {len(empty_content)}")
    print(f"Short documents (<150 chars): {len(short_content)}")
    print(f"Missing titles: {len(missing_titles)}")
    print(f"Missing URLs: {len(missing_urls)}")
    print(f"Average content length: {average_length}")

    if non_empty_lengths:
        print(f"Shortest content: {min(non_empty_lengths)}")
        print(f"Longest content: {max(non_empty_lengths)}")

    print("\nDOCUMENT CATEGORIES")
    print("-" * 40)

    for category, count in categories.most_common():
        print(f"{category}: {count}")

    print("\nDOCUMENT STATUSES")
    print("-" * 40)

    for status, count in statuses.most_common():
        print(f"{status}: {count}")

    print("\nSHORTEST DOCUMENTS")
    print("-" * 40)

    for doc in sorted(
        documents,
        key=lambda item: len(item.get("content") or "")
    )[:15]:

        print(
            f"{len(doc.get('content') or ''):>6} chars | "
            f"{doc.get('title')} | {doc.get('url')}"
        )

    # Save machine-readable report
    report = {
        "total_documents": total,
        "unique_urls": len(set(urls)),
        "duplicate_urls": duplicate_urls,
        "duplicate_document_ids": duplicate_ids,
        "duplicate_content": duplicate_content,
        "empty_documents": len(empty_content),
        "short_documents": len(short_content),
        "missing_titles": len(missing_titles),
        "missing_urls": len(missing_urls),
        "average_content_length": average_length,
        "category_counts": dict(categories),
        "status_counts": dict(statuses),
        "empty_content_urls": empty_content,
        "short_documents_details": short_content
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nValidation report saved to:")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()
