import json
import time
import logging

from pathlib import Path
from urllib.parse import urlparse

import requests

from ingestion.parser import parse_document


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

URLS_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "documentation_urls.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "n8n_full_corpus.json"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "full_download_report.json"
)


# --------------------------------------------------
# 2. DOWNLOAD CONFIGURATION
# --------------------------------------------------

REQUEST_DELAY = 1.5
REQUEST_TIMEOUT = 30

MAX_RETRIES = 3
RETRY_DELAY = 3

CHECKPOINT_INTERVAL = 20

HEADERS = {
    "User-Agent": (
        "DocumentationRAGBot/1.0 "
        "(Educational demonstration project)"
    )
}


# --------------------------------------------------
# 3. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 4. LOAD DOCUMENTATION URLS
# --------------------------------------------------

def load_documentation_urls():
    """Load and deduplicate discovered documentation URLs."""

    with URLS_FILE.open("r", encoding="utf-8") as file:
        pages = json.load(file)

    unique_pages = {}

    for page in pages:
        url = page["url"].rstrip("/")
        unique_pages.setdefault(url, page)

    return list(unique_pages.values())


# --------------------------------------------------
# 5. DOWNLOAD ONE PAGE
# --------------------------------------------------

def download_page(session, url):
    """Download a webpage with retry support."""

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            return response.text

        except requests.RequestException as error:

            logger.warning(
                "Attempt %s/%s failed for %s: %s",
                attempt,
                MAX_RETRIES,
                url,
                error
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

            else:
                raise


# --------------------------------------------------
# 6. SAVE PROGRESS
# --------------------------------------------------

def save_documents(documents):
    """Save the downloaded document collection."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )

    temporary_file.replace(OUTPUT_FILE)


def save_report(report):
    """Save the download report."""

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


# --------------------------------------------------
# 7. LOAD EXISTING PROGRESS
# --------------------------------------------------

def load_existing_documents():
    """Load previously downloaded documents for resuming."""

    if not OUTPUT_FILE.exists():
        return []

    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    logger.info(
        "Loaded %s previously downloaded documents.",
        len(documents)
    )

    return documents


# --------------------------------------------------
# 8. FULL DOCUMENTATION DOWNLOAD
# --------------------------------------------------

def download_full_corpus():

    if not URLS_FILE.exists():
        raise FileNotFoundError(
            f"Documentation URL file not found: {URLS_FILE}"
        )

    pages = load_documentation_urls()

    documents = load_existing_documents()

    downloaded_urls = {
        document["url"].rstrip("/")
        for document in documents
        if document.get("url")
    }

    pending_pages = [
        page for page in pages
        if page["url"].rstrip("/") not in downloaded_urls
    ]

    failures = []

    logger.info("========== FULL DOWNLOAD STARTED ==========")
    logger.info("Total unique URLs: %s", len(pages))
    logger.info("Previously downloaded: %s", len(downloaded_urls))
    logger.info("Remaining pages: %s", len(pending_pages))

    with requests.Session() as session:

        session.headers.update(HEADERS)

        for index, page in enumerate(pending_pages, start=1):

            url = page["url"]
            sitemap = page["source_sitemap"]

            logger.info(
                "Processing page %s/%s: %s",
                index,
                len(pending_pages),
                url
            )

            try:

                html = download_page(session, url)

                document = parse_document(
                    html=html,
                    url=url,
                    source_sitemap=sitemap
                )

                if document is None:
                    raise ValueError(
                        "No usable documentation content extracted."
                    )

                documents.append(document)

                downloaded_urls.add(url.rstrip("/"))

                logger.info(
                    "Downloaded successfully: %s",
                    document["title"]
                )

            except Exception as error:

                failures.append({
                    "url": url,
                    "error": str(error)
                })

                logger.exception(
                    "Failed to process: %s",
                    url
                )

            # Save progress periodically.
            if index % CHECKPOINT_INTERVAL == 0:

                save_documents(documents)

                save_report({
                    "total_unique_urls": len(pages),
                    "downloaded_documents": len(documents),
                    "remaining_at_checkpoint": (
                        len(pending_pages) - index
                    ),
                    "failures": failures
                })

                logger.info(
                    "Checkpoint saved: %s documents",
                    len(documents)
                )

            # Respectful delay between requests.
            if index < len(pending_pages):
                time.sleep(REQUEST_DELAY)

    # Save final documents and report.
    save_documents(documents)

    report = {
        "total_unique_urls": len(pages),
        "downloaded_documents": len(documents),
        "failed_pages": len(failures),
        "failures": failures
    }

    save_report(report)

    logger.info("========== DOWNLOAD COMPLETED ==========")
    logger.info("Total URLs: %s", len(pages))
    logger.info("Downloaded documents: %s", len(documents))
    logger.info("Failed pages: %s", len(failures))
    logger.info("Corpus saved to: %s", OUTPUT_FILE)
    logger.info("Report saved to: %s", REPORT_FILE)

    return report


if __name__ == "__main__":
    download_full_corpus()
