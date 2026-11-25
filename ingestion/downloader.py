import json
import time
import logging

from pathlib import Path
from urllib.parse import urlparse
import requests

from ingestion.parser import parse_document


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

URLS_FILE = PROJECT_ROOT / "data" / "metadata" / "documentation_urls.json"

OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "n8n_sample_documents.json"

REPORT_FILE = PROJECT_ROOT / "data" / "metadata" / "download_report.json"


# Download configuration
REQUEST_DELAY = 1.5
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": "DocumentationRAGBot/1.0 (Educational demonstration project)"
}


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_documentation_urls():
    """Load discovered documentation URLs."""

    with URLS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def select_sample_pages(pages, limit=5):
    """
    Select meaningful documentation pages from different categories.

    Avoid category landing pages by requiring a minimum URL depth.
    """

    minimum_depth = {
        "general": 1,
        "deploy": 2,
        "build": 2,
        "integrations": 4,
        "connect": 2
    }

    selected_pages = []
    selected_categories = set()

    for page in pages:

        sitemap_url = page["source_sitemap"]
        page_url = page["url"]

        sitemap_path = urlparse(sitemap_url).path

        category = (
            sitemap_path
            .replace("/sitemap-pages.xml", "")
            .strip("/")
        )

        if not category:
            category = "general"

        # We only want the five categories in our test
        if category not in minimum_depth:
            continue

        if category in selected_categories:
            continue

        # Count the actual URL path segments
        path = urlparse(page_url).path.strip("/")

        depth = len(path.split("/")) if path else 0

        required_depth = minimum_depth[category]

        if depth < required_depth:
            continue

        selected_pages.append(page)
        selected_categories.add(category)

        if len(selected_pages) >= limit:
            break

    return selected_pages


def download_page(session, url):
    """Download one documentation webpage."""

    response = session.get(
        url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.text


def download_documents():
    """Download and process a small sample of documentation pages."""

    if not URLS_FILE.exists():
        raise FileNotFoundError(
            f"Documentation URL file not found: {URLS_FILE}"
        )

    pages = load_documentation_urls()

    selected_pages = select_sample_pages(
        pages,
        limit=5
    )

    documents = []
    failures = []

    logger.info(
        "Starting download of %s documentation pages",
        len(selected_pages)
    )

    with requests.Session() as session:

        session.headers.update(HEADERS)

        for index, page in enumerate(selected_pages):

            url = page["url"]
            sitemap = page["source_sitemap"]

            logger.info(
                "Downloading page %s/%s: %s",
                index + 1,
                len(selected_pages),
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
                        "No usable documentation content extracted"
                    )

                documents.append(document)

                logger.info(
                    "Successfully extracted: %s",
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

            # Respectful delay between requests
            if index < len(selected_pages) - 1:
                time.sleep(REQUEST_DELAY)

    # Save extracted documents
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Save download report
    report = {
        "pages_selected": len(selected_pages),
        "pages_downloaded": len(documents),
        "failed_pages": len(failures),
        "failures": failures
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    logger.info("========== DOWNLOAD SUMMARY ==========")
    logger.info("Pages selected: %s", len(selected_pages))
    logger.info("Pages downloaded: %s", len(documents))
    logger.info("Failed pages: %s", len(failures))
    logger.info("Documents saved to: %s", OUTPUT_FILE)
    logger.info("Report saved to: %s", REPORT_FILE)

    return documents


if __name__ == "__main__":
    download_documents()
