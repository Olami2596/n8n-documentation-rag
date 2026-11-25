import json
import urllib.request
import xml.etree.ElementTree as ET

from pathlib import Path
from urllib.parse import urlparse


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SITEMAP_FILE = PROJECT_ROOT / "data" / "metadata" / "sitemap.xml"

OUTPUT_FILE = PROJECT_ROOT / "data" / "metadata" / "documentation_urls.json"

REPORT_FILE = PROJECT_ROOT / "data" / "metadata" / "sitemap_discovery_report.json"


# Website configuration
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DocumentationRAGBot/1.0)"
}


def get_tag_name(element):
    """Extract an XML tag name without its namespace."""
    return element.tag.split("}")[-1]


def download_sitemap(url):
    """Download a sitemap and return its XML content."""

    request = urllib.request.Request(
        url,
        headers=HEADERS
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def extract_sitemap_urls(xml_content):
    """Extract URLs and determine whether this is a sitemap index."""

    root = ET.fromstring(xml_content)

    root_type = get_tag_name(root)

    urls = []

    for element in root.iter():
        if get_tag_name(element) == "loc" and element.text:
            urls.append(element.text.strip())

    return root_type, urls


def discover_documentation_urls():
    """Discover all documentation pages from the sitemap index."""

    if not SITEMAP_FILE.exists():
        raise FileNotFoundError(
            f"Sitemap not found: {SITEMAP_FILE}"
        )

    visited_sitemaps = set()
    discovered_pages = {}
    errors = []

    def process_sitemap(sitemap_url):

        if sitemap_url in visited_sitemaps:
            return

        visited_sitemaps.add(sitemap_url)

        print(f"Inspecting: {sitemap_url}")

        try:
            xml_content = download_sitemap(sitemap_url)

            root_type, urls = extract_sitemap_urls(xml_content)

            if root_type == "sitemapindex":

                for child_url in urls:
                    process_sitemap(child_url)

            elif root_type == "urlset":

                for page_url in urls:
                    discovered_pages[page_url] = {
                        "url": page_url,
                        "domain": urlparse(page_url).netloc,
                        "source_sitemap": sitemap_url
                    }

            else:
                raise ValueError(
                    f"Unexpected XML root: {root_type}"
                )

        except Exception as error:
            errors.append({
                "sitemap": sitemap_url,
                "error": str(error)
            })

            print(f"ERROR: {error}")

    # Read our previously downloaded sitemap index
    sitemap_xml = SITEMAP_FILE.read_bytes()

    root_type, sitemap_urls = extract_sitemap_urls(sitemap_xml)

    if root_type == "sitemapindex":

        for sitemap_url in sitemap_urls:
            process_sitemap(sitemap_url)

    elif root_type == "urlset":

        # Also support a regular sitemap
        for page_url in sitemap_urls:
            discovered_pages[page_url] = {
                "url": page_url,
                "domain": urlparse(page_url).netloc,
                "source_sitemap": str(SITEMAP_FILE)
            }

    else:
        raise ValueError(
            f"Unexpected sitemap type: {root_type}"
        )

    # Convert discovered pages into a list
    pages = list(discovered_pages.values())

    # Save documentation URLs
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(pages, file, indent=4, ensure_ascii=False)

    # Save discovery report
    report = {
        "sitemaps_inspected": len(visited_sitemaps),
        "total_pages_discovered": len(pages),
        "failed_sitemaps": errors
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n========== DISCOVERY SUMMARY ==========")
    print(f"Sitemaps inspected: {len(visited_sitemaps)}")
    print(f"Documentation pages: {len(pages)}")
    print(f"Failed sitemaps: {len(errors)}")
    print(f"URLs saved to: {OUTPUT_FILE}")
    print(f"Report saved to: {REPORT_FILE}")

    return pages


if __name__ == "__main__":
    discover_documentation_urls()
