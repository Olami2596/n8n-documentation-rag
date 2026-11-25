
from ingestion.downloader import download_page
from ingestion.parser import parse_document

URL = (
    "https://docs.n8n.io/integrations/builtin/core-nodes/"
    "n8n-nodes-base.activationtrigger"
)

html = download_page(
    __import__("requests").Session(),
    URL
)

document = parse_document(
    html=html,
    url=URL,
    source_sitemap="https://docs.n8n.io/sitemap-integrations.xml"
)

print("Document title:", document["title"])
print("Document status:", document["status"])
