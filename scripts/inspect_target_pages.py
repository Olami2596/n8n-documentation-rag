import json
from pathlib import Path

URLS_FILE = Path("data/metadata/documentation_urls.json")

with URLS_FILE.open("r", encoding="utf-8") as file:
    pages = json.load(file)

keywords = (
    "scheduletrigger",
    "webhook",
    "workflowtrigger",
    "manualtrigger",
    "trigger-nodes",
)

matches = []

for page in pages:
    url = page["url"].lower()

    if any(keyword in url for keyword in keywords):
        matches.append(page["url"])

print(f"Trigger-related documentation pages: {len(matches)}")
print("=" * 75)

for index, url in enumerate(matches, start=1):
    print(f"{index}. {url}")
