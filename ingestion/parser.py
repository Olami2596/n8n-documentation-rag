import hashlib
import re

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

# DOCUMENT UTILITIES


def generate_document_id(url):
    """Generate a deterministic ID from a document URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def extract_category(source_sitemap):
    """Determine the documentation category from its sitemap."""

    path = urlparse(source_sitemap).path.strip("/")
    segments = path.split("/")

    if len(segments) == 1:
        return "general"

    return segments[0]


# HTML TO MARKDOWN CONVERSION


def render_table(table, base_url):
    """Convert an HTML table into Markdown."""

    rows = []

    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        row = []

        for cell in cells:
            text = render_element(cell, base_url)
            text = re.sub(r"\s+", " ", text).strip()
            text = text.replace("|", r"\|")
            row.append(text)

        if row:
            rows.append(row)

    if not rows:
        return ""

    column_count = max(len(row) for row in rows)

    rows = [row + [""] * (column_count - len(row)) for row in rows]

    header = rows[0]
    separator = ["---"] * column_count

    markdown_rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for row in rows[1:]:
        markdown_rows.append("| " + " | ".join(row) + " |")

    return "\n" + "\n".join(markdown_rows) + "\n"


def render_accessible_table(table, base_url):
    """
    Convert an accessible HTML table using ARIA roles into Markdown.

    Supports:
    - role="table"
    - role="row"
    - role="columnheader"
    - role="cell"
    """

    rows = []

    # Find all rows inside the accessible table.
    for row_element in table.find_all(attrs={"role": "row"}):

        cells = row_element.find_all(
            attrs={"role": ["columnheader", "cell"]},
            recursive=False
        )

        row = []

        for cell in cells:

            content = render_children(cell, base_url)

            # Normalize whitespace inside each cell.
            content = re.sub(r"\s+", " ", content).strip()

            # Escape Markdown table separators.
            content = content.replace("|", r"\|")

            row.append(content)

        if row:
            rows.append(row)

    if not rows:
        return ""

    # Find the header row.
    header_index = next(
        (
            index
            for index, row_element in enumerate(
                table.find_all(attrs={"role": "row"})
            )
            if row_element.find(
                attrs={"role": "columnheader"},
                recursive=False
            )
        ),
        None
    )

    if header_index is None:
        # No explicit header: generate generic column names.
        column_count = max(len(row) for row in rows)
        header = [
            f"Column {index + 1}"
            for index in range(column_count)
        ]
        data_rows = rows
    else:
        header = rows[header_index]
        data_rows = rows[header_index + 1:]

    # Ensure all rows have the same number of columns.
    column_count = max(
        len(header),
        max((len(row) for row in data_rows), default=0)
    )

    header += [""] * (column_count - len(header))

    normalized_rows = [
        row + [""] * (column_count - len(row))
        for row in data_rows
    ]

    # Build Markdown table.
    separator = ["---"] * column_count

    markdown_rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |"
    ]

    for row in normalized_rows:
        markdown_rows.append(
            "| " + " | ".join(row) + " |"
        )

    return "\n" + "\n".join(markdown_rows) + "\n\n"

def render_element(element, base_url):
    """Convert an HTML element into structured Markdown."""

    if isinstance(element, NavigableString):
        return str(element)

    if not isinstance(element, Tag):
        return ""

    tag = element.name

    # Ignore unwanted elements.
    if tag in [
        "script",
        "style",
        "nav",
        "footer",
        "aside",
        "form",
        "button",
        "noscript",
        "svg",
    ]:
        return ""

    # Headings.
    if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:

        level = int(tag[1])

        # Remove empty heading anchor links.
        for anchor in element.find_all("a"):
            if not anchor.get_text(strip=True) or "headerlink" in anchor.get(
                "class", []
            ):
                anchor.decompose()

        content = render_children(element, base_url).strip()

        return f"\n{'#' * level} {content}\n\n" if content else ""

    # Paragraphs.
    if tag == "p":

        content = render_children(element, base_url).strip()

        return f"\n{content}\n\n" if content else ""

    # Hyperlinks.
    if tag == "a":

        text = render_children(element, base_url).strip()
        href = element.get("href")

        if not text:
            return ""

        # Remove Previous/Next page navigation links.
        if re.match(r"^(previous|next)(?=\s|[A-Z])", text, re.IGNORECASE):
            return ""

        if not href:
            return text

        absolute_url = urljoin(base_url, href)

        return f"[{text}]({absolute_url})"

    # Inline code.
    if tag == "code":

        content = element.get_text()

        if element.find_parent("pre"):
            return content

        return f"`{content}`"

    # Code blocks.
    if tag == "pre":

        code_element = element.find("code")

        if code_element:
            content = code_element.get_text()
        else:
            content = element.get_text()

        content = content.strip("\n")

        if not content:
            return ""

        return f"\n```text\n{content}\n```\n\n"

    # Bold text.
    if tag in ["strong", "b"]:

        content = render_children(element, base_url).strip()

        return f"**{content}**"

    # Italic text.
    if tag in ["em", "i"]:

        content = render_children(element, base_url).strip()

        return f"*{content}*"

    # Line breaks.
    if tag == "br":
        return "\n"

    # Lists.
    if tag in ["ul", "ol"]:

        items = []

        for index, li in enumerate(element.find_all("li", recursive=False), start=1):

            content = render_children(li, base_url).strip()

            if not content:
                continue

            if tag == "ol":
                prefix = f"{index}."
            else:
                prefix = "-"

            items.append(f"{prefix} {content}")

        return "\n" + "\n".join(items) + "\n\n" if items else ""

    # Native HTML tables.
    if tag == "table":
        return render_table(element, base_url)

    # Accessible tables built with div elements and ARIA roles.
    if element.get("role") == "table":
        return render_accessible_table(element, base_url)

    # Blockquotes.
    if tag == "blockquote":

        content = render_children(element, base_url).strip()

        if not content:
            return ""

        lines = content.splitlines()

        return "\n" + "\n".join(f"> {line}" for line in lines) + "\n\n"

    # Horizontal rules.
    if tag == "hr":
        return "\n---\n\n"

    # Generic containers.
    return render_children(element, base_url)


def render_children(element, base_url):
    """Render all child elements."""

    return "".join(render_element(child, base_url) for child in element.children)


# DOCUMENT CLEANUP


def clean_document_html(soup):
    """Remove navigation, scripts and irrelevant page elements."""

    unwanted_selectors = [
        "nav",
        "footer",
        "aside",
        "script",
        "style",
        "noscript",
        "form",
        "button",
        ".pagination-nav",
        ".md-footer",
        ".md-content__button",
        ".page-nav",
    ]

    for selector in unwanted_selectors:
        for element in soup.select(selector):
            element.decompose()

    return soup


# DOCUMENT STATUS DETECTION


def detect_document_status(content_area):
    """
    Detect explicit deprecation notices.

    Avoid marking a complete document as deprecated merely
    because it mentions a deprecated feature.
    """
    # Detect an explicit deprecation badge.
    # The badge must contain exactly the word "Deprecated".
    for badge in content_area.select("span.truncate"):
        badge_text = badge.get_text(" ", strip=True)

        if badge_text.casefold() == "deprecated":
            return "deprecated"

    notice_selectors = [
        ".admonition",
        ".alert",
        ".callout",
        "[role='alert']",
        "[class*='deprecated']",
    ]

    for selector in notice_selectors:

        for notice in content_area.select(selector):

            notice_text = notice.get_text(" ", strip=True)

            if re.search(
                r"\b(this page|this documentation|this node|"
                r"this feature|this integration)\b.{0,100}"
                r"\bdeprecated\b",
                notice_text,
                re.IGNORECASE,
            ):
                return "deprecated"

            if re.search(
                r"\bdeprecated\b", " ".join(notice.get("class", [])), re.IGNORECASE
            ):
                return "deprecated"

    return "unspecified"


# MAIN DOCUMENT PARSER


def parse_document(html, url, source_sitemap):
    """Extract structured documentation from webpage HTML."""

    soup = BeautifulSoup(html, "html.parser")

    # Extract document title before cleaning.
    heading = soup.find("h1")

    if heading:
        title = heading.get_text(" ", strip=True)

    elif soup.title:
        title = soup.title.get_text(" ", strip=True)

    else:
        title = "Untitled"

    # Remove unnecessary page elements.
    clean_document_html(soup)

    # Locate the main documentation content.
    content_area = (
        soup.select_one("article.md-content__inner")
        or soup.find("article")
        or soup.find("main")
    )

    if content_area is None:
        return None

    # Remove unwanted elements inside the content area.
    for element in content_area.find_all(
        ["script", "style", "nav", "footer", "aside", "form", "button", "noscript"]
    ):
        element.decompose()

    # Detect explicit deprecation notices before conversion.
    status = detect_document_status(content_area)

    # Convert HTML into Markdown.
    content = render_children(content_area, url)

    # Normalize excessive whitespace.
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = content.strip()

    if not content:
        return None

    # Build structured document.
    document = {
        "document_id": generate_document_id(url),
        "url": url,
        "title": title,
        "category": extract_category(source_sitemap),
        "content": content,
        "source": "n8n",
        "status": status,
        "last_modified": None,
    }

    return document
