
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_splitter(
    chunk_size=800,
    chunk_overlap=120
):
    """
    Create a recursive text splitter for ordinary documentation text.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )


def is_markdown_table_line(line):
    """
    Identify a Markdown table row.
    """

    stripped = line.strip()

    return stripped.startswith("|") and stripped.count("|") >= 2


def split_markdown_table(table_lines, start_index, chunk_size):
    """
    Split a Markdown table by rows while repeating its header.

    An individual row is never split in the middle.
    """

    if not table_lines:
        return []

    # The first line is the header.
    # The second line is normally the Markdown separator.
    if (
        len(table_lines) >= 2
        and "---" in table_lines[1]
    ):
        header_lines = table_lines[:2]
        data_lines = table_lines[2:]
    else:
        header_lines = table_lines[:1]
        data_lines = table_lines[1:]

    header = "".join(header_lines)

    # Keep small tables together.
    complete_table = "".join(table_lines).rstrip()

    if len(complete_table) <= chunk_size:
        return [{
            "content": complete_table,
            "start_index": start_index
        }]

    results = []
    current_rows = []
    current_length = len(header)

    row_start_positions = []
    current_position = start_index + len(header)

    for row in data_lines:

        row_length = len(row)

        # Flush the current fragment before adding a row
        # that would exceed the target size.
        if current_rows and current_length + row_length > chunk_size:

            results.append({
                "content": (
                    header + "".join(current_rows)
                ).rstrip(),
                "start_index": row_start_positions[0]
            })

            current_rows = []
            current_length = len(header)

        if not current_rows:
            row_start_positions = [current_position]

        current_rows.append(row)
        current_length += row_length

        current_position += row_length

    # Save the final group of rows.
    if current_rows:
        results.append({
            "content": (
                header + "".join(current_rows)
            ).rstrip(),
            "start_index": row_start_positions[0]
        })

    # A table containing only a header still produces a chunk.
    if not results and header.strip():
        results.append({
            "content": header.rstrip(),
            "start_index": start_index
        })

    return results


def split_document_content(content, splitter, chunk_size):
    """
    Split document content while treating Markdown tables separately.

    Ordinary text uses RecursiveCharacterTextSplitter.
    Tables are divided by rows with repeated headers.
    """

    lines = content.splitlines(keepends=True)

    segments = []
    prose_lines = []

    current_position = 0
    prose_start = 0
    index = 0

    while index < len(lines):

        line = lines[index]

        if is_markdown_table_line(line):

            # Process ordinary text before the table.
            if prose_lines:

                prose_text = "".join(prose_lines)

                prose_chunks = splitter.create_documents(
                    texts=[prose_text]
                )

                for chunk in prose_chunks:
                    segments.append({
                        "content": chunk.page_content,
                        "start_index": (
                            prose_start
                            + chunk.metadata["start_index"]
                        )
                    })

                prose_lines = []

            table_start = current_position
            table_lines = []

            # Collect all consecutive Markdown table rows.
            while (
                index < len(lines)
                and is_markdown_table_line(lines[index])
            ):
                table_lines.append(lines[index])
                current_position += len(lines[index])
                index += 1

            table_chunks = split_markdown_table(
                table_lines=table_lines,
                start_index=table_start,
                chunk_size=chunk_size
            )

            segments.extend(table_chunks)

            prose_start = current_position

        else:

            if not prose_lines:
                prose_start = current_position

            prose_lines.append(line)
            current_position += len(line)
            index += 1

    # Process any remaining ordinary text.
    if prose_lines:

        prose_text = "".join(prose_lines)

        prose_chunks = splitter.create_documents(
            texts=[prose_text]
        )

        for chunk in prose_chunks:
            segments.append({
                "content": chunk.page_content,
                "start_index": (
                    prose_start
                    + chunk.metadata["start_index"]
                )
            })

    return segments


def chunk_document(document, splitter, chunk_size=800):
    """
    Split one documentation document into smaller chunks.

    Each chunk retains its original document metadata.
    """

    content = document.get("content", "").strip()

    if not content:
        return []

    content_chunks = split_document_content(
        content=content,
        splitter=splitter,
        chunk_size=chunk_size
    )

    results = []

    for index, chunk in enumerate(content_chunks):

        results.append({
            "chunk_id": f"{document['document_id']}_{index}",
            "document_id": document["document_id"],
            "chunk_index": index,
            "title": document["title"],
            "url": document["url"],
            "category": document["category"],
            "source": document["source"],
            "status": document["status"],
            "content": chunk["content"],
            "start_index": chunk["start_index"]
        })

    return results


def chunk_documents(documents, chunk_size=800, chunk_overlap=120):
    """
    Split multiple documentation documents into chunks.
    """

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []

    for document in documents:

        document_chunks = chunk_document(
            document=document,
            splitter=splitter,
            chunk_size=chunk_size
        )

        all_chunks.extend(document_chunks)

    return all_chunks
