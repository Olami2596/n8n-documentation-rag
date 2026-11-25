import json
from pathlib import Path

CHUNKS_FILE = Path("data/processed/n8n_full_embedded_chunks.json")
TARGET_URL = "https://docs.n8n.io/connect/n8n-api/build-and-run/executions"


def main():
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    execution_chunks = [chunk for chunk in chunks if chunk.get("url") == TARGET_URL]

    print(f"\nChunks from the Executions page: {len(execution_chunks)}")
    print("=" * 75)

    for chunk in execution_chunks:
        print(f"\nChunk index: {chunk.get('chunk_index')}")
        print(f"Start index: {chunk.get('start_index')}")
        print(f"Chunk ID: {chunk.get('chunk_id')}")
        print("-" * 75)
        print(chunk.get("content", ""))
        print("=" * 75)


if __name__ == "__main__":
    main()
