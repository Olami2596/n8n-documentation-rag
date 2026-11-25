import os
from importlib.metadata import version

from dotenv import load_dotenv
from pinecone import Pinecone


def main():

    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing.")

    print("Pinecone SDK version:", version("pinecone"))

    print("\nConnecting to Pinecone...")

    pc = Pinecone(api_key=api_key)

    indexes = list(pc.indexes.list())

    print("Successfully connected to Pinecone!")
    print(f"Existing indexes: {len(indexes)}")

    if indexes:
        print("\nYour indexes:")

        for index in indexes:
            print("-", index.name)
    else:
        print("No indexes exist yet.")


if __name__ == "__main__":
    main()
