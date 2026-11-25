import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


class PineconeStore:
    """
    Manage the n8n documentation vector index.
    """

    def __init__(
        self,
        index_name="n8n-documentation",
        dimension=384,
        metric="cosine"
    ):
        load_dotenv()

        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise ValueError("PINECONE_API_KEY is missing.")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric

    def create_index(self):
        """
        Create the Pinecone index if it doesn't exist.
        """

        existing_indexes = [
            index.name for index in self.pc.indexes.list()
        ]

        if self.index_name in existing_indexes:
            print(f"Index '{self.index_name}' already exists.")
            return

        print(f"Creating index: {self.index_name}")

        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Waiting for index to become ready...")

        while not self.pc.describe_index(
            self.index_name
        ).status["ready"]:
            time.sleep(2)

        print("Pinecone index created successfully.")

    def connect(self):
        """
        Connect to the existing index.
        """

        self.index = self.pc.Index(self.index_name)

        return self.index
