import os

from dotenv import load_dotenv


def main():
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY was not found in your .env file."
        )

    print("Pinecone API key loaded successfully.")
    print("API key is configured.")
    print("Key preview:", api_key[:4] + "...")


if __name__ == "__main__":
    main()
