
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generate numerical embeddings from documentation text.
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully.")

    def generate_embedding(self, text):
        """
        Convert one text into a numerical vector.
        """

        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def generate_embeddings(self, texts):
        """
        Convert multiple texts into numerical vectors.
        """

        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings.tolist()

    @property
    def embedding_dimension(self):
        """
        Return the number of dimensions in each embedding.
        """

        return self.model.get_embedding_dimension()
