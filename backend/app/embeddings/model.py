from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        self.model = SentenceTransformer(
            self.model_name,
            device="cpu"
        )

        self.dimension = self.model.get_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True
    ):
        vectors = self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=convert_to_numpy
        )

        return vectors

    def get_sentence_embedding_dimension(self):
        return self.dimension