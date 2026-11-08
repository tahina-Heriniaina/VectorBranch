from qdrant_client import QdrantClient, models


class QdrantEngine:
    def __init__(self):
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "vectorbench_demo"

    def create_collection(self, dimension: int, collection_name=None):
        name = collection_name or self.collection_name

        if self.client.collection_exists(name):
            if collection_name is not None:
                raise ValueError(f"La collection {name} existe déjà.")

            # Comportement conservé pour la démonstration initiale.
            self.client.delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE
            )
        )

    def insert_vectors(self, texts, vectors, collection_name=None):
        name = collection_name or self.collection_name
        batch_size = 256

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_vectors = vectors[start:start + batch_size]

            points = []

            for index, (text, vector) in enumerate(
                zip(batch_texts, batch_vectors),
                start=start
            ):
                points.append(
                    models.PointStruct(
                        id=index + 1,
                        vector=vector.tolist(),
                        payload={"text": text}
                    )
                )

            self.client.upsert(
                collection_name=name,
                points=points,
                wait=True
            )

    def search(self, vector, limit=5, collection_name=None):
        name = collection_name or self.collection_name

        results = self.client.query_points(
            collection_name=name,
            query=vector.tolist(),
            limit=limit,
            with_payload=True
        )

        return results.points