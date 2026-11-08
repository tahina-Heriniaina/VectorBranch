
import os
import re

from pinecone import Pinecone, ServerlessSpec


class PineconeEngine:

    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise ValueError(
                "La variable PINECONE_API_KEY est absente."
            )

        self.index_name = os.getenv(
            "PINECONE_INDEX",
            "vectorbench"
        )

        self.pc = Pinecone(api_key=api_key)

        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv(
            "PINECONE_REGION",
            "us-east-1"
        )

        self.index = None

    def _namespace(self, collection_name):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", collection_name):
            raise ValueError("Namespace Pinecone invalide.")

        return collection_name

    def create_collection(self, dimension: int, collection_name: str):
        namespace = self._namespace(collection_name)

        existing_indexes = self.pc.list_indexes().names()

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )

        description = self.pc.describe_index(self.index_name)

        if description.dimension != dimension:
            raise ValueError(
                "La dimension de l'index Pinecone ne correspond pas "
                "à celle des embeddings."
            )

        self.index = self.pc.Index(self.index_name)

        return namespace

    def insert_vectors(self, texts, vectors, collection_name: str):
        namespace = self._namespace(collection_name)

        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        if len(texts) != len(vectors):
            raise ValueError("Nombre de textes et vecteurs différent.")

        batch_size = 100

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_vectors = vectors[start:start + batch_size]

            records = []

            for index, (text, vector) in enumerate(
                zip(batch_texts, batch_vectors),
                start=start
            ):
                records.append({
                    "id": str(index + 1),
                    "values": vector.tolist(),
                    "metadata": {"text": text}
                })

            self.index.upsert(
                vectors=records,
                namespace=namespace
            )

    def search(self, vector, limit=5, collection_name=None):
        if not collection_name:
            raise ValueError("Namespace Pinecone nécessaire.")

        namespace = self._namespace(collection_name)

        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        response = self.index.query(
            vector=vector.tolist(),
            top_k=limit,
            namespace=namespace,
            include_metadata=True
        )

        results = []

        for match in response.matches:
            metadata = match.metadata or {}

            results.append({
                "id": match.id,
                "text": metadata.get("text", ""),
                "score": round(float(match.score), 4)
            })

        return results

    def delete_collection(self, collection_name: str):
        namespace = self._namespace(collection_name)

        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        self.index.delete(
            delete_all=True,
            namespace=namespace
        )
