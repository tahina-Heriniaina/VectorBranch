
import os
import re

from pymilvus import MilvusClient


class MilvusEngine:
    """
    Adaptateur VectorBench pour Milvus.
    Opérations : création, insertion, recherche et suppression.
    """

    def __init__(self):
        self.uri = os.getenv(
            "MILVUS_URI",
            "http://localhost:19530"
        )

        self.client = MilvusClient(uri=self.uri)

    def _validate_name(self, collection_name):
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", collection_name):
            raise ValueError("Nom de collection Milvus invalide.")

        return collection_name

    def create_collection(self, dimension: int, collection_name: str):
        name = self._validate_name(collection_name)

        if self.client.has_collection(collection_name=name):
            raise ValueError(
                f"La collection Milvus {name} existe déjà."
            )

        self.client.create_collection(
            collection_name=name,
            dimension=dimension,
            primary_field_name="id",
            id_type="int",
            vector_field_name="vector",
            metric_type="COSINE",
            auto_id=False
        )

        return name

    def insert_vectors(self, texts, vectors, collection_name: str):
        name = self._validate_name(collection_name)

        if len(texts) != len(vectors):
            raise ValueError(
                "Le nombre de textes et de vecteurs est différent."
            )

        batch_size = 256

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_vectors = vectors[start:start + batch_size]

            data = []

            for index, (text, vector) in enumerate(
                zip(batch_texts, batch_vectors),
                start=start
            ):
                data.append({
                    "id": index + 1,
                    "vector": vector.tolist(),
                    "text": text
                })

            self.client.insert(
                collection_name=name,
                data=data
            )

    def search(self, vector, limit=5, collection_name=None):
        if not collection_name:
            raise ValueError("Une collection Milvus est nécessaire.")

        name = self._validate_name(collection_name)

        if not self.client.has_collection(collection_name=name):
            raise ValueError(
                f"La collection Milvus {name} n'existe pas."
            )

        results = self.client.search(
            collection_name=name,
            data=[vector.tolist()],
            anns_field="vector",
            limit=limit,
            search_params={
                "metric_type": "COSINE",
                "params": {}
            },
            output_fields=["text"]
        )

        formatted_results = []

        for hit in results[0]:
            entity = hit.get("entity", {})

            formatted_results.append({
                "id": hit["id"],
                "text": entity.get("text", ""),
                "score": round(float(hit["distance"]), 4)
            })

        return formatted_results

    def delete_collection(self, collection_name: str):
        name = self._validate_name(collection_name)

        if self.client.has_collection(collection_name=name):
            self.client.drop_collection(collection_name=name)
