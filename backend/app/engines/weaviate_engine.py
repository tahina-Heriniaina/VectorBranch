
import re
import weaviate

from weaviate.classes.config import Configure
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery


class WeaviateEngine:

    def __init__(self):
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8081,
            grpc_port=50051
        )

    def _name(self, collection_name: str):
        prefix = "Vectorbench"

        # Éviter de répéter le préfixe
        if collection_name.lower().startswith(prefix.lower()):
            collection_name = collection_name[len(prefix):]

        parts = re.findall(r"[a-zA-Z0-9]+", collection_name)

        if not parts:
            raise ValueError("Nom de collection invalide.")

        return prefix + "".join(
            part[0].upper() + part[1:]
            for part in parts
        )

    def create_collection(self, dimension: int, collection_name: str):
        name = self._name(collection_name)

        if dimension <= 0:
            raise ValueError(
                "La dimension des vecteurs doit être positive."
            )

        if self.client.collections.exists(name):
            raise ValueError(f"La collection {name} existe déjà.")

        self.client.collections.create(
            name=name,
            vector_config=Configure.Vectors.self_provided()
        )

        return name

    def insert_vectors(
        self,
        texts,
        vectors,
        collection_name: str,
        doc_ids=None
    ):
        name = self._name(collection_name)
        collection = self.client.collections.get(name)

        if len(texts) != len(vectors):
            raise ValueError("Nombre de textes et vecteurs différent.")

        if doc_ids is None:
            doc_ids = range(1, len(texts) + 1)
        elif len(doc_ids) != len(texts):
            raise ValueError("Nombre d'identifiants et de textes différent.")

        objects = [
            DataObject(
                properties={
                    "text": text,
                    "doc_id": doc_id
                },
                vector=(
                    vector.tolist()
                    if hasattr(vector, "tolist")
                    else vector
                )
            )
            for doc_id, text, vector in zip(doc_ids, texts, vectors)
        ]

        batch_size = 100
        for start in range(0, len(objects), batch_size):
            response = collection.data.insert_many(
                objects[start:start + batch_size]
            )

            if response.has_errors:
                raise RuntimeError(
                    f"Erreur insertion Weaviate : {response.errors}"
                )

    def search(self, vector, limit=5, collection_name=None):
        if not collection_name:
            raise ValueError("Collection Weaviate nécessaire.")

        name = self._name(collection_name)

        if not self.client.collections.exists(name):
            raise ValueError(f"Collection {name} introuvable.")

        collection = self.client.collections.get(name)

        response = collection.query.near_vector(
            near_vector=vector.tolist(),
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )

        results = []

        for item in response.objects:
            distance = item.metadata.distance or 0.0

            results.append({
                "id": item.properties.get("doc_id", str(item.uuid)),
                "text": item.properties.get("text", ""),
                "score": round(1.0 - float(distance), 4)
            })

        return results

    def delete_collection(self, collection_name: str):
        name = self._name(collection_name)

        if self.client.collections.exists(name):
            self.client.collections.delete(name)

    def close(self):
        self.client.close()
