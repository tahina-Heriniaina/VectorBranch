
from pathlib import Path

import chromadb


BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "storage" / "chroma"


class ChromaEngine:

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

   

    def create_collection(self, dimension: int, collection_name: str):
        try:
            self.client.get_collection(name=collection_name)
            raise ValueError(
                f"La collection {collection_name} existe déjà."
            )
        except ValueError:
            raise
        except Exception:
            pass

        return self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def insert_vectors(self, texts, vectors, collection_name: str):
        collection = self.client.get_collection(name=collection_name)

        if len(texts) != len(vectors):
            raise ValueError("Nombre de textes et vecteurs différent.")

        batch_size = 256

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_vectors = vectors[start:start + batch_size]

            ids = [
                str(index + 1)
                for index in range(start, start + len(batch_texts))
            ]

            collection.upsert(
                ids=ids,
                documents=batch_texts,
                embeddings=[vector.tolist() for vector in batch_vectors]
            )

    def search(self, vector, limit=5, collection_name=None):
        if not collection_name:
            raise ValueError("Collection Chroma nécessaire.")

        collection = self.client.get_collection(name=collection_name)

        response = collection.query(
            query_embeddings=[vector.tolist()],
            n_results=limit,
            include=["documents", "distances"]
        )

        results = []

        ids = response["ids"][0]
        documents = response["documents"][0]
        distances = response["distances"][0]

        for identifier, document, distance in zip(
            ids, documents, distances
        ):
            results.append({
                "id": identifier,
                "text": document or "",
                "score": round(1.0 - float(distance), 4)
            })

        return results

    def delete_collection(self, collection_name: str):
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
