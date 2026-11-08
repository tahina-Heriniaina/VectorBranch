
import json
import re

from pathlib import Path

import faiss
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
FAISS_DIR = BASE_DIR / "storage" / "faiss"


class FaissEngine:

    def __init__(self):
        FAISS_DIR.mkdir(parents=True, exist_ok=True)

    def _paths(self, collection_name):
        if not re.fullmatch(r"[a-zA-Z0-9_]+", collection_name):
            raise ValueError("Nom d'index FAISS invalide.")

        return (
            FAISS_DIR / f"{collection_name}.index",
            FAISS_DIR / f"{collection_name}.json"
        )

    def create_collection(self, dimension: int, collection_name: str):
        index_path, documents_path = self._paths(collection_name)

        if index_path.exists():
            raise ValueError(
                f"L'index {collection_name} existe déjà."
            )

        index = faiss.IndexIDMap2(
            faiss.IndexFlatIP(dimension)
        )

        faiss.write_index(index, str(index_path))

        with open(documents_path, "w", encoding="utf-8") as file:
            json.dump({
                "dimension": dimension,
                "documents": {}
            }, file, ensure_ascii=False, indent=2)

        return collection_name

    def insert_vectors(self, texts, vectors, collection_name: str):
        index_path, documents_path = self._paths(collection_name)

        if not index_path.exists() or not documents_path.exists():
            raise ValueError("Index FAISS introuvable.")

        if len(texts) != len(vectors):
            raise ValueError("Nombre de textes et vecteurs différent.")

        index = faiss.read_index(str(index_path))

        with open(documents_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        matrix = np.asarray(vectors, dtype=np.float32).copy()

        if matrix.shape[1] != metadata["dimension"]:
            raise ValueError("Dimension des vecteurs incorrecte.")

        faiss.normalize_L2(matrix)

        start = index.ntotal
        ids = np.arange(
            start + 1,
            start + len(texts) + 1,
            dtype=np.int64
        )

        index.add_with_ids(matrix, ids)

        for identifier, text in zip(ids, texts):
            metadata["documents"][str(int(identifier))] = text

        faiss.write_index(index, str(index_path))

        with open(documents_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    def search(self, vector, limit=5, collection_name=None):
        if not collection_name:
            raise ValueError("Index FAISS nécessaire.")

        index_path, documents_path = self._paths(collection_name)

        if not index_path.exists() or not documents_path.exists():
            raise ValueError("Index FAISS introuvable.")

        index = faiss.read_index(str(index_path))

        with open(documents_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        query = np.asarray(
            [vector],
            dtype=np.float32
        )

        faiss.normalize_L2(query)

        scores, ids = index.search(query, limit)

        results = []

        for score, identifier in zip(scores[0], ids[0]):
            if identifier == -1:
                continue

            results.append({
                "id": int(identifier),
                "text": metadata["documents"].get(
                    str(int(identifier)), ""
                ),
                "score": round(float(score), 4)
            })

        return results

    def delete_collection(self, collection_name: str):
        index_path, documents_path = self._paths(collection_name)

        if index_path.exists():
            index_path.unlink()

        if documents_path.exists():
            documents_path.unlink()
