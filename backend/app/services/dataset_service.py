
import csv
import io
import json
import uuid
import time

from pathlib import Path

import numpy as np

from app.engines.engine_manager import engine_manager
from app.services.runtime import get_embedding_model


BASE_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = BASE_DIR / "datasets" / "imported"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)


class DatasetService:

    def parse_dataset(self, filename: str, content: bytes):
        extension = Path(filename).suffix.lower()

        try:
            decoded = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(
                "Le fichier doit être encodé en UTF-8."
            ) from error

        documents = []

        if extension == ".csv":
            reader = csv.DictReader(io.StringIO(decoded))

            if not reader.fieldnames or "text" not in reader.fieldnames:
                raise ValueError(
                    "Le fichier CSV doit contenir une colonne text."
                )

            documents = [
                row["text"].strip()
                for row in reader
                if row.get("text") and row["text"].strip()
            ]

        elif extension == ".jsonl":
            for line_number, line in enumerate(decoded.splitlines(), 1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON invalide à la ligne {line_number}."
                    ) from error

                if not isinstance(item, dict):
                    raise ValueError(
                        f"La ligne {line_number} doit contenir un objet JSON."
                    )

                text = item.get("text")

                if isinstance(text, str) and text.strip():
                    documents.append(text.strip())

        elif extension == ".txt":
            documents = [
                line.strip()
                for line in decoded.splitlines()
                if line.strip()
            ]

        else:
            raise ValueError(
                "Format non accepté. Utilise CSV, JSONL ou TXT."
            )

        if not documents:
            raise ValueError("Le dataset ne contient aucun texte valide.")

        if len(documents) > 100_000:
            raise ValueError(
                "Limite initiale : 100 000 documents par importation."
            )

        return documents

    def import_dataset(self, filename: str, content: bytes):
        if len(content) > 20 * 1024 * 1024:
            raise ValueError("Le fichier ne doit pas dépasser 20 Mo.")

        documents = self.parse_dataset(filename, content)

        dataset_id = uuid.uuid4().hex[:12]
        collection_name = f"dataset_{dataset_id}"

        dataset_dir = DATASETS_DIR / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Génération commune des embeddings
        model = get_embedding_model()

        start_embedding = time.perf_counter()

        vectors = model.encode(
            documents,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        embedding_time_ms = (
            time.perf_counter() - start_embedding
        ) * 1000

        # Sauvegarde locale des données communes
        documents_file = dataset_dir / "documents.jsonl"

        with open(documents_file, "w", encoding="utf-8") as file:
            for index, document in enumerate(documents, start=1):
                file.write(
                    json.dumps(
                        {"id": index, "text": document},
                        ensure_ascii=False
                    ) + "\n"
                )

        np.save(dataset_dir / "embeddings.npy", vectors)

        # Indexation dans chaque moteur
        engine_collections = {}
        indexing_times_ms = {}
        engine_errors = {}

        for engine_name in engine_manager.list_engines():
            engine = None
            created_collection = None

            try:
                engine = engine_manager.get_engine(engine_name)

                start_indexing = time.perf_counter()

                created_collection = engine.create_collection(
                    dimension=model.get_sentence_embedding_dimension(),
                    collection_name=collection_name
                )

                if not created_collection:
                    created_collection = collection_name

                engine.insert_vectors(
                    texts=documents,
                    vectors=vectors,
                    collection_name=created_collection
                )

                elapsed_ms = (
                    time.perf_counter() - start_indexing
                ) * 1000

                engine_collections[engine_name] = created_collection
                indexing_times_ms[engine_name] = round(elapsed_ms, 2)

            except Exception as error:
                engine_errors[engine_name] = str(error)

                if engine is not None and created_collection is not None:
                    try:
                        engine.delete_collection(created_collection)
                    except Exception as cleanup_error:
                        engine_errors[engine_name] += (
                            f" | Erreur nettoyage : {cleanup_error}"
                        )

        if not engine_collections:
            raise RuntimeError(
                "Aucun moteur n'a réussi l'indexation. "
                f"Erreurs : {engine_errors}"
            )

        metadata = {
            "dataset_id": dataset_id,
            "filename": filename,
            "collection_name": collection_name,
            "engine_collections": engine_collections,
            "engine_errors": engine_errors,
            "document_count": len(documents),
            "dimension": int(vectors.shape[1]),
            "distance": "Cosine",
            "embedding_time_ms": round(embedding_time_ms, 2),
            "indexing_times_ms": indexing_times_ms,
            # Compatibilité avec l'ancien frontend Qdrant
            "indexing_time_ms": indexing_times_ms.get("qdrant", 0)
        }

        with open(
            dataset_dir / "metadata.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(metadata, file, ensure_ascii=False, indent=4)

        return metadata

    def get_metadata(self, dataset_id: str):
        if not dataset_id.isalnum():
            raise ValueError("Identifiant de dataset invalide.")

        dataset_dir = DATASETS_DIR / dataset_id
        metadata_file = dataset_dir / "metadata.json"

        if not dataset_dir.is_dir() or not metadata_file.is_file():
            raise ValueError(f"Dataset introuvable : {dataset_id}")

        with open(metadata_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def ensure_engine_index(self, dataset_id: str, engine_name: str):
        metadata = self.get_metadata(dataset_id)
        engine_key = engine_manager.normalize_name(engine_name)
        engine_collections = metadata.setdefault("engine_collections", {})

        if engine_collections.get(engine_key):
            return metadata

        files = self.get_dataset_files(dataset_id)
        vectors = files["embeddings"]
        documents = files["documents"]
        collection_name = metadata["collection_name"]
        engine = engine_manager.get_engine(engine_key)
        created_collection = None
        created_here = False
        already_indexed = False
        missing_indices = list(range(len(documents)))
        start_indexing = time.perf_counter()

        try:
            if engine_key == "weaviate":
                resolved_name = engine._name(collection_name)
                if engine.client.collections.exists(resolved_name):
                    created_collection = resolved_name
                    collection = engine.client.collections.get(resolved_name)
                    indexed_objects = collection.query.fetch_objects(
                        limit=len(documents),
                        return_properties=["doc_id"]
                    )
                    indexed_ids = {
                        int(item.properties["doc_id"])
                        for item in indexed_objects.objects
                        if item.properties.get("doc_id") is not None
                    }
                    expected_ids = set(range(1, len(documents) + 1))
                    already_indexed = expected_ids.issubset(indexed_ids)

                    missing_indices = [
                        index
                        for index in missing_indices
                        if index + 1 not in indexed_ids
                    ]

            if created_collection is None:
                created_collection = engine.create_collection(
                    dimension=int(vectors.shape[1]),
                    collection_name=collection_name
                )
                created_here = True

                if not created_collection:
                    created_collection = collection_name

            if not already_indexed:
                texts_to_insert = [
                    documents[index]["text"]
                    for index in missing_indices
                ]
                vectors_to_insert = vectors[missing_indices]

                if engine_key == "weaviate":
                    engine.insert_vectors(
                        texts=texts_to_insert,
                        vectors=vectors_to_insert,
                        collection_name=created_collection,
                        doc_ids=[index + 1 for index in missing_indices]
                    )
                else:
                    engine.insert_vectors(
                        texts=texts_to_insert,
                        vectors=vectors_to_insert,
                        collection_name=created_collection
                    )
        except Exception as error:
            if created_here and created_collection is not None:
                try:
                    engine.delete_collection(created_collection)
                except Exception as cleanup_error:
                    error = RuntimeError(
                        f"{error} | Erreur nettoyage : {cleanup_error}"
                    )

            engine_errors = metadata.setdefault("engine_errors", {})
            engine_errors[engine_key] = str(error)
            metadata_file = DATASETS_DIR / dataset_id / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=False, indent=4)

            raise ValueError(
                f"Impossible d'indexer le dataset dans {engine_key} : {error}"
            ) from error

        indexing_time_ms = (
            time.perf_counter() - start_indexing
        ) * 1000
        engine_collections[engine_key] = created_collection
        metadata.setdefault("engine_errors", {}).pop(engine_key, None)
        metadata.setdefault("indexing_times_ms", {})[engine_key] = round(
            indexing_time_ms, 2
        )
        if engine_key == "qdrant":
            metadata["indexing_time_ms"] = round(indexing_time_ms, 2)

        metadata_file = DATASETS_DIR / dataset_id / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=4)

        return metadata

    def list_datasets(self):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)

        datasets = []

        for metadata_file in DATASETS_DIR.glob("*/metadata.json"):
            with open(metadata_file, "r", encoding="utf-8") as file:
                datasets.append(json.load(file))

        return datasets

    def get_dataset_files(self, dataset_id: str):
        metadata = self.get_metadata(dataset_id)

        dataset_dir = DATASETS_DIR / dataset_id

        documents_file = dataset_dir / "documents.jsonl"
        embeddings_file = dataset_dir / "embeddings.npy"

        if not documents_file.is_file():
            raise ValueError("Fichier documents.jsonl introuvable.")

        if not embeddings_file.is_file():
            raise ValueError("Fichier embeddings.npy introuvable.")

        embeddings = np.load(embeddings_file, allow_pickle=False)

        if embeddings.ndim != 2:
            raise ValueError("Format des embeddings invalide.")

        if embeddings.shape[0] != metadata["document_count"]:
            raise ValueError(
                "Nombre d'embeddings différent du nombre de documents."
            )

        with open(documents_file, "r", encoding="utf-8") as file:
            documents = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        return {
            "metadata": metadata,
            "documents_file": documents_file,
            "embeddings_file": embeddings_file,
            "embeddings": embeddings,
            "documents": documents
        }
