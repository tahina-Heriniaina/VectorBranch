
import json
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from app.engines.engine_manager import engine_manager
from app.services.runtime import get_embedding_model
from app.services.dataset_service import DATASETS_DIR


BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR.parent / "results" / "benchmarks"


class BenchmarkService:

    def run(
        self,
        dataset_id: str,
        queries: list[str],
        top_k: int = 10,
        repetitions: int = 10,
        engine_name: str = "qdrant"
    ):
        if not dataset_id.isalnum():
            raise ValueError("Identifiant de dataset invalide.")

        if not queries or any(
            not isinstance(query, str) or not query.strip()
            for query in queries
        ):
            raise ValueError("Les requêtes ne doivent pas être vides.")

        if not 1 <= top_k <= 100:
            raise ValueError("top_k doit être compris entre 1 et 100.")

        if not 1 <= repetitions <= 100:
            raise ValueError(
                "repetitions doit être compris entre 1 et 100."
            )

        dataset_dir = DATASETS_DIR / dataset_id
        metadata_file = dataset_dir / "metadata.json"
        embeddings_file = dataset_dir / "embeddings.npy"
        documents_file = dataset_dir / "documents.jsonl"

        if not all(
            path.is_file()
            for path in [
                metadata_file,
                embeddings_file,
                documents_file
            ]
        ):
            raise ValueError(
                f"Dataset '{dataset_id}' introuvable ou incomplet."
            )

        with open(metadata_file, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        engine_key = engine_manager.normalize_name(engine_name)

        engine_collections = metadata.get("engine_collections", {})

        # Compatibilité avec les anciens datasets Qdrant
        collection_name = engine_collections.get(engine_key)

        if not collection_name and engine_key == "qdrant":
            collection_name = metadata.get("collection_name")

        if not collection_name:
            engine_error = metadata.get("engine_errors", {}).get(
                engine_key
            )

            raise ValueError(
                f"Le dataset n'est pas disponible dans {engine_key}. "
                f"Erreur d'indexation : {engine_error or 'aucun index trouvé'}"
            )

        vectors = np.load(embeddings_file, allow_pickle=False)

        with open(documents_file, "r", encoding="utf-8") as file:
            documents = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if len(documents) != len(vectors):
            raise ValueError(
                "Le nombre de documents et de vecteurs diffère."
            )

        k = min(top_k, len(documents))

        model = get_embedding_model()

        query_vectors = model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        # Vecteurs normalisés pour le calcul exact de référence
        vectors = vectors.astype(np.float32)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)

        engine = engine_manager.get_engine(engine_key)

        latencies = []
        recalls = []

        process = psutil.Process()

        # Associer le texte à son identifiant documentaire
        text_to_id = {
            item["text"]: int(item["id"])
            for item in documents
        }

        for query_vector in query_vectors:

            # Référence exacte par produit scalaire sur vecteurs normalisés
            similarities = vectors @ query_vector

            exact_indices = np.argsort(-similarities)[:k]

            exact_ids = {
                int(index) + 1
                for index in exact_indices
            }

            # Échauffement exclu des mesures
            engine.search(
                vector=query_vector,
                limit=k,
                collection_name=collection_name
            )

            for _ in range(repetitions):

                start = time.perf_counter()

                results = engine.search(
                    vector=query_vector,
                    limit=k,
                    collection_name=collection_name
                )

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                latencies.append(elapsed_ms)

                returned_ids = set()

                for result in results:
                    if isinstance(result, dict):
                        identifier = result.get("id")
                        text = result.get("text", "")
                    else:
                        identifier = result.id
                        text = result.payload.get("text", "")

                    try:
                        returned_ids.add(int(identifier))
                    except (TypeError, ValueError):
                        # Compatibilité temporaire avec les UUID Weaviate
                        if text in text_to_id:
                            returned_ids.add(text_to_id[text])

                recall = (
                    len(exact_ids.intersection(returned_ids)) / k
                )

                recalls.append(recall)

        total_seconds = sum(latencies) / 1000

        average_latency = float(np.mean(latencies))

        qps = (
            len(latencies) / total_seconds
            if total_seconds > 0
            else 0
        )

        memory_mb = (
            process.memory_info().rss / (1024 * 1024)
        )

        benchmark_id = uuid.uuid4().hex[:12]

        indexing_time = metadata.get(
            "indexing_times_ms", {}
        ).get(
            engine_key,
            metadata.get("indexing_time_ms", 0)
        )

        result = {
            "benchmark_id": benchmark_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine": engine_key.capitalize(),
            "engine_key": engine_key,
            "dataset_id": dataset_id,
            "document_count": len(documents),
            "query_count": len(queries),
            "top_k": k,
            "repetitions": repetitions,
            "indexing_time_ms": indexing_time,
            "metrics": {
                "recall_at_k": round(float(np.mean(recalls)), 4),
                "average_latency_ms": round(average_latency, 3),
                "p50_latency_ms": round(
                    float(np.percentile(latencies, 50)), 3
                ),
                "p95_latency_ms": round(
                    float(np.percentile(latencies, 95)), 3
                ),
                "p99_latency_ms": round(
                    float(np.percentile(latencies, 99)), 3
                ),
                "qps": round(float(qps), 2),
                "backend_process_ram_mb": round(memory_mb, 2)
            }
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        result_file = RESULTS_DIR / f"{benchmark_id}.json"

        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=4)

        return result

    def list_results(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        results = []

        for result_file in RESULTS_DIR.glob("*.json"):
            with open(result_file, "r", encoding="utf-8") as file:
                results.append(json.load(file))

        return sorted(
            results,
            key=lambda item: item["created_at"],
            reverse=True
        )
