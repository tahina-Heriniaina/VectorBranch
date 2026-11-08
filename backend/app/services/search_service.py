
from app.engines.engine_manager import engine_manager
from app.services.runtime import get_embedding_model


class SearchService:

    def __init__(self):
        self.engines = engine_manager

    def get_engine_status(self, engine_name="qdrant"):
        try:
            engine_key = self.engines.normalize_name(engine_name)
            engine = self.engines.get_engine(engine_key)

            if engine_key == "qdrant":
                collections = engine.client.get_collections()

                names = [
                    item.name
                    for item in collections.collections
                ]

                return {
                    "engine": engine_key,
                    "connected": True,
                    "collections": names
                }

            if engine_key == "weaviate":
                return {
                    "engine": engine_key,
                    "connected": engine.client.is_ready()
                }

            if engine_key == "milvus":
                collections = engine.client.list_collections()
                return {
                    "engine": engine_key,
                    "connected": collections is not None,
                    "collections": collections
                }

            if engine_key == "pinecone":
                return {
                    "engine": engine_key,
                    "connected": True,
                    "indexes": engine.pc.list_indexes().names()
                }

            return {
                "engine": engine_key,
                "connected": True
            }

        except Exception as error:
            return {
                "engine": engine_name.lower(),
                "connected": False,
                "error": str(error)
            }

    def search(
        self,
        query: str,
        limit: int = 5,
        collection_name=None,
        engine_name="qdrant"
    ):
        try:
            engine = self.engines.get_engine(engine_name)

            if not collection_name:
                if engine_name.lower() == "qdrant":
                    collection_name = engine.collection_name
                else:
                    raise ValueError(
                        "Importe d'abord un dataset pour ce moteur."
                    )

            model = get_embedding_model()
            query_vector = model.encode(
                [query],
                normalize_embeddings=True
            )[0]

            results = engine.search(
                vector=query_vector,
                limit=limit,
                collection_name=collection_name
            )

            # Qdrant renvoie des objets PointStruct.
            # Les autres adaptateurs renvoient déjà des dictionnaires.
            formatted_results = []

            for result in results:
                if isinstance(result, dict):
                    formatted_results.append(result)
                else:
                    formatted_results.append({
                        "id": result.id,
                        "text": result.payload.get("text", ""),
                        "score": round(float(result.score), 4)
                    })

            return formatted_results

        except ValueError:
            raise

        except Exception as error:
            raise ConnectionError(
                f"Erreur de recherche avec {engine_name} : {error}"
            ) from error
