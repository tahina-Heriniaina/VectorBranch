
from app.engines.qdrant_engine import QdrantEngine
from app.engines.milvus_engine import MilvusEngine
from app.engines.weaviate_engine import WeaviateEngine
from app.engines.chroma_engine import ChromaEngine
from app.engines.faiss_engine import FaissEngine
from app.engines.pgvector_engine import PgVectorEngine
from app.engines.pinecone_engine import PineconeEngine


class EngineManager:
    """
    Gestionnaire central des sept moteurs vectoriels.
    Les moteurs sont initialisés uniquement lorsqu'ils sont demandés.
    """

    ENGINE_CLASSES = {
        "qdrant": QdrantEngine,
        "milvus": MilvusEngine,
        "weaviate": WeaviateEngine,
        "chroma": ChromaEngine,
        "faiss": FaissEngine,
        "pgvector": PgVectorEngine,
        "pinecone": PineconeEngine,
    }

    def __init__(self):
        self.instances = {}

    def normalize_name(self, engine_name: str) -> str:
        name = engine_name.strip().lower()

        aliases = {
            "postgres": "pgvector",
            "postgresql": "pgvector",
        }

        name = aliases.get(name, name)

        if name not in self.ENGINE_CLASSES:
            raise ValueError(
                f"Moteur inconnu : {engine_name}. "
                f"Moteurs disponibles : {', '.join(self.ENGINE_CLASSES)}"
            )

        return name

    def get_engine(self, engine_name: str = "qdrant"):
        name = self.normalize_name(engine_name)

        if name not in self.instances:
            engine_class = self.ENGINE_CLASSES[name]
            self.instances[name] = engine_class()

        return self.instances[name]

    def list_engines(self):
        return list(self.ENGINE_CLASSES.keys())

    def close(self):
        for engine in self.instances.values():
            close_method = getattr(engine, "close", None)

            if callable(close_method):
                close_method()

        self.instances.clear()



engine_manager = EngineManager()
