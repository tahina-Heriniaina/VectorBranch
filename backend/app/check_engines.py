import os

def check(name, function):
    try:
        result = function()
        print(f"[OK] {name} : {result}")
    except Exception as error:
        print(f"[ERREUR] {name} : {error}")


def check_milvus():
    from pymilvus import MilvusClient

    client = MilvusClient(uri="http://localhost:19530")
    collections = client.list_collections()
    return f"Connecté, {len(collections)} collection(s)"


def check_weaviate():
    import weaviate

    client = weaviate.connect_to_local(
        host="localhost",
        port=8081,
        grpc_port=50051
    )

    try:
        return f"Prêt : {client.is_ready()}"
    finally:
        client.close()


def check_chroma():
    import chromadb

    client = chromadb.PersistentClient(
        path="./storage/chroma"
    )

    return f"Connecté, {len(client.list_collections())} collection(s)"


def check_faiss():
    import faiss

    index = faiss.IndexFlatIP(384)

    return f"Index créé, dimension={index.d}"


def check_pgvector():
    import psycopg
    from pgvector.psycopg import register_vector

    with psycopg.connect(
        "host=localhost port=5433 "
        "dbname=vectorbench_vectors "
        "user=vectorbench password=vectorbench_dev_2026"
    ) as connection:

        register_vector(connection)

        version = connection.execute(
            "SELECT extversion FROM pg_extension "
            "WHERE extname = 'vector'"
        ).fetchone()

        if version is None:
            raise RuntimeError(
                "Extension pgvector non installée."
            )

        return f"Extension activée, version {version[0]}"


def check_pinecone():
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Clé API manquante : configure PINECONE_API_KEY."
        )

    client = Pinecone(api_key=api_key)
    indexes = client.list_indexes()

    return f"Connecté, {len(indexes)} index"


if __name__ == "__main__":
    print("\n===== VECTORBENCH : VÉRIFICATION DES MOTEURS =====\n")

    check("Milvus", check_milvus)
    check("Weaviate", check_weaviate)
    check("Chroma", check_chroma)
    check("FAISS", check_faiss)
    check("pgvector", check_pgvector)
    check("Pinecone", check_pinecone)