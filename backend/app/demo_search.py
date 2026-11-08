from app.embeddings.model import EmbeddingModel
from app.engines.qdrant_engine import QdrantEngine
from app.datasets.sample_data import DOCUMENTS


def main():

    print("Chargement du modèle IA...")

    embedding_model = EmbeddingModel()

    print("Dimension :", embedding_model.dimension)

    print("Génération des embeddings...")

    vectors = embedding_model.encode(DOCUMENTS)

    print("Connexion à Qdrant...")

    engine = QdrantEngine()

    print("Création de la collection...")

    engine.create_collection(
        embedding_model.dimension
    )

    print("Insertion des documents...")

    engine.insert_vectors(
        DOCUMENTS,
        vectors
    )

    query = "Je veux un PC rapide pour mon travail"

    print("\nQuestion :", query)

    query_vector = embedding_model.encode([query])[0]

    results = engine.search(
        query_vector,
        limit=5
    )

    print("\nRésultats de recherche :")

    for result in results:

        print("------------------------")

        print("Texte :", result.payload["text"])

        print("Score :", round(result.score, 4))


if __name__ == "__main__":
    main()