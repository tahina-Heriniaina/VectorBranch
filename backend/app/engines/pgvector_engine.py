
import os
import re

import psycopg

from psycopg import sql
from pgvector import Vector
from pgvector.psycopg import register_vector


class PgVectorEngine:

    def __init__(self):
        self.dsn = os.getenv(
            "PGVECTOR_DSN",
            "postgresql://vectorbench:vectorbench_dev_2026@localhost:5433/vectorbench_vectors"
        )

    def _table(self, collection_name):
        if not re.fullmatch(r"[a-zA-Z0-9_]+", collection_name):
            raise ValueError("Nom de table pgvector invalide.")

        return "vb_" + collection_name

    def create_collection(self, dimension: int, collection_name: str):
        if dimension <= 0:
            raise ValueError("Dimension invalide.")

        table = self._table(collection_name)
        index_name = table + "_hnsw"

        with psycopg.connect(self.dsn) as conn:
            conn.execute(
                "CREATE EXTENSION IF NOT EXISTS vector"
            )

            conn.execute(
                sql.SQL("""
                    CREATE TABLE {} (
                        id BIGINT PRIMARY KEY,
                        text TEXT NOT NULL,
                        embedding vector({})
                    )
                """).format(
                    sql.Identifier(table),
                    sql.Literal(dimension)
                )
            )

            conn.execute(
                sql.SQL("""
                    CREATE INDEX {}
                    ON {} USING hnsw (embedding vector_cosine_ops)
                """).format(
                    sql.Identifier(index_name),
                    sql.Identifier(table)
                )
            )

            register_vector(conn)

        return table

    def insert_vectors(self, texts, vectors, collection_name: str):
        table = self._table(collection_name)

        if len(texts) != len(vectors):
            raise ValueError("Nombre de textes et vecteurs différent.")

        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)

            query = sql.SQL("""
                INSERT INTO {} (id, text, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding
            """).format(sql.Identifier(table))

            for start in range(0, len(texts), 256):
                batch_texts = texts[start:start + 256]
                batch_vectors = vectors[start:start + 256]

                for index, (text, vector) in enumerate(
                    zip(batch_texts, batch_vectors),
                    start=start
                ):
                    conn.execute(
                        query,
                        (
                            index + 1,
                            text,
                            Vector(vector.tolist())
                        )
                    )

    def search(self, vector, limit=5, collection_name=None):
        if not collection_name:
            raise ValueError("Table pgvector nécessaire.")

        table = self._table(collection_name)

        query = sql.SQL("""
            SELECT
                id,
                text,
                1 - (embedding <=> %s) AS score
            FROM {}
            ORDER BY embedding <=> %s
            LIMIT %s
        """).format(sql.Identifier(table))

        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)

            rows = conn.execute(
                query,
                (
                    Vector(vector.tolist()),
                    Vector(vector.tolist()),
                    limit
                )
            ).fetchall()

        return [
            {
                "id": row[0],
                "text": row[1],
                "score": round(float(row[2]), 4)
            }
            for row in rows
        ]

    def delete_collection(self, collection_name: str):
        table = self._table(collection_name)

        with psycopg.connect(self.dsn) as conn:
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {}").format(
                    sql.Identifier(table)
                )
            )
