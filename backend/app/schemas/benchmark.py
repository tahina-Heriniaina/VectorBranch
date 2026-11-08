
from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):

    dataset_id: str = Field(
        min_length=1,
        description="Identifiant du dataset importé."
    )

    engine: str = Field(
        default="qdrant",
        description="Moteur vectoriel à évaluer."
    )

    queries: list[str] = Field(min_length=1)

    top_k: int = Field(
        default=10,
        ge=1,
        le=100
    )

    repetitions: int = Field(
        default=10,
        ge=1,
        le=100
    )
