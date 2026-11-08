from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=5, ge=1, le=20)
    dataset_id: str | None = None
    engine: str = "qdrant"


class SearchResult(BaseModel):
    id: int
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResult]