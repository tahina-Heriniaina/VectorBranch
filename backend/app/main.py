from functools import lru_cache

from fastapi import FastAPI, HTTPException, UploadFile, File

from app.schemas.search import SearchRequest, SearchResponse
from app.schemas.benchmark import BenchmarkRequest

from app.services.search_service import SearchService
from app.services.dataset_service import DatasetService
from app.services.benchmark_service import BenchmarkService
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="VectorBench API",
    description="Plateforme de benchmark des bases vectorielles",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_search_service():
    return SearchService()


@lru_cache(maxsize=1)
def get_dataset_service():
    return DatasetService()


@lru_cache(maxsize=1)
def get_benchmark_service():
    return BenchmarkService()


@app.get("/")
def home():
    return {
        "project": "VectorBench",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/engines")
def get_engines():
    return {
        "engines": [
            "Qdrant",
            "Milvus",
            "Weaviate",
            "Chroma",
            "FAISS",
            "pgvector",
            "Pinecone"
        ]
    }


@app.get("/api/engines/{engine_name}/status")
def engine_status(engine_name: str):
    status = get_search_service().get_engine_status(engine_name)

    if not status["connected"]:
        raise HTTPException(status_code=503, detail=status)

    return status


@app.post("/api/datasets")
async def import_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()

        result = get_dataset_service().import_dataset(
            filename=file.filename or "dataset.txt",
            content=content
        )

        return {
            "message": "Dataset importé avec succès",
            "dataset": result
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/datasets")
def list_datasets():
    return {
        "datasets": get_dataset_service().list_datasets()
    }



@app.post("/api/search", response_model=SearchResponse)
def semantic_search(request: SearchRequest):
    try:
        collection_name = None

        if request.dataset_id:
            metadata = get_dataset_service().get_metadata(
                request.dataset_id
            )

            engine_key = (
                get_search_service()
                .engines
                .normalize_name(request.engine)
            )

            collection_name = metadata.get(
                "engine_collections", {}
            ).get(engine_key)

            if not collection_name and engine_key == "weaviate":
                metadata = get_dataset_service().ensure_engine_index(
                    request.dataset_id,
                    engine_key
                )
                collection_name = metadata.get(
                    "engine_collections", {}
                ).get(engine_key)

            # Compatibilité avec les anciens datasets Qdrant.
            if not collection_name and engine_key == "qdrant":
                collection_name = metadata.get("collection_name")

            if not collection_name:
                raise ValueError(
                    f"Dataset non indexé dans {engine_key}."
                )

        results = get_search_service().search(
            query=request.query,
            limit=request.limit,
            collection_name=collection_name,
            engine_name=request.engine
        )

        return {
            "query": request.query,
            "total": len(results),
            "results": results
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except ConnectionError as error:
        raise HTTPException(status_code=503, detail=str(error))


@app.post("/api/benchmarks/run")
def run_benchmark(request: BenchmarkRequest):
    try:
        return get_benchmark_service().run(
            dataset_id=request.dataset_id,
            queries=request.queries,
            top_k=request.top_k,
            repetitions=request.repetitions,
            engine_name=request.engine
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))



@app.get("/api/benchmarks/results")
def list_benchmark_results():
    return {
        "results": get_benchmark_service().list_results()
    }

@app.get("/api/engines/{engine_name}/status")
def engine_status(engine_name: str):
    status = get_search_service().get_engine_status(engine_name)

    if not status["connected"]:
        raise HTTPException(status_code=503, detail=status)

    return status