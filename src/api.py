from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from src.indexer  import Indexer
from src.search import bm25_search, semantic_search, reciprocal_rank_fusion
from src.rerank import ReRanker

app = FastAPI(title="Nextraction Search API")
# upload model une fois
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "corpus.json"

indexer = Indexer(DATA_PATH)
indexer.load_data()
indexer.build_indexes()

documents, bm25, embeddings = indexer.get_data()
reranker = ReRanker()


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # bm25 | semantic | hybrid


@app.get("/")
def root():
    return {"message": "Nextraction API is running"}


@app.post("/search")
def search(req: SearchRequest):
    query = req.query

    if req.mode == "bm25":
        results = bm25_search(query, bm25, documents)

    elif req.mode == "semantic":
        results = semantic_search(query, indexer.model, embeddings)

    else:
        bm25_res = bm25_search(query, bm25, documents)
        sem_res = semantic_search(query, indexer.model, embeddings)
        results = reciprocal_rank_fusion(bm25_res, sem_res)

    return {
        "query": query,
        "results": [
            {
                "score": float(score),
                "text": documents[i]["text"]
            }
            for i, score in results[:5]
        ]
    }
