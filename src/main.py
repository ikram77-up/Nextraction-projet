# src/main.py
from indexer import Indexer
from search import bm25_search, semantic_search, reciprocal_rank_fusion
from rerank import ReRanker
from evaluate import precision_at_k, mean_reciprocal_rank

DATA_PATH = "data/corpus.json"

def main():
   
    indexer = Indexer(DATA_PATH)
    indexer.load_data()
    indexer.build_indexes()  

    documents, bm25, embeddings = indexer.get_data()

    query = input(" Entrez votre requête : ")


    print("\n--- Résultats BM25 ---")
    bm25_results = bm25_search(query, bm25, documents)
    for idx, score in bm25_results:
        print(f"{score:.3f} | {documents[idx]['text']}")

  
    print("\n--- Résultats Sémantiques ---")
    sem_results = semantic_search(query, indexer.model, embeddings)
    for idx, score in sem_results:
        print(f"{score:.3f} | {documents[idx]['text']}")


    print("\n--- Résultats Hybrides (RRF) ---")
    hybrid = reciprocal_rank_fusion(bm25_results, sem_results)
    for idx, score in hybrid[:5]:
        print(f"{score:.3f} | {documents[idx]['text']}")

    print("\n--- Résultats Rerankés ---")
    reranker = ReRanker()
    candidate_texts = [documents[idx]["text"] for idx, _ in hybrid[:5]]
    reranked = reranker.rerank(query, candidate_texts)
    for rank, score in reranked:
        print(f"{score:.3f} | {candidate_texts[rank]}")

    relvant_docs = [0, 2]
    print("\n--- Évaluation ---")
    print("Precision@5:", precision_at_k(hybrid, relvant_docs, k=5))
    print("MRR:", mean_reciprocal_rank(hybrid, relvant_docs))

if __name__ == "__main__":
    main()
