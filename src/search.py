
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

""" search les mots rapidement dans les documents indexés   """

def bm25_search(query, bm25, documents, top_k=5):

    tokenized_query = query.lower().split()  
    scores = bm25.get_scores(tokenized_query)  
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]  


def semantic_search(query, model, embeddings, top_k=5):
    query_vec = model.encode([query]) 
    scores = cosine_similarity(query_vec, embeddings)[0] 
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def reciprocal_rank_fusion(results_a, results_b, k=60):

    scores = {}

    for rank, (idx, _) in enumerate(results_a):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    for rank, (idx, _) in enumerate(results_b):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

   
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
