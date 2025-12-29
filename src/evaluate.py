def precision_at_k(results,relvant_docs,k=5):
    retrieved = [doc_id for doc_id, _ in results[:k]]
    relvant_set = set(relvant_docs)

    if not retrieved:
        return 0.0
    
    relvant_count =(1 for doc in retrieved if doc in relvant_set)
    return relvant_count / k

def mean_reciprocal_rank(results,relvant_docs):
    for rank, (doc_id, _) in enumerate(results, start=1):
        if doc_id in relvant_docs:
            return 1 / rank
    return 0.0