from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ReRanker:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def rerank(self, query, candidate_texts):
    
        query_vec = self.model.encode([query])
        doc_vecs = self.model.encode(candidate_texts)

        scores = cosine_similarity(query_vec, doc_vecs)[0]
        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked  
