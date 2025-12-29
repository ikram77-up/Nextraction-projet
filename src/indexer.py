import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class Indexer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.documents = []
        self.texts = []
        self.tokenized_texts = []
        self.embeddings = None

        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.bm25 = None

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

        self.texts = [doc["text"] for doc in self.documents]
        self.tokenized_texts = [text.lower().split() for text in self.texts]

    def build_indexes(self):
        self.bm25 = BM25Okapi(self.tokenized_texts)
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def get_data(self):
        return self.documents, self.bm25, self.embeddings
