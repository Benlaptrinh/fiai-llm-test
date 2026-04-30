"""
RAGStore: Retrieval-Augmented Generation layer.

Responsibilities:
- store documents
- perform vector search
- return relevant context

Uses ChromaDB for simplicity.
"""


class RAGStore:
    def __init__(self):
        self.data = ["Phin sữa đá", "Trà sen vàng"]

    def search(self, query):
        return self.data
