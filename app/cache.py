"""
SimpleCache: reduce repeated LLM calls.

Uses normalized string matching.
Can be extended to semantic cache using embeddings.
"""


class SimpleCache:
    def __init__(self):
        self.store = {}

    def get(self, query):
        return self.store.get(query)

    def set(self, query, value):
        self.store[query] = value
