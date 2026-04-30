"""
Ingestion pipeline for F&B knowledge base.

Sources:
- data/menu.csv
- data/faq.csv
- data/docs.txt

Output:
- ChromaDB persistent vector store
"""

from __future__ import annotations

import os
import shutil
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CHROMA_DIR
from app.rag import RAGStore


def reset_vector_store() -> None:
    """
    Remove old ChromaDB directory before re-ingestion.
    This prevents duplicated stale documents.
    """
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"Removed existing vector store: {CHROMA_DIR}")


def ingest_menu(store: RAGStore) -> None:
    """Ingest menu rows into vector store."""
    dataframe = pd.read_csv("data/menu.csv")

    ids = []
    docs = []
    metas = []
    for _, row in dataframe.iterrows():
        doc_id = f"menu_{row['id']}"
        text = (
            f"Tên món: {row['name']}. "
            f"Danh mục: {row['category']}. "
            f"Size: {row['size']}. "
            f"Giá: {row['price']} VND. "
            f"Caffeine: {row['caffeine']}. "
            f"Sweetness: {row['sweetness']}. "
            f"Tags: {row['tags']}. "
            f"Thành phần: {row['ingredients']}. "
            f"Mô tả: {row['description']}."
        )

        ids.append(doc_id)
        docs.append(text)
        metas.append(
            {
                "domain": "menu",
                "type": "MenuItem",
                "name": row["name"],
                "category": row["category"],
                "size": row["size"],
                "price": int(row["price"]),
                "caffeine": row["caffeine"],
                "sweetness": row["sweetness"],
            }
        )

    store.collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"Ingested menu items: {len(ids)}")


def ingest_faq(store: RAGStore) -> None:
    """Ingest FAQ rows into vector store."""
    dataframe = pd.read_csv("data/faq.csv")

    ids = []
    docs = []
    metas = []
    for _, row in dataframe.iterrows():
        doc_id = f"faq_{row['id']}"
        text = f"Câu hỏi: {row['question']}\nTrả lời: {row['answer']}"

        ids.append(doc_id)
        docs.append(text)
        metas.append({"domain": "faq", "type": "FAQ"})

    store.collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"Ingested FAQ items: {len(ids)}")


def ingest_docs(store: RAGStore) -> None:
    """Ingest internal policy docs into vector store."""
    with open("data/docs.txt", "r", encoding="utf-8") as file:
        content = file.read()

    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

    ids = []
    docs = []
    metas = []
    for idx, chunk in enumerate(chunks):
        ids.append(f"doc_{idx}")
        docs.append(chunk)
        metas.append({"domain": "doc", "type": "Chunk"})

    store.collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"Ingested document chunks: {len(ids)}")


if __name__ == "__main__":
    reset_vector_store()
    rag_store = RAGStore()

    ingest_menu(rag_store)
    ingest_faq(rag_store)
    ingest_docs(rag_store)

    print("Total documents:", rag_store.count())
