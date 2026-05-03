"""
C2.2: Auto-invalidation when knowledge base (menu/FAQ) changes.

Watches data/ directory for changes to menu.csv, faq.csv, docs.txt.
When any data file changes:
1. Invalidate the entire cache (SimpleCache.invalidate())
2. Re-ingest RAG embeddings (re-embed menu + FAQ)

Usage:
  # Run alongside the FastAPI app (separate process):
  python scripts/watch_data_and_invalidate.py

  # Or import as a module:
  from scripts.watch_data_and_invalidate import start_watching
  start_watching(cache=my_cache, rag_store=my_rag)

For production: This replaces manual cache invalidation.
Trigger: menu.csv, faq.csv, docs.txt changes → cache.clear() + RAG re-ingest.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
except ImportError:
    Observer = None
    Observer = FileSystemEventHandler = FileModifiedEvent = None

from app.cache import SimpleCache
from app.rag import RAGStore
from app.config import CHROMA_DIR


# Files that trigger cache invalidation + RAG re-ingest
DATA_FILES = [
    Path("data/menu.csv"),
    Path("data/faq.csv"),
    Path("data/docs.txt"),
]
WATCH_DIR = Path("data")


class DataFileHandler(FileSystemEventHandler if Observer else object):
    """
    Watches data/ directory. On change to menu.csv, faq.csv, or docs.txt:
    - Invalidates cache
    - Re-ingests RAG embeddings
    """

    def __init__(
        self,
        cache: Optional[SimpleCache] = None,
        rag_store: Optional[RAGStore] = None,
        debounce_seconds: float = 2.0,
    ) -> None:
        self.cache = cache
        self.rag_store = rag_store
        self.debounce_seconds = debounce_seconds
        self._last_trigger: float = 0.0
        self._trigger_count: int = 0
        self._suppress_next: set[str] = set()

    def on_modified(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path).resolve()

        # Only care about our specific data files
        relevant = any(path == f or path.name == f.name for f in DATA_FILES)
        if not relevant:
            return

        # Debounce: ignore rapid successive events (e.g. save + save)
        now = time.time()
        if now - self._last_trigger < self.debounce_seconds:
            self._suppress_next.add(str(path))
            return

        self._last_trigger = now
        self._trigger_count += 1
        self._suppress_next.discard(str(path))

        print(f"\n[C2.2] Data file changed: {path.name}")
        self._invalidate_and_reingest(path)

    def on_created(self, event):
        self.on_modified(event)

    def _invalidate_and_reingest(self, changed_file: Path):
        print(f"[C2.2] Triggering cache invalidation + RAG re-ingest...")

        # 1. Invalidate cache
        if self.cache:
            try:
                self.cache.invalidate()
                print(f"[C2.2] ✓ Cache invalidated ({self.cache.backend_name()} backend)")
            except Exception as e:
                print(f"[C2.2] ✗ Cache invalidation failed: {e}")
        else:
            print("[C2.2] Warning: No cache instance provided")

        # 2. Re-ingest RAG (reload menu + FAQ into ChromaDB)
        if self.rag_store:
            try:
                self.rag_store.clear()  # Clear existing embeddings
                self._reingest_rag(changed_file)
                print("[C2.2] ✓ RAG re-ingested successfully")
            except Exception as e:
                print(f"[C2.2] ✗ RAG re-ingest failed: {e}")
        else:
            print("[C2.2] Warning: No RAG store provided")

        print(f"[C2.2] Done. Total triggers: {self._trigger_count}")

    def _reingest_rag(self, changed_file: Path):
        """Re-ingest specific data file into RAG store."""
        if changed_file.name == "menu.csv":
            self._reingest_menu()
        elif changed_file.name == "faq.csv":
            self._reingest_faq()
        elif changed_file.name == "docs.txt":
            self._reingest_docs()

    def _reingest_menu(self):
        """Re-ingest menu items."""
        import csv
        from app.config import EMBEDDING_MODEL

        if not self.rag_store or not self.rag_store.collection:
            return

        menu_path = Path("data/menu.csv")
        if not menu_path.exists():
            return

        with menu_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc = (
                    f"{row['name']} - {row['category']} - "
                    f"{row['price']}VND - caffeine: {row['caffeine']} - "
                    f"tags: {row['tags']} - description: {row['description']}"
                )
                self.rag_store.add(doc, intent="order", metadata=row)

        print(f"[C2.2] Re-ingested menu.csv")

    def _reingest_faq(self):
        """Re-ingest FAQ entries."""
        import csv

        if not self.rag_store or not self.rag_store.collection:
            return

        faq_path = Path("data/faq.csv")
        if not faq_path.exists():
            return

        with faq_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc = f"Q: {row['question']}\nA: {row['answer']}"
                self.rag_store.add(doc, intent="faq", metadata={"question": row["question"]})

        print(f"[C2.2] Re-ingested faq.csv")

    def _reingest_docs(self):
        """Re-ingest policy docs."""
        docs_path = Path("data/docs.txt")
        if not docs_path.exists():
            return

        if not self.rag_store or not self.rag_store.collection:
            return

        content = docs_path.read_text(encoding="utf-8")
        chunks = content.split("\n\n")
        for chunk in chunks:
            if chunk.strip():
                self.rag_store.add(chunk.strip(), intent="faq", metadata={"source": "docs"})

        print(f"[C2.2] Re-ingested docs.txt ({len(chunks)} chunks)")


def start_watching(
    cache: Optional[SimpleCache] = None,
    rag_store: Optional[RAGStore] = None,
    watch_dir: Optional[Path] = None,
    debounce_seconds: float = 2.0,
) -> Optional[Observer]:
    """
    Start watching data/ directory for file changes.

    Args:
        cache: SimpleCache instance to invalidate on changes
        rag_store: RAGStore instance to re-ingest on changes
        watch_dir: Directory to watch (default: data/)
        debounce_seconds: Ignore events within this window (default: 2s)

    Returns:
        Observer instance (or None if watchdog not installed)
    """
    if Observer is None:
        print("[C2.2] Error: watchdog not installed. Run: pip install watchdog")
        return None

    watch_dir = watch_dir or WATCH_DIR
    handler = DataFileHandler(
        cache=cache,
        rag_store=rag_store,
        debounce_seconds=debounce_seconds,
    )

    observer = Observer()
    observer.schedule(handler, str(watch_dir.resolve()), recursive=False)
    observer.start()
    print(f"[C2.2] Watching {watch_dir} for changes...")
    return observer


def run_watcher():
    """
    Standalone watcher script. Run with:
        python scripts/watch_data_and_invalidate.py

    Requires: pip install watchdog
    """
    if Observer is None:
        print("[C2.2] Error: watchdog not installed.")
        print("  pip install watchdog")
        return

    print("[C2.2] Starting data watcher (C2.2 cache invalidation)...")
    print(f"  Watching: {WATCH_DIR.resolve()}")
    print(f"  Trigger files: {[f.name for f in DATA_FILES]}")

    # Import lazily to avoid breaking if app modules have issues
    from app.cache import SimpleCache
    from app.rag import RAGStore

    cache = SimpleCache()
    rag_store = RAGStore()

    observer = start_watching(cache=cache, rag_store=rag_store)

    if observer is None:
        return

    print("[C2.2] Watcher started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[C2.2] Stopping watcher...")
        observer.stop()
    observer.join()
    print("[C2.2] Watcher stopped.")


if __name__ == "__main__":
    run_watcher()
