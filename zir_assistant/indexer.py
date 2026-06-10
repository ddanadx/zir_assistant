"""Індексатор: ембедить Q&A через Gemini і складає в ChromaDB."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.api.types import Embeddings
from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from zir_assistant.config import settings

logger = logging.getLogger(__name__)


def _gemini_client() -> genai.Client:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY не встановлено. Скопіюйте .env.example -> .env і додайте ключ.")
    return genai.Client(api_key=settings.gemini_api_key)


def _chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _embed_batch(client: genai.Client, texts: list[str], task_type: str) -> list[list[float]]:
    """Ембедити batch через Gemini. task_type: RETRIEVAL_DOCUMENT або RETRIEVAL_QUERY."""
    result = client.models.embed_content(
        model=settings.gemini_embed_model,
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in result.embeddings]


def _iter_items(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _doc_text(item: dict) -> str:
    """Текст для ембедингу. Питання домінантне, бо за ним шукаємо."""
    parts = []
    if item.get("category"):
        parts.append(f"[{item['category']}]")
    parts.append(f"Питання: {item['question']}")
    parts.append(f"Відповідь: {item['answer_full']}")
    return "\n".join(parts)


def build_index(reset: bool = False) -> int:
    """Побудувати індекс з data/zir.jsonl -> data/chroma/."""
    if not settings.scraped_path.exists():
        raise FileNotFoundError(
            f"{settings.scraped_path} не знайдено. Спочатку запустіть: python -m scripts.scrape"
        )

    gemini = _gemini_client()
    chroma = _chroma_client()

    if reset:
        try:
            chroma.delete_collection(settings.rag_collection_name)
            logger.info("Стару колекцію видалено")
        except Exception:
            pass

    collection = chroma.get_or_create_collection(
        name=settings.rag_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get()["ids"])
    logger.info("У колекції вже є %d документів", len(existing_ids))

    # Зібрати нові документи для ембедингу
    new_items: list[dict] = []
    for item in _iter_items(settings.scraped_path):
        doc_id = str(item["id"])
        if doc_id in existing_ids:
            continue
        new_items.append(item)

    logger.info("Треба проембедити %d нових документів", len(new_items))

    if not new_items:
        return 0

    batch_size = settings.rag_embed_batch
    total_added = 0
    with tqdm(total=len(new_items), desc="Embedding") as pbar:
        for i in range(0, len(new_items), batch_size):
            batch = new_items[i : i + batch_size]
            texts = [_doc_text(it) for it in batch]
            embeddings: Embeddings = _embed_batch(gemini, texts, task_type="RETRIEVAL_DOCUMENT")

            collection.add(
                ids=[str(it["id"]) for it in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "id": it["id"],
                        "url": it["url"],
                        "category": it.get("category") or "",
                        "question": it["question"],
                        "status": it.get("status") or "",
                    }
                    for it in batch
                ],
            )
            total_added += len(batch)
            pbar.update(len(batch))

    logger.info("Додано %d документів", total_added)
    return total_added


def embed_query(text: str) -> list[float]:
    """Ембедити запит користувача."""
    client = _gemini_client()
    vecs = _embed_batch(client, [text], task_type="RETRIEVAL_QUERY")
    return vecs[0]


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Повернути топ-K релевантних Q&A."""
    top_k = top_k or settings.rag_top_k
    chroma = _chroma_client()
    collection = chroma.get_collection(settings.rag_collection_name)
    query_vec = embed_query(query)
    res = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
    ):
        hits.append(
            {
                "id": meta["id"],
                "url": meta["url"],
                "category": meta["category"],
                "question": meta["question"],
                "status": meta["status"],
                "text": doc,
                "distance": dist,
            }
        )
    return hits
