"""Бенчмарк retrieval та generation.

Запускає:
  R1 — Recall@K vs K (1, 3, 5, 6, 10, 20)
  R2 — task-asymmetric vs symmetric
  R3 — формат документа (Q+A, Q-only, A-only)
  G2 — citation accuracy на 20 прикладах (швидкий підрахунок)

Результати: data/eval/results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import chromadb
from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential

from zir_assistant.config import settings
from zir_assistant.indexer import _embed_batch, _gemini_client

logger = logging.getLogger(__name__)

# ──────────────────────────── helpers ────────────────────────────

def load_testset(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=4, max=60))
def embed_query(client: genai.Client, text: str, task_type: str) -> list[float]:
    vecs = _embed_batch(client, [text], task_type=task_type)
    return vecs[0]


# ──────────────────────────── R1: Recall@K ────────────────────────────

def run_r1(testset: list[dict], collection: chromadb.Collection,
           client: genai.Client, k_values: list[int]) -> dict:
    max_k = max(k_values)
    hits_at_k: dict[int, int] = defaultdict(int)
    reciprocal_ranks: list[float] = []

    for i, ex in enumerate(testset):
        vec = embed_query(client, ex["query"], task_type="RETRIEVAL_QUERY")
        res = collection.query(
            query_embeddings=[vec],
            n_results=max_k,
            include=["metadatas"],
        )
        retrieved_ids = [int(m["id"]) for m in res["metadatas"][0]]
        gold = ex["gold_id"]

        # MRR
        try:
            rank = retrieved_ids.index(gold) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)

        for k in k_values:
            if gold in retrieved_ids[:k]:
                hits_at_k[k] += 1

        if (i + 1) % 10 == 0:
            logger.info("R1 progress: %d/%d", i + 1, len(testset))
        time.sleep(0.3)  # щоб не перевантажувати embedding quota

    n = len(testset)
    result = {
        "recall_at_k": {k: round(hits_at_k[k] / n, 4) for k in k_values},
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "n": n,
    }
    return result


# ──────────────────────────── R2: asymmetric vs symmetric ────────────────────────────

def run_r2(testset: list[dict], client: genai.Client, k: int = 5) -> dict:
    chroma = chromadb.PersistentClient(path=str(settings.chroma_dir))

    results = {}
    for query_type, doc_type, label in [
        ("RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "asymmetric"),
        ("SEMANTIC_SIMILARITY", "SEMANTIC_SIMILARITY", "symmetric"),
    ]:
        # Для symmetric треба переіндексувати — дорого, тому тільки для малої вибірки (20)
        # Оцінка тільки за query-side: використовуємо існуючий index (RETRIEVAL_DOCUMENT)
        # і варіюємо тільки query task_type
        col = chroma.get_collection(settings.rag_collection_name)
        hits = 0
        sample = testset[:20]  # швидка вибірка
        for ex in sample:
            vec = embed_query(client, ex["query"], task_type=query_type)
            res = col.query(query_embeddings=[vec], n_results=k, include=["metadatas"])
            retrieved_ids = [int(m["id"]) for m in res["metadatas"][0]]
            if ex["gold_id"] in retrieved_ids:
                hits += 1
            time.sleep(0.3)
        results[label] = {"recall_at_k": round(hits / len(sample), 4), "k": k, "n": len(sample)}

    return results


# ──────────────────────────── R3: document format ────────────────────────────

def run_r3(testset: list[dict], client: genai.Client) -> dict:
    """
    Порівнює різні форматами документа в тимчасовому in-memory collection.
    Використовує підвибірку 30 документів для швидкості.
    """
    # Завантажити повний корпус
    corpus: dict[int, dict] = {}
    with settings.scraped_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                corpus[d["id"]] = d

    sample_ids = list(set(ex["gold_id"] for ex in testset[:30]))

    formats = {
        "Q+A": lambda d: f"Питання: {d['question']}\nВідповідь: {d['answer_full']}",
        "Q-only": lambda d: d["question"],
        "A-only": lambda d: d["answer_full"],
    }

    results = {}
    for fmt_name, fmt_fn in formats.items():
        logger.info("R3: indexing format %s ...", fmt_name)
        # Тимчасовий in-memory chroma (унікальне ім'я щоб уникнути конфлікту)
        tmp = chromadb.EphemeralClient()
        col_name = f"tmp_r3_{fmt_name.replace('+', '_')}"
        col = tmp.create_collection(col_name, metadata={"hnsw:space": "cosine"})

        # Ембедимо тільки gold-документи + рандомні 200 як distractors
        import random
        all_ids = list(corpus.keys())
        random.seed(42)
        distractors = random.sample([i for i in all_ids if i not in sample_ids],
                                    min(200, len(all_ids) - len(sample_ids)))
        index_ids = sample_ids + distractors

        texts = [fmt_fn(corpus[i]) for i in index_ids if i in corpus]
        valid_ids = [i for i in index_ids if i in corpus]

        # Batch embed
        BATCH = 50
        all_vecs = []
        for bi in range(0, len(texts), BATCH):
            batch = texts[bi:bi+BATCH]
            vecs = _embed_batch(client, batch, task_type="RETRIEVAL_DOCUMENT")
            all_vecs.extend(vecs)
            time.sleep(1.0)

        col.add(
            ids=[str(i) for i in valid_ids],
            embeddings=all_vecs,
            documents=texts,
        )

        # Оцінка Recall@5
        hits = 0
        for ex in testset[:30]:
            if ex["gold_id"] not in corpus:
                continue
            qvec = embed_query(client, ex["query"], task_type="RETRIEVAL_QUERY")
            res = col.query(query_embeddings=[qvec], n_results=5, include=["metadatas"])
            retrieved_ids = [int(r_id) for r_id in res["ids"][0]]
            if ex["gold_id"] in retrieved_ids:
                hits += 1
            time.sleep(0.3)

        results[fmt_name] = {"recall_at_5": round(hits / 30, 4), "n": 30}
        logger.info("R3 %s: Recall@5 = %.3f", fmt_name, hits / 30)

    return results


# ──────────────────────────── G2: citation accuracy ────────────────────────────

def run_g2(testset: list[dict], k: int = 6) -> dict:
    """Перевіряє citation accuracy на 20 прикладах через answer_stream."""
    from zir_assistant.rag import answer_stream
    import asyncio

    sample = testset[:20]
    correct_cites = 0
    total_cites = 0

    async def check(ex: dict) -> tuple[int, int]:
        hits_context_ids = set()
        answer_text = ""
        async for event in answer_stream(ex["query"]):
            if event["type"] == "citations":
                for item in event["items"]:
                    hits_context_ids.add(str(item["id"]))
            elif event["type"] == "token":
                answer_text += event["text"]

        # Extract [№ID] citations
        cited_ids = re.findall(r"\[№(\d+)\]", answer_text)
        correct = sum(1 for c in cited_ids if c in hits_context_ids)
        return correct, len(cited_ids)

    for i, ex in enumerate(sample):
        c, t = asyncio.run(check(ex))
        correct_cites += c
        total_cites += t
        logger.info("G2 [%d/%d] correct=%d total=%d", i + 1, len(sample), c, t)

    acc = round(correct_cites / max(total_cites, 1), 4)
    return {"citation_accuracy": acc, "correct": correct_cites, "total": total_cites, "n": len(sample)}


# ──────────────────────────── main ────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="data/eval/testset.jsonl")
    parser.add_argument("--out", default="data/eval/results.json")
    parser.add_argument("--skip-r3", action="store_true", help="пропустити R3 (дорого за кількістю API-calls)")
    parser.add_argument("--skip-g2", action="store_true", help="пропустити G2")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    testset_path = Path(args.testset)
    if not testset_path.exists():
        raise FileNotFoundError(f"Тест-сет не знайдено: {testset_path}. Спочатку: python -m scripts.build_testset")

    testset = load_testset(testset_path)
    logger.info("Тест-сет: %d прикладів", len(testset))

    client = _gemini_client()
    chroma = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = chroma.get_collection(settings.rag_collection_name)
    logger.info("ChromaDB: %d документів", collection.count())

    results: dict = {}

    logger.info("=== R1: Recall@K ===")
    results["R1"] = run_r1(testset, collection, client, k_values=[1, 3, 5, 6, 10, 20])
    logger.info("R1: %s", results["R1"])

    logger.info("=== R2: asymmetric vs symmetric ===")
    results["R2"] = run_r2(testset, client)
    logger.info("R2: %s", results["R2"])

    if not args.skip_r3:
        logger.info("=== R3: document format ===")
        results["R3"] = run_r3(testset, client)
        logger.info("R3: %s", results["R3"])

    if not args.skip_g2:
        logger.info("=== G2: citation accuracy ===")
        results["G2"] = run_g2(testset)
        logger.info("G2: %s", results["G2"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Результати збережено: %s", out_path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
