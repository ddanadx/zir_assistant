"""Генерація синтетичного тест-сету для оцінки retrieval і generation.

Стратегія: для N випадкових документів з корпусу LLM генерує природномовний
запит, який неспеціаліст-платник написав би до ЗІР.

Результат: data/eval/testset.jsonl
  {id: str, query: str, gold_id: int, source: "synth", status: str}
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential

from zir_assistant.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ти допомагаєш будувати тест-сет для оцінки RAG-системи на корпусі ЗІР (Загальнодоступний інформаційно-довідковий ресурс ДПС України).

Тебе годують ОДНИМ Q&A-записом (питання + відповідь), і ти маєш сформулювати ІНШИЙ запит — такий, який написав би ЗВИЧАЙНИЙ ПЛАТНИК (ФОП або бухгалтер), НЕ знаючи точного формулювання з ЗІР. 

Вимоги:
1. Запит — природна мова, від 10 до 40 слів.
2. НЕ копіюй слова з оригінального питання — парафразуй концептуально.
3. Запит повинен точно адресувати той самий документ (не ширший, не вужчий).
4. Не вигадуй факти, яких нема в документі.
5. Поверни ТІЛЬКИ текст запиту — жодних пояснень, жодних лапок."""

USER_TEMPLATE = """Ось Q&A-запис з ЗІР:

ПИТАННЯ: {question}

ВІДПОВІДЬ (перші 300 слів): {answer_preview}

Сформулюй природномовний запит від платника, що шукає саме цю відповідь."""


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=3, max=60))
def generate_query(client: genai.Client, question: str, answer: str) -> str:
    answer_preview = " ".join(answer.split()[:300])
    user_msg = USER_TEMPLATE.format(question=question, answer_preview=answer_preview)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_msg,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=600,
            # thinking model витрачає ~200-500 thought-токени до відповіді;
            # з budget_tokens=0 вимикаємо thinking і output йде повністю
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text.strip()


def load_corpus(path: Path) -> list[dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100, help="кількість прикладів")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/eval/testset.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Завантажити вже зроблені (для resume)
    done_ids: set[int] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["gold_id"])
        logger.info("Вже готово: %d прикладів", len(done_ids))

    corpus = load_corpus(settings.scraped_path)
    logger.info("Корпус: %d документів", len(corpus))

    # Стратифікована вибірка: ~50% чинних, ~50% нечинних
    valid = [d for d in corpus if "Не чинна" not in (d.get("status") or "")]
    invalid = [d for d in corpus if "Не чинна" in (d.get("status") or "")]

    rng = random.Random(args.seed)
    n_valid = min(args.n // 2, len(valid))
    n_invalid = args.n - n_valid
    selected = rng.sample(valid, n_valid) + rng.sample(invalid, min(n_invalid, len(invalid)))
    rng.shuffle(selected)

    # Виключити вже зроблені
    selected = [d for d in selected if d["id"] not in done_ids]
    need = args.n - len(done_ids)
    selected = selected[:need]
    logger.info("Треба згенерувати: %d прикладів", len(selected))

    if not selected:
        logger.info("Все вже готово!")
        return

    client = genai.Client(api_key=settings.gemini_api_key)

    with out_path.open("a", encoding="utf-8") as fout:
        for i, doc in enumerate(selected, 1):
            try:
                query = generate_query(client, doc["question"], doc["answer_full"])
                record = {
                    "id": f"synth_{doc['id']}",
                    "query": query,
                    "gold_id": doc["id"],
                    "source": "synth",
                    "status": doc.get("status") or "чинна",
                    "original_question": doc["question"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                logger.info("[%d/%d] id=%d → %s", i, len(selected), doc["id"], query[:80])
                # ~3 RPM safety
                time.sleep(1.5)
            except Exception as e:
                logger.error("Помилка для id=%d: %s", doc["id"], e)

    # Підсумок
    total = sum(1 for _ in out_path.open("r").readlines() if _.strip())
    logger.info("Готово! Тест-сет: %d прикладів у %s", total, out_path)


if __name__ == "__main__":
    main()
