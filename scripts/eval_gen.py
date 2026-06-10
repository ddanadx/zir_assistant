"""Evaluation G1 (Faithfulness vs Temperature) та G3 (RAG vs zero-shot baseline).

G1: генеруємо відповіді при T∈{0.0, 0.2, 0.5, 1.0}, оцінюємо LLM-as-Judge.
G3: порівнюємо RAG (T=0.2) з zero-shot Gemini (без контексту).

Результати: data/eval/results_gen.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential

from zir_assistant.config import settings
from zir_assistant.indexer import search
from zir_assistant.rag import SYSTEM_PROMPT, _format_context, _build_user_prompt

logger = logging.getLogger(__name__)

# ───── Промпт для LLM-as-Judge ─────
JUDGE_PROMPT = """Ти — суворий суддя, що оцінює відповідь AI-асистента.

КОНТЕКСТ (документи ЗІР, надані AI перед відповіддю):
{context}

ЗАПИТ КОРИСТУВАЧА:
{query}

ВІДПОВІДЬ AI:
{answer}

Завдання: оціни кожен фактологічний CLAIM у відповіді.
Для кожного claim визнач: чи підтверджений він прямо контекстом вище?

Правила:
- "підтверджений" = факт буквально або семантично присутній у КОНТЕКСТІ
- "не підтверджений" = факт відсутній у контексті або суперечить йому (галюцинація)
- Ігноруй загальні вступні фрази ("Відповідно до ЗІР", "Згідно з документами")
- Порожня відповідь або абстенція ("Не знайдено") = faithfulness 1.0

Поверни JSON у точному форматі:
{{
  "claims": [
    {{"text": "...", "supported": true}},
    ...
  ],
  "faithfulness": 0.0-1.0,
  "notes": "..."
}}
"""


def load_testset(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=3, max=60))
def generate_answer(client: genai.Client, query: str, context: str,
                    temperature: float, model: str = "gemini-2.5-flash-lite") -> str:
    """Генерує відповідь із контекстом."""
    user_prompt = _build_user_prompt(query, context)
    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=1024,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text.strip()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=3, max=60))
def generate_zero_shot(client: genai.Client, query: str,
                       model: str = "gemini-2.5-flash-lite") -> str:
    """Генерує відповідь БЕЗ контексту (zero-shot baseline)."""
    resp = client.models.generate_content(
        model=model,
        contents=query,
        config=genai_types.GenerateContentConfig(
            system_instruction=(
                "Ти — консультант з податкового законодавства України. "
                "Відповідай коротко і конкретно українською мовою."
            ),
            temperature=0.2,
            max_output_tokens=512,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text.strip()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=90))
def judge_faithfulness(client: genai.Client, query: str,
                       context: str, answer: str) -> dict:
    """LLM-as-Judge: повертає dict з faithfulness [0..1] та claims."""
    prompt = JUDGE_PROMPT.format(
        context=context[:3000],
        query=query,
        answer=answer[:1500],
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1024,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = resp.text.strip()
    # Витягуємо JSON навіть якщо є markdown огортання
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"faithfulness": -1.0, "error": text[:200]}


# ───── G1: Faithfulness vs Temperature ─────

def run_g1(testset: list[dict], client: genai.Client, n: int = 25) -> dict:
    temperatures = [0.0, 0.2, 0.5, 1.0]
    sample = testset[:n]
    results: dict[str, list[float]] = {str(t): [] for t in temperatures}

    for i, ex in enumerate(sample):
        hits = search(ex["query"], top_k=6)
        context = _format_context(hits)
        logger.info("G1 [%d/%d] query: %s", i + 1, n, ex["query"][:60])

        for temp in temperatures:
            try:
                answer = generate_answer(client, ex["query"], context, temperature=temp)
                judge = judge_faithfulness(client, ex["query"], context, answer)
                f = judge.get("faithfulness", -1.0)
                results[str(temp)].append(f)
                logger.info("  T=%.1f → faithfulness=%.2f", temp, f)
                time.sleep(0.5)
            except Exception as e:
                logger.error("  T=%.1f error: %s", temp, e)
                results[str(temp)].append(-1.0)

        time.sleep(1.0)

    summary = {}
    for t, scores in results.items():
        valid = [s for s in scores if s >= 0]
        summary[t] = {
            "mean_faithfulness": round(sum(valid) / max(len(valid), 1), 4),
            "n_valid": len(valid),
            "n_total": len(scores),
        }
    return summary


# ───── G3: RAG vs Zero-shot ─────

def run_g3(testset: list[dict], client: genai.Client, n: int = 20) -> dict:
    sample = testset[:n]
    rag_faithfulness = []
    zeroshot_faithfulness = []
    zeroshot_hallucination_count = 0

    for i, ex in enumerate(sample):
        hits = search(ex["query"], top_k=6)
        context = _format_context(hits)
        logger.info("G3 [%d/%d] query: %s", i + 1, n, ex["query"][:60])

        # RAG відповідь (T=0.2)
        rag_answer = generate_answer(client, ex["query"], context, temperature=0.2)
        rag_judge = judge_faithfulness(client, ex["query"], context, rag_answer)
        rag_f = rag_judge.get("faithfulness", -1.0)
        rag_faithfulness.append(rag_f)

        # Zero-shot відповідь (без контексту)
        zs_answer = generate_zero_shot(client, ex["query"])
        # Оцінюємо zero-shot теж відносно того ж контексту ЗІР (як reference)
        zs_judge = judge_faithfulness(client, ex["query"], context, zs_answer)
        zs_f = zs_judge.get("faithfulness", -1.0)
        zeroshot_faithfulness.append(zs_f)

        # Лічильник "явних галюцинацій": конкретні числа без опори на контекст
        if zs_f < 0.5 and zs_f >= 0:
            zeroshot_hallucination_count += 1

        logger.info("  RAG faith=%.2f | ZS faith=%.2f", rag_f, zs_f)
        time.sleep(2.0)

    def avg(lst: list[float]) -> float:
        v = [x for x in lst if x >= 0]
        return round(sum(v) / max(len(v), 1), 4)

    return {
        "rag": {
            "mean_faithfulness": avg(rag_faithfulness),
            "n": n,
        },
        "zero_shot": {
            "mean_faithfulness": avg(zeroshot_faithfulness),
            "hallucination_rate": round(zeroshot_hallucination_count / n, 4),
            "n": n,
        },
    }


# ───── main ─────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="data/eval/testset.jsonl")
    parser.add_argument("--out", default="data/eval/results_gen.json")
    parser.add_argument("--skip-g1", action="store_true")
    parser.add_argument("--skip-g3", action="store_true")
    parser.add_argument("--n-g1", type=int, default=25)
    parser.add_argument("--n-g3", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    testset = load_testset(Path(args.testset))
    logger.info("Тест-сет: %d прикладів", len(testset))

    client = genai.Client(api_key=settings.gemini_api_key)
    results: dict = {}

    if not args.skip_g1:
        logger.info("=== G1: Faithfulness vs Temperature (n=%d) ===", args.n_g1)
        results["G1"] = run_g1(testset, client, n=args.n_g1)
        logger.info("G1: %s", results["G1"])

    if not args.skip_g3:
        logger.info("=== G3: RAG vs Zero-shot (n=%d) ===", args.n_g3)
        results["G3"] = run_g3(testset, client, n=args.n_g3)
        logger.info("G3: %s", results["G3"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Збережено: %s", out)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
