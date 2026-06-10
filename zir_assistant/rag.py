"""RAG-ланцюжок: retrieve з Chroma + generate через Gemini."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types

from zir_assistant.config import settings
from zir_assistant.indexer import search

logger = logging.getLogger(__name__)


# Моделі в порядку fallback: якщо перша перевантажена — пробуємо наступну.
# Free tier квоти різні: 2.5-flash 5 RPM, 2.0-flash 15 RPM, 2.5-flash-lite 15 RPM.
_MODEL_FALLBACK_CHAIN = [
    settings.gemini_chat_model,
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def _is_retryable(exception: Exception) -> bool:
    """503 (перевантаження) / 429 (квота) — варто повторити або перемкнути модель."""
    msg = str(exception)
    return any(code in msg for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


SYSTEM_PROMPT = """Ти — помічник з податкового законодавства України.
Твоя база знань — офіційний ресурс ДПС України ЗІР (zir.tax.gov.ua).

Правила:
1. Відповідай ТІЛЬКИ на основі контексту нижче. Якщо в контексті немає відповіді —
   чесно скажи "В базі знань ЗІР я не знайшов відповіді на це питання" і запропонуй
   уточнити або звернутись до ДПС напряму.
2. Завжди посилайся на джерела у форматі [№ID] в кінці релевантних тверджень.
3. Відповідай українською, чітко, структуровано. Уникай канцеляриту.
4. Якщо питання неоднозначне — попроси уточнити.
5. Не давай юридичних консультацій — лише довідкову інформацію з ЗІР.
6. Якщо документ має статус "Не чинна" — обов'язково попередь про це.
"""


@dataclass
class Citation:
    id: int
    url: str
    category: str
    question: str
    status: str


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        status_note = f" (СТАТУС: {hit['status']})" if hit.get("status") else ""
        block = (
            f"--- [№{hit['id']}]{status_note} ---\n"
            f"Категорія: {hit.get('category') or '—'}\n"
            f"{hit['text']}\n"
            f"URL: {hit['url']}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _build_user_prompt(question: str, context: str) -> str:
    return f"""КОНТЕКСТ З ЗІР:
{context}

ЗАПИТАННЯ КОРИСТУВАЧА:
{question}

Дай відповідь із посиланнями на документи у форматі [№ID]."""


def _client() -> genai.Client:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY не встановлено.")
    return genai.Client(api_key=settings.gemini_api_key)


async def answer_stream(question: str) -> AsyncIterator[dict]:
    """Async генератор. Видає:
    - {"type": "citations", "items": [...]} — одразу, із джерелами
    - {"type": "token", "text": "..."} — поступово текст відповіді
    - {"type": "done"} — в кінці
    - {"type": "error", "message": "..."} — у разі помилки
    """
    try:
        hits = search(question)
    except Exception as e:
        logger.exception("Retrieval failed")
        yield {"type": "error", "message": f"Помилка пошуку: {e}"}
        return

    citations = [
        {
            "id": h["id"],
            "url": h["url"],
            "category": h.get("category") or "",
            "question": h["question"],
            "status": h.get("status") or "",
        }
        for h in hits
    ]
    yield {"type": "citations", "items": citations}

    if not hits:
        yield {"type": "token", "text": "В базі знань ЗІР я не знайшов релевантних документів."}
        yield {"type": "done"}
        return

    context = _format_context(hits)
    user_prompt = _build_user_prompt(question, context)

    client = _client()
    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
    )

    last_exception: Exception | None = None
    # Деконфліктуємо дублікати в ланцюжку моделей (бо settings.gemini_chat_model може збігатись з одним із fallback).
    seen_models: set[str] = set()
    models_tried: list[str] = []
    for model_name in _MODEL_FALLBACK_CHAIN:
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        models_tried.append(model_name)

        # Ретраї з exponential backoff на тимчасові помилки
        for attempt in range(3):
            try:
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=user_prompt,
                    config=config,
                )
                produced_any_token = False
                for chunk in stream:
                    text = getattr(chunk, "text", None)
                    if text:
                        produced_any_token = True
                        yield {"type": "token", "text": text}
                if produced_any_token:
                    yield {"type": "done"}
                    return
                break  # stream завершився без помилки, але без токенів — пробуємо іншу модель
            except Exception as e:
                last_exception = e
                if not _is_retryable(e):
                    logger.exception("Non-retryable generation error")
                    yield {"type": "error", "message": f"Помилка генерації: {e}"}
                    return
                delay = 1.5 * (2**attempt)
                logger.warning(
                    "Retryable error on %s (attempt %d): %s — sleep %.1fs",
                    model_name, attempt + 1, str(e)[:120], delay,
                )
                await asyncio.sleep(delay)
        # Всі ретраї для цієї моделі вичерпані — переходимо на наступну
        logger.warning("Switching to fallback model after %s", model_name)

    logger.error("All models exhausted. Last error: %s", last_exception)
    yield {
        "type": "error",
        "message": (
            f"Усі доступні моделі Gemini перевантажені або вичерпали квоту: "
            f"{', '.join(models_tried)}. Спробуйте через хвилину."
        ),
    }
