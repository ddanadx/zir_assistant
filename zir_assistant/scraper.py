"""Скрейпер ЗІР (zir.tax.gov.ua).

Сайт не має публічного API для Q&A, тому ми ітеруємо по ID. URL-шаблон:
    https://zir.tax.gov.ua/main/bz/view/?id=<ID>&src=ques

Сторінка містить:
- Категорію (напр. "101. Податок на додану вартість")
- Поле "Питання"
- Поле "Відповідь" (коротка і повна можуть бути однакові або різні)
- Статус (Чинне / Не чинне)

Якщо ID не існує — сервер повертає 200 з порожньою/дефолтною сторінкою.
Ми детектимо це через відсутність питання/відповіді.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm as atqdm

from zir_assistant.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ZIRItem:
    """Одна Q&A з ЗІР."""

    id: int
    url: str
    category: str | None
    question: str
    answer_short: str | None
    answer_full: str
    status: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class ZIRScraper:
    """Асинхронний скрейпер з семафором і затримками."""

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(settings.scrape_concurrency)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> ZIRScraper:
        headers = {
            "User-Agent": settings.scrape_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(settings.scrape_timeout_s),
            follow_redirects=True,
            http2=False,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _fetch(self, item_id: int) -> str | None:
        assert self._client is not None
        url = f"{settings.scrape_base_url}?id={item_id}&src=ques"
        async with self._semaphore:
            response = await self._client.get(url)
            await asyncio.sleep(settings.scrape_delay_ms / 1000)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.text

    def _parse(self, item_id: int, html: str) -> ZIRItem | None:
        """Витягнути Q&A з HTML. Повертає None якщо сторінка порожня/не існує.

        ЗІР має приховані input'и #expQues та #expAnsw (для експорту в PDF),
        що містять чистий текст питання і відповіді — це найнадійніше джерело.
        Fallback: витягнути з <fieldset class="bz_hide_fieldset ques/answ">.
        """
        soup = BeautifulSoup(html, "lxml")

        # Основний шлях: приховані інпути для експорту в PDF
        question = _value_of(soup, "#expQues")
        answer = _value_of(soup, "#expAnsw")

        # Fallback: legend + span у fieldset
        if not question:
            question = _fieldset_text(soup, "ques")
        if not answer:
            answer = _fieldset_text(soup, "answ")

        if not question or not answer:
            return None

        question_cleaned = _clean_html_text(question)
        answer_cleaned = _clean_html_text(answer)

        # Статус застарілості ("Діяло до DD.MM.YYYY") витягуємо ДО того, як почистити текст
        status = _extract_expiry_from_text(question_cleaned, answer_cleaned)
        question_cleaned = _strip_expiry_marker(question_cleaned)
        answer_cleaned = _strip_expiry_marker(answer_cleaned)

        # Типовий префікс дубля
        for prefix in (
            "Коротка та повна відповіді ідентичні.",
            "Коротка та повна відповіді ідентичні:",
        ):
            if answer_cleaned.startswith(prefix):
                answer_cleaned = answer_cleaned[len(prefix) :].strip()
                break

        # Якщо структура "Коротка: ... Повна: ..." — розділяємо
        answer_short: str | None = None
        for short_label, full_label in (
            ("Коротка відповідь:", "Повна відповідь:"),
            ("Коротка:", "Повна:"),
        ):
            if short_label in answer_cleaned and full_label in answer_cleaned:
                short_start = answer_cleaned.find(short_label)
                full_start = answer_cleaned.find(full_label)
                if short_start < full_start:
                    answer_short = answer_cleaned[short_start + len(short_label) : full_start].strip()
                    answer_cleaned = answer_cleaned[full_start + len(full_label) :].strip()
                    break

        return ZIRItem(
            id=item_id,
            url=f"{settings.scrape_base_url}?id={item_id}&src=ques",
            category=_extract_category(soup),
            question=question_cleaned,
            answer_short=answer_short,
            answer_full=answer_cleaned,
            status=status,
        )

    async def scrape_one(self, item_id: int) -> ZIRItem | None:
        try:
            html = await self._fetch(item_id)
        except httpx.HTTPStatusError as e:
            logger.warning("HTTP %s for id=%s", e.response.status_code, item_id)
            return None
        except Exception as e:
            logger.warning("Failed id=%s: %s", item_id, e)
            return None

        if html is None:
            return None
        return self._parse(item_id, html)

    async def scrape_range(
        self,
        start_id: int,
        max_id: int,
        output_path: Path,
        resume: bool = True,
    ) -> int:
        """Проскрейпити діапазон ID, записуючи кожну знайдену Q&A в JSONL."""
        seen_ids: set[int] = set()
        if resume and output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        seen_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            logger.info("Resume: %d вже збережено, пропускаємо", len(seen_ids))

        todo = [i for i in range(start_id, max_id + 1) if i not in seen_ids]
        logger.info("Скрейп %d ID (з %d до %d)", len(todo), start_id, max_id)

        found = 0
        with output_path.open("a", encoding="utf-8") as out:
            tasks = [self.scrape_one(i) for i in todo]
            for coro in atqdm.as_completed(tasks, total=len(tasks), desc="ЗІР"):
                item = await coro
                if item is None:
                    continue
                out.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
                out.flush()
                found += 1

        logger.info("Знайдено %d Q&A", found)
        return found


# -----------------------------------------------------------------------------
# HTML helpers


def _value_of(soup: BeautifulSoup, selector: str) -> str | None:
    """Повернути атрибут value тегу за CSS-селектором."""
    el = soup.select_one(selector)
    if el is None:
        return None
    val = el.get("value")
    if val is None or not str(val).strip():
        return None
    return str(val)


def _fieldset_text(soup: BeautifulSoup, kind: str) -> str | None:
    """Fallback: дістати текст із <fieldset class="bz_hide_fieldset {kind}">."""
    fs = soup.select_one(f"fieldset.bz_hide_fieldset.{kind}")
    if fs is None:
        return None
    # legend містить лейбл ("Питання" / "Відповідь"), нас цікавить все решта
    legend = fs.find("legend")
    if legend is not None:
        legend.decompose()
    # Замінюємо <br> на переноси рядків, щоб зберегти структуру
    for br in fs.find_all("br"):
        br.replace_with("\n")
    return fs.get_text(separator=" ", strip=True)


_STATUS_FILTER_VALUES = {"чинні", "не чинні", "всі", "чинна", "не чинна"}


def _extract_category(soup: BeautifulSoup) -> str | None:
    """Категорія — у <select> з selected option, що відповідає шаблону 'NNN. Назва'.
    Пропускаємо фільтр чинності, плейсхолдери та розділювачі."""
    import re

    category_re = re.compile(r"^\d{3}\.\s")
    for sel in soup.find_all("select"):
        selected = sel.find("option", attrs={"selected": True})
        if selected is None:
            continue
        text = selected.get_text(strip=True)
        if not text or text.lower() in _STATUS_FILTER_VALUES:
            continue
        if text.lower().startswith("обрати"):
            continue
        if text.startswith("-----") or text.endswith("-----"):
            continue
        if category_re.match(text):
            return text
    return None


_EXPIRED_RE = None


def _extract_expiry_from_text(*texts: str) -> str | None:
    """ЗІР маркує застарілі документи фразою 'Діяло до DD.MM.YYYY' / 'Діяла до ...'
    на початку питання і відповіді. Повертаємо найкращий збіг."""
    import re

    global _EXPIRED_RE
    if _EXPIRED_RE is None:
        _EXPIRED_RE = re.compile(r"Діял[аое]?\s+до\s+(\d{2}\.\d{2}\.\d{4})", re.IGNORECASE)
    for text in texts:
        if not text:
            continue
        m = _EXPIRED_RE.search(text)
        if m:
            return f"Не чинна з {m.group(1)}"
    return None


def _strip_expiry_marker(text: str) -> str:
    """Прибираємо 'Діяло до ...' з початку тексту, щоб воно не засмічувало питання/відповідь."""
    import re

    return re.sub(
        r"^\s*Діял[аое]?\s+до\s+\d{2}\.\d{2}\.\d{4}\s*[:.]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _clean_html_text(text: str) -> str:
    """Нормалізувати текст з HTML-джерел: прибрати <br>, &nbsp;, зайві пробіли."""
    import html as _html
    import re

    text = _html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


async def run_scrape(start_id: int, max_id: int) -> int:
    async with ZIRScraper() as scraper:
        return await scraper.scrape_range(start_id, max_id, settings.scraped_path)
