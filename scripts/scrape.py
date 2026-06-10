"""Запуск скрейпу ЗІР.

Приклади:
    python -m scripts.scrape                          # повний скрейп 1..SCRAPE_MAX_ID
    python -m scripts.scrape --start-id 1 --max-id 200   # діапазон
    python -m scripts.scrape --no-resume              # почати з нуля
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from zir_assistant.config import settings
from zir_assistant.scraper import ZIRScraper


async def _run(start_id: int, max_id: int, output: Path, resume: bool) -> None:
    async with ZIRScraper() as scraper:
        await scraper.scrape_range(start_id, max_id, output, resume=resume)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скрейпер ЗІР")
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--max-id", type=int, default=settings.scrape_max_id)
    parser.add_argument("--output", type=Path, default=settings.scraped_path)
    parser.add_argument("--no-resume", action="store_true", help="не продовжувати з існуючого файлу")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    asyncio.run(_run(args.start_id, args.max_id, args.output, resume=not args.no_resume))


if __name__ == "__main__":
    main()
