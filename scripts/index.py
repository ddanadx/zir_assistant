"""Запуск побудови індексу.

Приклади:
    python -m scripts.index           # додати нові документи до існуючої БД
    python -m scripts.index --reset   # повністю перебудувати
"""

from __future__ import annotations

import argparse
import logging

from zir_assistant.indexer import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Побудова векторного індексу")
    parser.add_argument("--reset", action="store_true", help="видалити колекцію і побудувати заново")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    added = build_index(reset=args.reset)
    print(f"\n✓ Готово. Додано {added} документів.")


if __name__ == "__main__":
    main()
