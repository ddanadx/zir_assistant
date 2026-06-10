# ZIR Assistant

RAG-помічник по **[ЗІР](https://zir.tax.gov.ua)** (Загальнодоступний інформаційно-довідковий ресурс ДПС України) на базі Google Gemini.

Замість того щоб тренувати LLM з нуля, ми беремо готову мультимовну модель (Gemini) і «підсовуємо» їй релевантні Q&A з ЗІР через Retrieval-Augmented Generation. Це дає точні відповіді з **посиланнями на першоджерела**, актуальні до моменту останнього скрейпу.

## Архітектура

```
[користувач] → [FastAPI]
                   ↓
            [пошук у Chroma] ←── [Gemini embeddings]
                   ↓
      [топ-K релевантних Q&A з ЗІР]
                   ↓
         [Gemini 2.5 Flash + промпт]
                   ↓
    [відповідь зі streaming + цитати]
```

## Швидкий старт

### 1. Встановлення залежностей

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # якщо uv не встановлений
uv venv
uv sync
source .venv/bin/activate
```

### 2. Конфігурація

```bash
cp .env.example .env
# відредагуйте .env та додайте GEMINI_API_KEY
# ключ безкоштовно: https://aistudio.google.com/apikey
```

### 3. Скрейп ЗІР

```bash
python -m scripts.scrape --max-id 35000
```

Спочатку можна обмежитись меншим діапазоном для швидкої перевірки:
```bash
python -m scripts.scrape --start-id 1 --max-id 200
```

Результат: `data/zir.jsonl` — по одному Q&A на рядок.

### 4. Побудова індексу (ембединги + Chroma)

```bash
python -m scripts.index
```

Результат: `data/chroma/` — векторна БД.

### 5. Запуск сервера

```bash
python -m zir_assistant.app
```

Відкрити: http://127.0.0.1:8000/

## Структура проекту

```
zir_assistant/
├── config.py         # налаштування з .env
├── scraper.py        # асинхронний скрейпер ЗІР (httpx + BeautifulSoup)
├── indexer.py        # Gemini embeddings + ChromaDB
├── rag.py            # retrieve + generate зі streaming
├── app.py            # FastAPI backend (SSE для streaming)
└── static/
    └── index.html    # single-file чат UI

scripts/
├── scrape.py         # запуск скрейпу
└── index.py          # запуск побудови індексу
```

## Обмеження

- **Актуальність.** База знань ЗІР оновлюється — перезапускайте скрейп періодично.
- **Юридична дисклеймерність.** Помічник — довідковий, не замінює офіційну консультацію.
- **Rate limit Gemini.** На безкоштовному tier ~15 RPM / 1M TPM — достатньо для особистого використання.
- **Повага до сервера ЗІР.** Скрейпер робить затримки між запитами (`SCRAPE_DELAY_MS`).

## Ліцензія

MIT
