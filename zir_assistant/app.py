"""FastAPI backend для ZIR Assistant.

Endpoint-и:
- GET  /            — HTML чат (статичний файл)
- POST /api/ask     — запит питання, відповідь Server-Sent Events зі streaming
- GET  /api/health  — перевірка, чи готова БД
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from zir_assistant.config import settings
from zir_assistant.rag import answer_stream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="ZIR Assistant", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    import chromadb

    try:
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        col = client.get_collection(settings.rag_collection_name)
        count = col.count()
        return {"status": "ok", "documents": count, "model": settings.gemini_chat_model}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


@app.post("/api/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Порожнє питання")

    async def sse_generator():
        """Формат SSE: кожна подія — рядок `data: {...}\\n\\n`."""
        try:
            async for event in answer_stream(question):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                # даємо event loop шанс відправити байти в мережу одразу
                await asyncio.sleep(0)
        except Exception as e:
            logger.exception("SSE failed")
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "zir_assistant.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
