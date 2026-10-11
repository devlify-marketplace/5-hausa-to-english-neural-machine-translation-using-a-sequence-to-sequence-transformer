"""Local FastAPI service; requests are processed in memory and not logged or stored."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hausa_nmt.predict import HausaTranslator

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("HAUSA_NMT_MODEL_DIR", str(ROOT / "artifacts" / "model")))
WEB_DIR = ROOT / "web"
WEB_FILE = WEB_DIR / "index.html"
app = FastAPI(title="Hausa-to-English NMT", version="1.0.0", description="Local research prototype for Hausa → English translation.")
app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000, description="Hausa text to translate")


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=20, description="One to twenty Hausa texts")


@lru_cache(maxsize=1)
def get_translator() -> HausaTranslator:
    return HausaTranslator(MODEL_DIR)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_FILE)


@app.get("/api/health")
def health():
    configured = (MODEL_DIR / "config.json").exists()
    ready = configured and (any(MODEL_DIR.glob("*.safetensors")) or any(MODEL_DIR.glob("pytorch_model*.bin")))
    return {"status": "ready" if ready else "model_unavailable", "direction": "hau-en", "model_path": str(MODEL_DIR)}


@app.post("/api/translate")
def translate(request: TranslateRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text must contain non-whitespace characters")
    try:
        result = get_translator().translate(request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"source_language": "Hausa", "target_language": "English", "source_text": request.text, "translation": result}


@app.post("/api/translate-batch")
def translate_batch(request: BatchRequest):
    if any(not text.strip() for text in request.texts):
        raise HTTPException(status_code=422, detail="Every text must contain non-whitespace characters")
    try:
        results = get_translator().translate_batch(request.texts)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"source_language": "Hausa", "target_language": "English", "translations": results}
