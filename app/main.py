from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.pipeline import analyze_document, redact_analysis


BASE = Path(__file__).resolve().parent.parent
FRONTEND = BASE / "frontend"

app = FastAPI(
    title="Privadoc - Motor de documentos peruanos",
    version="0.10.2",
)

SUPPORTED = {
    "dni_azul",
    "dni_amarillo",
    "dni_electronico",
    "comprobantes",
}


# ============================================================
# CONTROL DE CONCURRENCIA
# ============================================================

# Cantidad máxima de análisis/censuras pesadas ejecutándose
# al mismo tiempo.
#
# 2 es un buen punto de inicio para la evaluación.
MAX_CONCURRENT_JOBS = max(
    1,
    int(os.getenv("PRIVADOC_MAX_CONCURRENT", "2")),
)

CPU_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


# ============================================================
# CACHE TEMPORAL DE ANÁLISIS
# ============================================================

# Los análisis contienen imágenes y pueden consumir bastante RAM.
# Por eso no deben permanecer indefinidamente.

CACHE_TTL_SECONDS = max(
    300,
    int(os.getenv("PRIVADOC_CACHE_TTL", "1200")),
)

# 1200 segundos = 20 minutos

CACHE_MAX_ITEMS = max(
    2,
    int(os.getenv("PRIVADOC_CACHE_MAX", "8")),
)

CACHE: dict[str, dict[str, Any]] = {}

CACHE_LOCK = asyncio.Lock()


def _cleanup_cache_locked() -> None:
    """
    Elimina análisis expirados.

    Debe llamarse teniendo CACHE_LOCK adquirido.
    """
    now = time.monotonic()

    expired = [
        analysis_id
        for analysis_id, entry in CACHE.items()
        if entry["expires_at"] <= now
    ]

    for analysis_id in expired:
        CACHE.pop(analysis_id, None)


async def _cache_put(
    analysis_id: str,
    result: dict[str, Any],
) -> None:

    async with CACHE_LOCK:
        _cleanup_cache_locked()

        # Si llegamos al límite, quitar el análisis más antiguo.
        while len(CACHE) >= CACHE_MAX_ITEMS:
            oldest_id = min(
                CACHE,
                key=lambda k: CACHE[k]["created_at"],
            )

            CACHE.pop(oldest_id, None)

        now = time.monotonic()

        CACHE[analysis_id] = {
            "created_at": now,
            "expires_at": now + CACHE_TTL_SECONDS,
            "result": result,
        }


async def _cache_get(
    analysis_id: str,
) -> dict[str, Any] | None:

    async with CACHE_LOCK:
        _cleanup_cache_locked()

        entry = CACHE.get(analysis_id)

        if entry is None:
            return None

        # Si el usuario sigue trabajando, renovar la sesión.
        entry["expires_at"] = (
            time.monotonic() + CACHE_TTL_SECONDS
        )

        return entry["result"]


# ============================================================
# MODELOS API
# ============================================================


class DocumentSelection(BaseModel):
    document_id: str
    field_ids: list[str] = Field(default_factory=list)
    manual_boxes: list[list[int]] = Field(default_factory=list)


class RedactRequest(BaseModel):
    analysis_id: str
    selections: list[DocumentSelection] = Field(
        default_factory=list
    )
    mode: str = "black"


# ============================================================
# HEALTH
# ============================================================


@app.get("/api/health")
async def health():
    async with CACHE_LOCK:
        _cleanup_cache_locked()
        active_sessions = len(CACHE)

    return {
        "ok": True,
        "mode": "server",
        "version": "0.10.2",
        "documents": sorted(SUPPORTED),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "active_sessions": active_sessions,
    }


# ============================================================
# ANÁLISIS
# ============================================================


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    document_type: str = Form("dni_azul"),
):
    if document_type not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=(
                "Esta versión procesa DNI azul, "
                "DNI amarillo de menor de edad, "
                "DNI electrónico y comprobantes digitales."
            ),
        )

    # Esperar aquí evita cargar muchos archivos gigantes
    # simultáneamente en memoria.
    async with CPU_SLOTS:

        raw = await file.read()

        if not raw:
            raise HTTPException(
                status_code=400,
                detail="El archivo está vacío.",
            )

        if len(raw) > 30 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="El archivo supera 30 MB.",
            )

        try:
            # MUY IMPORTANTE:
            #
            # OCR + OpenCV + ONNX son trabajo pesado.
            # Los movemos fuera del event loop de FastAPI.
            result = await asyncio.to_thread(
                analyze_document,
                raw,
                file.filename or "documento",
                file.content_type,
                document_type,
            )

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )

    aid = result["analysis_id"]

    await _cache_put(
        aid,
        result,
    )

    public_docs = [
        document.public_dict()
        for document in result["documents"]
    ]

    return JSONResponse(
        {
            "analysis_id": aid,
            "filename": result["filename"],
            "source_is_pdf": result["source_is_pdf"],
            "page_count": result["page_count"],
            "document_count": len(public_docs),
            "document_type": result["document_type"],
            "ocr_engine": result["ocr_engine"],
            "documents": public_docs,
        }
    )


# ============================================================
# CENSURA
# ============================================================


@app.post("/api/redact")
async def redact(req: RedactRequest):

    item = await _cache_get(
        req.analysis_id
    )

    if item is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "El análisis expiró. "
                "Vuelve a cargar el documento."
            ),
        )

    if not req.selections:
        raise HTTPException(
            status_code=400,
            detail=(
                "Selecciona al menos un dato "
                "o zona para censurar."
            ),
        )

    clean = [
        selection.model_dump()
        for selection in req.selections
        if selection.field_ids
        or selection.manual_boxes
    ]

    if not clean:
        raise HTTPException(
            status_code=400,
            detail=(
                "Selecciona al menos un dato "
                "o zona para censurar."
            ),
        )

    async with CPU_SLOTS:

        try:
            (
                _,
                preview,
                mime,
                payload,
                filename,
            ) = await asyncio.to_thread(
                redact_analysis,
                item,
                clean,
                req.mode,
            )

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )

    b64 = base64.b64encode(
        payload
    ).decode("ascii")

    return {
        "preview": preview,
        "download_data": (
            f"data:{mime};base64,{b64}"
        ),
        "filename": filename,
        "mime": mime,
    }


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND),
    name="assets",
)


@app.get("/")
def index():
    return FileResponse(
        FRONTEND / "index.html"
    )