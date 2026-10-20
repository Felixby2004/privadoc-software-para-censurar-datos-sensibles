from __future__ import annotations

import asyncio
import base64
import gc
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.pipeline import analyze_document, redact_analysis


BASE = Path(__file__).resolve().parent.parent
FRONTEND = BASE / "frontend"

VERSION = "0.10.3"

SUPPORTED = {
    "dni_azul",
    "dni_amarillo",
    "dni_electronico",
    "comprobantes",
}


# -----------------------------------------------------------------------------
# CONTROL DE HILOS
# -----------------------------------------------------------------------------

try:
    cv2.setNumThreads(
        max(
            1,
            int(
                os.getenv(
                    "PRIVADOC_OPENCV_THREADS",
                    "1",
                )
            ),
        )
    )
except Exception:
    pass


app = FastAPI(
    title="Privadoc - Motor de documentos peruanos",
    version=VERSION,
)

logger = logging.getLogger("privadoc")


# -----------------------------------------------------------------------------
# CONCURRENCIA Y CACHE TEMPORAL
# -----------------------------------------------------------------------------

MAX_CONCURRENT_JOBS = max(
    1,
    int(
        os.getenv(
            "PRIVADOC_MAX_CONCURRENT",
            "1",
        )
    ),
)

CACHE_TTL_SECONDS = max(
    120,
    int(
        os.getenv(
            "PRIVADOC_CACHE_TTL",
            "600",
        )
    ),
)

CACHE_MAX_ITEMS = max(
    1,
    int(
        os.getenv(
            "PRIVADOC_CACHE_MAX",
            "2",
        )
    ),
)

CPU_SLOTS = asyncio.Semaphore(
    MAX_CONCURRENT_JOBS
)

CACHE_LOCK = asyncio.Lock()

CACHE: dict[str, dict[str, Any]] = {}


def _cleanup_cache_locked() -> None:
    now = time.monotonic()

    expired = [
        analysis_id
        for analysis_id, entry in CACHE.items()
        if float(
            entry.get(
                "expires_at",
                0,
            )
        ) <= now
    ]

    for analysis_id in expired:
        CACHE.pop(
            analysis_id,
            None,
        )


async def _cache_put(
    analysis_id: str,
    result: dict[str, Any],
) -> None:

    async with CACHE_LOCK:
        _cleanup_cache_locked()

        while len(CACHE) >= CACHE_MAX_ITEMS:

            oldest_id = min(
                CACHE,
                key=lambda key: float(
                    CACHE[key].get(
                        "created_at",
                        0,
                    )
                ),
            )

            CACHE.pop(
                oldest_id,
                None,
            )

        now = time.monotonic()

        CACHE[analysis_id] = {
            "created_at": now,
            "expires_at": (
                now
                + CACHE_TTL_SECONDS
            ),
            "result": result,
        }


async def _cache_get(
    analysis_id: str,
) -> dict[str, Any] | None:

    async with CACHE_LOCK:

        _cleanup_cache_locked()

        entry = CACHE.get(
            analysis_id
        )

        if entry is None:
            return None

        entry["expires_at"] = (
            time.monotonic()
            + CACHE_TTL_SECONDS
        )

        return entry["result"]


# -----------------------------------------------------------------------------
# MODELOS
# -----------------------------------------------------------------------------


class DocumentSelection(BaseModel):
    document_id: str

    field_ids: list[str] = Field(
        default_factory=list
    )

    manual_boxes: list[
        list[int]
    ] = Field(
        default_factory=list
    )


class RedactRequest(BaseModel):
    analysis_id: str

    selections: list[
        DocumentSelection
    ] = Field(
        default_factory=list
    )

    mode: str = "black"


# -----------------------------------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------------------------------


@app.get(
    "/health",
    include_in_schema=False,
)
async def health():
    """
    Health check ultraligero para Render.

    No toca OCR, cache, OpenCV ni locks.
    """
    return {
        "ok": True
    }


@app.get("/api/health")
async def api_health():
    """
    Diagnóstico manual de PRIVADOC.

    Este endpoint NO debe configurarse
    como health check de Render.
    """

    async with CACHE_LOCK:

        _cleanup_cache_locked()

        active_sessions = len(
            CACHE
        )

    return {
        "ok": True,

        "mode":
            "server"
            if os.getenv("RENDER")
            else "local",

        "version":
            VERSION,

        "documents":
            sorted(
                SUPPORTED
            ),

        "max_concurrent_jobs":
            MAX_CONCURRENT_JOBS,

        "active_sessions":
            active_sessions,

        "cache_max_items":
            CACHE_MAX_ITEMS,

        "cache_ttl_seconds":
            CACHE_TTL_SECONDS,

        "low_memory":
            os.getenv(
                "PRIVADOC_LOW_MEMORY",
                "0",
            )
            == "1",
    }


# -----------------------------------------------------------------------------
# ANÁLISIS
# -----------------------------------------------------------------------------


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    document_type: str = Form(
        "dni_azul"
    ),
):

    if document_type not in SUPPORTED:

        raise HTTPException(
            status_code=400,
            detail=(
                "Esta versión procesa "
                "DNI azul, "
                "DNI amarillo de menor de edad, "
                "DNI electrónico "
                "y comprobantes digitales."
            ),
        )

    async with CPU_SLOTS:

        raw = await file.read()

        if not raw:

            raise HTTPException(
                status_code=400,
                detail=(
                    "El archivo está vacío."
                ),
            )

        if (
            len(raw)
            > 30 * 1024 * 1024
        ):

            raise HTTPException(
                status_code=413,
                detail=(
                    "El archivo supera 30 MB."
                ),
            )

        try:

            started_at = (
                time.perf_counter()
            )

            logger.info(
                (
                    "[PRIVADOC] START analyze | "
                    "type=%s | "
                    "filename=%s | "
                    "size=%.2f MB"
                ),
                document_type,
                file.filename
                or "documento",
                len(raw)
                / (1024 * 1024),
            )

            result = await asyncio.to_thread(
                analyze_document,
                raw,
                file.filename
                or "documento",
                file.content_type,
                document_type,
            )

            logger.info(
                (
                    "[PRIVADOC] END analyze | "
                    "type=%s | "
                    "documents=%s | "
                    "elapsed=%.2fs"
                ),
                document_type,
                len(
                    result.get(
                        "documents",
                        [],
                    )
                ),
                (
                    time.perf_counter()
                    - started_at
                ),
            )

        except MemoryError:

            logger.exception(
                "[PRIVADOC] MemoryError durante analyze"
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "El servidor no tiene "
                    "memoria suficiente para "
                    "este archivo. Prueba con "
                    "una imagen o PDF de menor "
                    "resolución."
                ),
            )

        except Exception as exc:

            logger.exception(
                "[PRIVADOC] Error durante analyze"
            )

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        finally:

            raw = b""

            gc.collect()

    analysis_id = result[
        "analysis_id"
    ]

    await _cache_put(
        analysis_id,
        result,
    )

    public_docs = [
        document.public_dict()
        for document
        in result["documents"]
    ]

    return JSONResponse(
        {
            "analysis_id":
                analysis_id,

            "filename":
                result["filename"],

            "source_is_pdf":
                result[
                    "source_is_pdf"
                ],

            "page_count":
                result[
                    "page_count"
                ],

            "document_count":
                len(public_docs),

            "document_type":
                result[
                    "document_type"
                ],

            "ocr_engine":
                result[
                    "ocr_engine"
                ],

            "documents":
                public_docs,
        }
    )


# -----------------------------------------------------------------------------
# CENSURA
# -----------------------------------------------------------------------------


@app.post("/api/redact")
async def redact(
    req: RedactRequest
):

    item = await _cache_get(
        req.analysis_id
    )

    if item is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "El análisis expiró. "
                "Vuelve a cargar "
                "el documento."
            ),
        )

    if not req.selections:

        raise HTTPException(
            status_code=400,
            detail=(
                "Selecciona al menos "
                "un dato o zona "
                "para censurar."
            ),
        )

    clean = [
        selection.model_dump()
        for selection
        in req.selections
        if (
            selection.field_ids
            or selection.manual_boxes
        )
    ]

    if not clean:

        raise HTTPException(
            status_code=400,
            detail=(
                "Selecciona al menos "
                "un dato o zona "
                "para censurar."
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

        except MemoryError:

            raise HTTPException(
                status_code=503,
                detail=(
                    "El servidor no tuvo "
                    "memoria suficiente para "
                    "generar la copia protegida."
                ),
            )

        except Exception as exc:

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        finally:

            gc.collect()

    encoded = base64.b64encode(
        payload
    ).decode(
        "ascii"
    )

    del payload

    gc.collect()

    return {
        "preview":
            preview,

        "download_data":
            (
                f"data:{mime};"
                f"base64,{encoded}"
            ),

        "filename":
            filename,

        "mime":
            mime,
    }


# -----------------------------------------------------------------------------
# FRONTEND
# -----------------------------------------------------------------------------

app.mount(
    "/assets",
    StaticFiles(
        directory=FRONTEND
    ),
    name="assets",
)


@app.get("/")
def index():

    return FileResponse(
        FRONTEND
        / "index.html",

        headers={
            "Cache-Control":
                (
                    "no-store, "
                    "no-cache, "
                    "must-revalidate, "
                    "max-age=0"
                ),

            "Pragma":
                "no-cache",

            "Expires":
                "0",
        },
    )


@app.get(
    "/favicon.ico",
    include_in_schema=False,
)
def favicon():
    """
    Evita un 404 irrelevante si el navegador
    solicita un favicon y todavía no tenemos uno.
    """
    from fastapi import Response

    return Response(
        status_code=204
    )