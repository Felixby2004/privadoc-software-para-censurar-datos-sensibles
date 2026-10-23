from __future__ import annotations

import asyncio
import base64
import gc
import logging
import multiprocessing
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.pipeline import analyze_document, redact_analysis


BASE = Path(__file__).resolve().parent.parent
FRONTEND = BASE / "frontend"

VERSION = "0.10.4"

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
            "2",
        )
    ),
)

JOB_QUEUE_TIMEOUT_SECONDS = max(
    1,
    int(
        os.getenv(
            "PRIVADOC_QUEUE_TIMEOUT",
            "5",
        )
    ),
)

JOB_TIMEOUT_SECONDS = max(
    30,
    int(
        os.getenv(
            "PRIVADOC_JOB_TIMEOUT",
            "180",
        )
    ),
)

LOW_MEMORY_MODE = os.getenv("PRIVADOC_LOW_MEMORY", "0") == "1"

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
            "16",
        )
    ),
)

CPU_SLOTS = asyncio.Semaphore(
    MAX_CONCURRENT_JOBS
)

CACHE_LOCK = asyncio.Lock()

CACHE: dict[str, dict[str, Any]] = {}


def _job_worker(
    connection: Any,
    function: Callable[..., Any],
    args: tuple[Any, ...],
) -> None:
    """Ejecuta OCR fuera del proceso web para poder terminar un trabajo atascado."""
    try:
        result = function(*args)
        connection.send(("ok", result))
    except BaseException as exc:
        try:
            connection.send(
                (
                    "error",
                    f"{type(exc).__name__}: {exc}",
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _run_isolated(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    timeout: int,
) -> Any:
    """Aísla OCR/OpenCV para que un bloqueo no deje inutilizable el servidor."""
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_job_worker,
        args=(child_connection, function, args),
    )
    process.start()
    child_connection.close()

    try:
        if not parent_connection.poll(timeout):
            if process.is_alive():
                process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            raise TimeoutError(
                f"El procesamiento superó el límite de {timeout} segundos."
            )

        try:
            status, payload = parent_connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                "El proceso de análisis terminó sin devolver un resultado."
            ) from exc

        if status == "error":
            raise RuntimeError(str(payload))
        return payload
    finally:
        parent_connection.close()
        if process.is_alive():
            process.terminate()
        process.join(5)


@asynccontextmanager
async def _job_slot():
    try:
        await asyncio.wait_for(
            CPU_SLOTS.acquire(),
            timeout=JOB_QUEUE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                "El servidor está procesando otros documentos. "
                "Espera unos segundos y vuelve a intentarlo."
            ),
        ) from exc

    try:
        yield
    finally:
        CPU_SLOTS.release()


async def _run_job(
    function: Callable[..., Any],
    args: tuple[Any, ...],
) -> Any:
    """Evita duplicar el modelo OCR en instancias Render con poca memoria."""
    if LOW_MEMORY_MODE:
        return await asyncio.wait_for(
            asyncio.to_thread(function, *args),
            timeout=JOB_TIMEOUT_SECONDS,
        )
    return await asyncio.to_thread(
        _run_isolated,
        function,
        args,
        JOB_TIMEOUT_SECONDS,
    )


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

        "queue_timeout_seconds":
            JOB_QUEUE_TIMEOUT_SECONDS,

        "job_timeout_seconds":
            JOB_TIMEOUT_SECONDS,

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

    async with _job_slot():

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

            result = await _run_job(
                analyze_document,
                (
                    raw,
                    file.filename
                    or "documento",
                    file.content_type,
                    document_type,
                ),
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

        except TimeoutError:

            logger.exception(
                "[PRIVADOC] Timeout durante analyze"
            )

            raise HTTPException(
                status_code=504,
                detail=(
                    "El análisis tardó demasiado y "
                    "se canceló para no bloquear el servidor."
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

    async with _job_slot():

        try:

            (
                _,
                preview,
                mime,
                payload,
                filename,
            ) = await _run_job(
                redact_analysis,
                (
                    item,
                    clean,
                    req.mode,
                ),
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

        except TimeoutError:

            logger.exception(
                "[PRIVADOC] Timeout durante redact"
            )

            raise HTTPException(
                status_code=504,
                detail=(
                    "La generación de la copia tardó demasiado "
                    "y se canceló para no bloquear el servidor."
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