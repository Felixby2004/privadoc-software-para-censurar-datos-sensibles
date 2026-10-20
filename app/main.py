from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.pipeline import analyze_document, redact_analysis

BASE = Path(__file__).resolve().parent.parent
FRONTEND = BASE / "frontend"

app = FastAPI(title="Privadoc - Motor local de documentos peruanos", version="0.10.1")
CACHE: dict[str, dict[str, Any]] = {}
SUPPORTED = {"dni_azul", "dni_amarillo", "dni_electronico", "comprobantes"}


class DocumentSelection(BaseModel):
    document_id: str
    field_ids: list[str] = Field(default_factory=list)
    manual_boxes: list[list[int]] = Field(default_factory=list)


class RedactRequest(BaseModel):
    analysis_id: str
    selections: list[DocumentSelection] = Field(default_factory=list)
    mode: str = "black"


@app.get("/api/health")
def health():
    return {"ok": True, "mode": "local", "version": "0.10.1", "documents": sorted(SUPPORTED)}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), document_type: str = Form("dni_azul")):
    if document_type not in SUPPORTED:
        raise HTTPException(status_code=400, detail="Esta versión procesa DNI azul, DNI amarillo de menor de edad, DNI electrónico y comprobantes digitales.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="El archivo supera 30 MB.")
    try:
        result = analyze_document(raw, file.filename or "documento", file.content_type, document_type=document_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    aid = result["analysis_id"]
    CACHE[aid] = result
    public_docs = [d.public_dict() for d in result["documents"]]
    return JSONResponse({
        "analysis_id": aid,
        "filename": result["filename"],
        "source_is_pdf": result["source_is_pdf"],
        "page_count": result["page_count"],
        "document_count": len(public_docs),
        "document_type": result["document_type"],
        "ocr_engine": result["ocr_engine"],
        "documents": public_docs,
    })


@app.post("/api/redact")
def redact(req: RedactRequest):
    item = CACHE.get(req.analysis_id)
    if not item:
        raise HTTPException(status_code=404, detail="El análisis expiró. Vuelve a cargar el documento.")
    if not req.selections:
        raise HTTPException(status_code=400, detail="Selecciona al menos un dato o zona para censurar.")
    clean = [s.model_dump() for s in req.selections if s.field_ids or s.manual_boxes]
    if not clean:
        raise HTTPException(status_code=400, detail="Selecciona al menos un dato o zona para censurar.")
    try:
        _, preview, mime, payload, filename = redact_analysis(item, clean, mode=req.mode)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    b64 = base64.b64encode(payload).decode("ascii")
    return {"preview": preview, "download_data": f"data:{mime};base64,{b64}", "filename": filename, "mime": mime}


app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
