from __future__ import annotations

import uuid
from typing import Any

import cv2
import numpy as np

from app.core.detector import detect_cards
from app.core.geometry import box_to_polygon, transform_points
from app.core.imaging import decode_upload, image_bytes, image_to_data_url, images_to_pdf
from app.documents.dni_blue.processor import process_blue_dni
from app.documents.dni_yellow.processor import process_yellow_dni
from app.documents.dni_electronic.processor import process_electronic_dni
from app.documents.receipts.processor import process_receipt


def analyze_document(data: bytes, filename: str, content_type: str | None, document_type: str = "dni_azul") -> dict[str, Any]:
    pages, is_pdf = decode_upload(data, filename, content_type)
    documents = []
    engines: set[str] = set()

    if document_type == "dni_amarillo":
        processor = process_yellow_dni
        color_hint = "yellow"
    elif document_type == "dni_electronico":
        processor = process_electronic_dni
        color_hint = "electronic"
    elif document_type == "comprobantes":
        processor = process_receipt
        color_hint = None
    else:
        processor = process_blue_dni
        color_hint = "blue"

    for page_index, page in enumerate(pages):
        if document_type == "comprobantes":
            doc, engine = processor(page, page_index)
            documents.append(doc)
            engines.add(engine)
            continue
        candidates = detect_cards(page, max_cards=6, color_hint=color_hint)
        for cand in candidates:
            doc, engine = processor(page, page_index, cand.quad, cand.confidence)
            documents.append(doc)
            engines.add(engine)

    if not documents:
        raise ValueError("No se pudo preparar ningún documento para análisis.")

    return {
        "analysis_id": str(uuid.uuid4()),
        "filename": filename,
        "source_is_pdf": is_pdf,
        "page_count": len(pages),
        "document_count": len(documents),
        "document_type": document_type,
        "ocr_engine": " + ".join(sorted(engines)),
        "documents": documents,
        "_pages": pages,
    }


def _apply_polygons(page: np.ndarray, polygons: list[np.ndarray], mode: str) -> np.ndarray:
    if not polygons:
        return page
    out = page.copy()
    mask = np.zeros(page.shape[:2], dtype=np.uint8)
    for poly in polygons:
        p = np.round(poly).astype(np.int32)
        cv2.fillConvexPoly(mask, p, 255)
    if mode == "blur":
        blurred = cv2.GaussianBlur(out, (61, 61), 0)
        out[mask > 0] = blurred[mask > 0]
    else:
        out[mask > 0] = (0, 0, 0)
    return out


def redact_analysis(item: dict[str, Any], selections: list[dict[str, Any]], mode: str = "black") -> tuple[list[np.ndarray], str, str, bytes, str]:
    pages = [p.copy() for p in item["_pages"]]
    docs_by_id = {d.id: d for d in item["documents"]}
    polygons_by_page: dict[int, list[np.ndarray]] = {}

    for sel in selections:
        doc = docs_by_id.get(sel.get("document_id"))
        if doc is None:
            continue
        chosen = set(sel.get("field_ids") or [])
        field_map = {f.id: f for f in doc.fields}
        for fid in chosen:
            f = field_map.get(fid)
            if f is None:
                continue
            masked_partial = getattr(f, "mask_box", None) is not None
            target_box = f.mask_box if masked_partial else f.box
            x1, y1, x2, y2 = target_box
            if masked_partial:
                px, py = 1, 1
            else:
                px = max(5, int((x2-x1)*0.05)); py = max(4, int((y2-y1)*0.15))
            poly = box_to_polygon(target_box, (px, py))
            page_poly = transform_points(poly, doc._canonical_to_page)
            polygons_by_page.setdefault(doc.page_index, []).append(page_poly)
        for box in (sel.get("manual_boxes") or [])[:50]:
            if not isinstance(box, list) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, x2 = sorted((max(0, min(doc.width, x1)), max(0, min(doc.width, x2))))
            y1, y2 = sorted((max(0, min(doc.height, y1)), max(0, min(doc.height, y2))))
            if x2-x1 < 3 or y2-y1 < 3:
                continue
            page_poly = transform_points(box_to_polygon([x1,y1,x2,y2]), doc._canonical_to_page)
            polygons_by_page.setdefault(doc.page_index, []).append(page_poly)

    for i, page in enumerate(pages):
        pages[i] = _apply_polygons(page, polygons_by_page.get(i, []), "blur" if mode == "blur" else "black")

    preview = image_to_data_url(pages[0])
    if item.get("source_is_pdf"):
        payload = images_to_pdf(pages)
        return pages, preview, "application/pdf", payload, "documento_protegido.pdf"
    payload = image_bytes(pages[0], "png")
    return pages, preview, "image/png", payload, "documento_protegido.png"
