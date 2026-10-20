from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

from app.core.models import OCRLine

try:
    from rapidocr import RapidOCR
except Exception:
    RapidOCR = None

try:
    import pytesseract
except Exception:
    pytesseract = None


_ENGINE = None
if RapidOCR is not None:
    try:
        _ENGINE = RapidOCR()
    except Exception:
        _ENGINE = None


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def norm_text(s: str) -> str:
    s = (s or "").upper().translate(str.maketrans("ÁÉÍÓÚÜÑ", "AEIOUUN"))
    s = re.sub(r"[^A-Z0-9<> ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _box_from_poly(poly: Any) -> list[int]:
    a = np.array(poly, dtype=float).reshape(-1, 2)
    x1, y1 = np.floor(a.min(axis=0)).astype(int)
    x2, y2 = np.ceil(a.max(axis=0)).astype(int)
    return [int(x1), int(y1), int(x2), int(y2)]


def _rapid(img: np.ndarray) -> list[OCRLine]:
    raw = _ENGINE(img)
    out: list[OCRLine] = []
    if hasattr(raw, "boxes") and hasattr(raw, "txts") and hasattr(raw, "scores"):
        boxes = getattr(raw, "boxes", None)
        txts = getattr(raw, "txts", None)
        scores = getattr(raw, "scores", None)
        if boxes is None or txts is None or scores is None:
            return out
        for poly, text, conf in zip(boxes, txts, scores):
            try:
                text = clean_text(str(text)); conf = float(conf)
                if text and conf >= 0.20:
                    out.append(OCRLine(text, conf, _box_from_poly(poly)))
            except Exception:
                pass
        return out
    result = raw[0] if isinstance(raw, tuple) else raw
    if not result:
        return out
    for item in result:
        try:
            poly, text, conf = item[0], clean_text(str(item[1])), float(item[2])
            if text and conf >= 0.20:
                out.append(OCRLine(text, conf, _box_from_poly(poly)))
        except Exception:
            pass
    return out


def _tesseract(img: np.ndarray) -> list[OCRLine]:
    if pytesseract is None:
        return []
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, config="--oem 3 --psm 11", output_type=pytesseract.Output.DICT, lang="spa+eng")
    grouped: dict[tuple[int, int, int], list[tuple[str, int, int, int, int, float]]] = {}
    for i, txt in enumerate(data.get("text", [])):
        text = clean_text(txt)
        if not text:
            continue
        try:
            conf = float(data["conf"][i]) / 100.0
        except Exception:
            conf = 0.0
        if conf < 0.20:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        grouped.setdefault(key, []).append((
            text, int(data["left"][i]), int(data["top"][i]),
            int(data["width"][i]), int(data["height"][i]), conf
        ))
    out: list[OCRLine] = []
    for items in grouped.values():
        items.sort(key=lambda x: x[1])
        text = " ".join(x[0] for x in items)
        x1 = min(x[1] for x in items); y1 = min(x[2] for x in items)
        x2 = max(x[1] + x[3] for x in items); y2 = max(x[2] + x[4] for x in items)
        conf = sum(x[5] for x in items) / len(items)
        out.append(OCRLine(text, conf, [x1, y1, x2, y2]))
    return out


def _run_single_engine(img: np.ndarray) -> tuple[list[OCRLine], str]:
    if _ENGINE is not None:
        return _rapid(img), "RapidOCR 3.x / ONNX Runtime local"
    out = _tesseract(img)
    if out:
        return out, "Tesseract local (respaldo)"
    raise RuntimeError("No hay un motor OCR disponible. Instala las dependencias del requirements.txt.")


def run_ocr(img: np.ndarray) -> tuple[list[OCRLine], str]:
    return _run_single_engine(img)


def _map_line_back(line: OCRLine, inv_scale: float) -> OCRLine:
    x1, y1, x2, y2 = line.box
    return OCRLine(
        line.text,
        line.confidence,
        [
            int(round(x1 * inv_scale)),
            int(round(y1 * inv_scale)),
            int(round(x2 * inv_scale)),
            int(round(y2 * inv_scale)),
        ],
    )


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter / ua)


def _merge_lines(lines: list[OCRLine]) -> list[OCRLine]:
    if not lines:
        return []
    kept: list[OCRLine] = []
    ordered = sorted(lines, key=lambda ln: (ln.confidence, len(norm_text(ln.text)), (ln.box[2]-ln.box[0])*(ln.box[3]-ln.box[1])), reverse=True)
    for ln in ordered:
        nt = norm_text(ln.text)
        if not nt:
            continue
        duplicate = False
        for old in kept:
            ont = norm_text(old.text)
            if _iou(old.box, ln.box) >= 0.55:
                if ont == nt or ont in nt or nt in ont:
                    duplicate = True
                    break
            ocx = (old.box[0] + old.box[2]) / 2.0
            ocy = (old.box[1] + old.box[3]) / 2.0
            lcx = (ln.box[0] + ln.box[2]) / 2.0
            lcy = (ln.box[1] + ln.box[3]) / 2.0
            if abs(ocx - lcx) <= 10 and abs(ocy - lcy) <= 10 and (ont == nt or ont in nt or nt in ont):
                duplicate = True
                break
        if not duplicate:
            kept.append(ln)
    kept.sort(key=lambda ln: (ln.box[1], ln.box[0]))
    return kept


def _resize_for_scale(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return img
    h, w = img.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (nw, nh), interpolation=interp)


def _enhanced_variants(img: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    variants = [img, cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)]
    if max(img.shape[:2]) <= 2600:
        adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
    return variants


def run_ocr_resilient(img: np.ndarray) -> tuple[list[OCRLine], str]:
    """OCR más tolerante a texto pequeño/grande y a capturas/PDFs reescalados.

    Mantiene `run_ocr` simple para OCR ROI repetitivo, y reserva esta versión para
    el OCR principal de página o documento normalizado.
    """
    h, w = img.shape[:2]
    max_side = max(h, w)

    engine_name = None
    collected: list[OCRLine] = []

    def _run_variant(var: np.ndarray, scale_inv: float = 1.0):
        nonlocal engine_name, collected
        try:
            lines, engine = _run_single_engine(var)
        except Exception:
            return []
        engine_name = engine_name or engine
        mapped = [_map_line_back(ln, scale_inv) for ln in lines]
        collected.extend(mapped)
        return mapped

    # 1) Pase base en imagen original.
    base = _run_variant(img, 1.0)
    merged = _merge_lines(collected)
    avg_conf = (sum(x.confidence for x in merged) / len(merged)) if merged else 0.0
    if len(merged) >= 12 and avg_conf >= 0.48:
        return merged, (engine_name or "RapidOCR 3.x / ONNX Runtime local") + " + multiescala"

    # 2) Rescate ligero con contraste local al mismo tamaño.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    _run_variant(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), 1.0)
    merged = _merge_lines(collected)
    avg_conf = (sum(x.confidence for x in merged) / len(merged)) if merged else 0.0
    if len(merged) >= 12 and avg_conf >= 0.46:
        return merged, (engine_name or "RapidOCR 3.x / ONNX Runtime local") + " + multiescala"

    # 3) Sólo si sigue corto, ampliar una vez. Esto ayuda a textos muy pequeños en PDF/capturas.
    if max_side <= 1000:
        scale = 2.1
    elif max_side <= 1600:
        scale = 1.55
    else:
        scale = 1.28
    scaled = _resize_for_scale(img, scale)
    inv = 1.0 / scale
    _run_variant(scaled, inv)
    if len(merged) < 8:
        sgray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        sclahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(sgray)
        adaptive = cv2.adaptiveThreshold(sclahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        _run_variant(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR), inv)

    merged = _merge_lines(collected)
    if merged:
        return merged, (engine_name or "RapidOCR 3.x / ONNX Runtime local") + " + multiescala"
    return _run_single_engine(img)



def _normalized_roi_to_box(roi: list[float] | tuple[float, float, float, float], width: int, height: int) -> list[int]:
    vals = [float(x) for x in roi]
    if max(vals) <= 1.5:
        x1 = int(round(vals[0] * width)); y1 = int(round(vals[1] * height))
        x2 = int(round(vals[2] * width)); y2 = int(round(vals[3] * height))
    else:
        x1, y1, x2, y2 = [int(round(x)) for x in vals]
    x1 = max(0, min(width - 1, x1)); y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2)); y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def _line_center_inside(line: OCRLine, box: list[int]) -> bool:
    cx = (line.box[0] + line.box[2]) / 2.0
    cy = (line.box[1] + line.box[3]) / 2.0
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _ocr_crop_variants(crop: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Genera pocas variantes útiles para un ROI esperado, sin barrer a ciegas toda la imagen."""
    h, w = crop.shape[:2]
    variants: list[tuple[np.ndarray, float]] = [(crop, 1.0)]
    target_h = 170
    if h < target_h:
        scale = min(3.0, max(1.4, target_h / max(1, h)))
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
        sharp = cv2.addWeighted(clahe, 1.45, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.45, 0)
        variants.append((cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR), scale))
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants.append((cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), 1.0))
    return variants


def refine_ocr_by_layout(
    img: np.ndarray,
    lines: list[OCRLine],
    field_defs: list[dict],
    max_regions: int = 10,
) -> list[OCRLine]:
    """Refuerza OCR sólo dentro de regiones de texto esperadas por la estructura del documento.

    La primera lectura sigue siendo global. Esta fase sólo rescanea ROIs de campos configurados
    cuando no existe ya texto de buena confianza dentro de la zona. Así mejora texto pequeño,
    PDF reescalado y capturas comprimidas sin reemplazar la lógica específica de cada DNI.
    """
    h, w = img.shape[:2]
    textual_categories = {"identifier", "personal", "location", "family", "health"}
    skip_tokens = ("photo", "foto", "firma", "signature", "mrz", "pdf417", "qr", "barcode", "codigo_barras", "chip", "huella", "finger")
    merged = list(lines)
    used = 0

    for fd in field_defs:
        if used >= max_regions:
            break
        fid = str(fd.get("id", "")).lower()
        category = str(fd.get("category", "")).lower()
        if category not in textual_categories:
            continue
        if any(tok in fid for tok in skip_tokens):
            continue
        roi = fd.get("ocr_roi") or fd.get("value_roi") or fd.get("roi")
        if not roi or len(roi) != 4:
            continue
        box = _normalized_roi_to_box(roi, w, h)
        area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(1.0, w * h)
        if area_ratio > 0.24:
            continue

        good = [ln for ln in merged if ln.confidence >= 0.62 and len(norm_text(ln.text)) >= 1 and _line_center_inside(ln, box)]
        if good:
            continue

        # Un margen moderado preserva el valor aunque la plantilla/registro se desplace unos píxeles.
        mx = max(4, int((box[2] - box[0]) * 0.08))
        my = max(3, int((box[3] - box[1]) * 0.20))
        x1 = max(0, box[0] - mx); y1 = max(0, box[1] - my)
        x2 = min(w, box[2] + mx); y2 = min(h, box[3] + my)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        candidates: list[OCRLine] = []
        for variant, scale in _ocr_crop_variants(crop):
            try:
                local, _ = run_ocr(variant)
            except Exception:
                local = []
            inv = 1.0 / scale
            for ln in local:
                bx1, by1, bx2, by2 = ln.box
                mapped = [
                    int(round(bx1 * inv)) + x1,
                    int(round(by1 * inv)) + y1,
                    int(round(bx2 * inv)) + x1,
                    int(round(by2 * inv)) + y1,
                ]
                if ln.confidence >= 0.22 and norm_text(ln.text):
                    candidates.append(OCRLine(ln.text, ln.confidence, mapped))
            if candidates:
                break
        if candidates:
            merged.extend(candidates)
        used += 1

    return _merge_lines(merged)

def run_ocr_for_scoring(img: np.ndarray, max_width: int = 900) -> list[OCRLine]:
    h, w = img.shape[:2]
    if w > max_width:
        s = max_width / w
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        small = img
    lines, _ = run_ocr(small)
    return lines
