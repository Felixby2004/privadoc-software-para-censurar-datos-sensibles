from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
except Exception:
    pytesseract = None

from app.core.geometry import perspective_matrix, rotate_image, rotation_homography, warp_with_matrix
from app.core.imaging import image_to_data_url
from app.core.models import DocumentResult, OCRLine, SensitiveField
from app.core.ocr import norm_text, run_ocr, run_ocr_resilient, run_ocr_for_scoring, refine_ocr_by_layout

BASE = Path(__file__).resolve().parent
SPEC = json.loads((BASE / "spec.json").read_text(encoding="utf-8"))
CANON_W, CANON_H = [int(x) for x in SPEC["canonical_size"]]

_REFERENCE_CACHE: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}


def _roi_to_box(roi: list[float]) -> list[int]:
    x1, y1, x2, y2 = roi
    return [round(x1 * CANON_W), round(y1 * CANON_H), round(x2 * CANON_W), round(y2 * CANON_H)]


def _center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _center_in_roi(box: list[int], roi_box: list[int], margin: float = 0.04) -> bool:
    cx, cy = _center(box)
    rw = max(1.0, roi_box[2] - roi_box[0]); rh = max(1.0, roi_box[3] - roi_box[1])
    mx, my = rw * margin, rh * margin
    return (roi_box[0] - mx <= cx <= roi_box[2] + mx and roi_box[1] - my <= cy <= roi_box[3] + my)


def _inside_score(box: list[int], roi_box: list[int]) -> float:
    cx, cy = _center(box)
    if roi_box[0] <= cx <= roi_box[2] and roi_box[1] <= cy <= roi_box[3]:
        return 1.0
    dx = max(roi_box[0] - cx, 0, cx - roi_box[2])
    dy = max(roi_box[1] - cy, 0, cy - roi_box[3])
    diag = max(1.0, ((roi_box[2] - roi_box[0]) ** 2 + (roi_box[3] - roi_box[1]) ** 2) ** 0.5)
    return max(0.0, 1.0 - ((dx * dx + dy * dy) ** 0.5) / diag)


def _anchor_score(lines: list[OCRLine], anchors: list[str]) -> float:
    joined = " | ".join(norm_text(x.text) for x in lines)
    score = 0.0
    for anchor in anchors:
        a = norm_text(anchor)
        tokens = [t for t in a.split() if len(t) >= 3]
        if a in joined:
            score += 2.2
        elif tokens and sum(1 for t in tokens if t in joined) / len(tokens) >= 0.65:
            score += 1.0
    score += min(2.0, sum(1 for x in lines if x.confidence >= 0.60) / 18.0)
    return score


def _safe_ocr_for_scoring(img: np.ndarray) -> list[OCRLine]:
    try:
        return run_ocr_for_scoring(img)
    except Exception:
        return []


def _load_reference(side: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    cached = _REFERENCE_CACHE.get(side)
    if cached is not None:
        return cached
    ref_path = BASE / "references" / f"{side}.png"
    if not ref_path.exists():
        _REFERENCE_CACHE[side] = (None, None)
        return _REFERENCE_CACHE[side]
    ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if ref is None:
        _REFERENCE_CACHE[side] = (None, None)
        return _REFERENCE_CACHE[side]
    ref = cv2.resize(ref, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)
    mask_path = BASE / "references" / f"{side}_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    if mask is not None:
        mask = cv2.resize(mask, (CANON_W, CANON_H), interpolation=cv2.INTER_NEAREST)
    _REFERENCE_CACHE[side] = (ref, mask)
    return _REFERENCE_CACHE[side]


def _prep_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _register_reference(img: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, float, bool]:
    ref, mask = _load_reference(side)
    if ref is None:
        return img, np.eye(3, dtype=np.float64), 0.0, False

    g1 = _prep_gray(img)
    g2 = _prep_gray(ref)

    if mask is None:
        mask = np.full((CANON_H, CANON_W), 255, dtype=np.uint8)

    orb = cv2.ORB_create(nfeatures=3200, fastThreshold=7, edgeThreshold=15, patchSize=31)
    k1, d1 = orb.detectAndCompute(g1, mask)
    k2, d2 = orb.detectAndCompute(g2, mask)
    if d1 is None or d2 is None or len(k1) < 16 or len(k2) < 16:
        return img, np.eye(3, dtype=np.float64), 0.18, False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in knn if m.distance < 0.76 * n.distance]
    if len(good) < 12:
        return img, np.eye(3, dtype=np.float64), min(0.30, len(good) / 40.0), False

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or inlier_mask is None:
        return img, np.eye(3, dtype=np.float64), 0.24, False

    inliers = int(inlier_mask.ravel().sum())
    ratio = inliers / max(1, len(good))
    if inliers < 10 or ratio < 0.33:
        return img, np.eye(3, dtype=np.float64), max(0.22, min(0.38, ratio)), False

    aligned = cv2.warpPerspective(img, H, (CANON_W, CANON_H), flags=cv2.INTER_CUBIC)

    # Sanity check: projected corners should stay close to the canvas.
    corners = np.float32([[[0, 0]], [[CANON_W - 1, 0]], [[CANON_W - 1, CANON_H - 1]], [[0, CANON_H - 1]]])
    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    x1, y1 = proj.min(axis=0)
    x2, y2 = proj.max(axis=0)
    inside = max(0.0, min(CANON_W, x2) - max(0.0, x1)) * max(0.0, min(CANON_H, y2) - max(0.0, y1))
    bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
    frame_fit = max(0.0, min(1.0, inside / bbox_area))

    conf = max(0.45, min(0.99, 0.38 + 0.42 * ratio + 0.12 * min(1.0, inliers / 55.0) + 0.08 * frame_fit))
    return aligned, H.astype(np.float64), float(conf), True


def choose_orientation_and_side(card: np.ndarray) -> tuple[np.ndarray, int, str, float]:
    candidates: list[tuple[float, int, str, np.ndarray, float, float]] = []
    for angle in (0, 180):
        rotated = rotate_image(card, angle)
        lines = _safe_ocr_for_scoring(rotated)
        ocr_front = _anchor_score(lines, SPEC["front_anchors"]) if lines else 0.0
        ocr_back = _anchor_score(lines, SPEC["back_anchors"]) if lines else 0.0
        ocr_front_conf = max(0.0, min(1.0, ocr_front / 8.0))
        ocr_back_conf = max(0.0, min(1.0, ocr_back / 8.0))

        _, _, ref_front_conf, ref_front_used = _register_reference(rotated, "front")
        _, _, ref_back_conf, ref_back_used = _register_reference(rotated, "back")
        if not ref_front_used:
            ref_front_conf = 0.0
        if not ref_back_used:
            ref_back_conf = 0.0

        front_combined = (0.58 * ocr_front_conf + 0.42 * ref_front_conf) if (lines or ref_front_conf > 0) else 0.20
        back_combined = (0.58 * ocr_back_conf + 0.42 * ref_back_conf) if (lines or ref_back_conf > 0) else 0.20

        candidates.append((front_combined, angle, "front", rotated, ocr_front_conf, ref_front_conf))
        candidates.append((back_combined, angle, "back", rotated, ocr_back_conf, ref_back_conf))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else best
    combined, angle, side, rotated, ocr_conf, ref_conf = best
    margin = max(0.0, combined - second[0])
    conf = max(0.20, min(0.99, combined * 0.78 + margin * 0.40 + max(ocr_conf, ref_conf) * 0.12))
    return rotated, angle, side, conf


def _parse_date_value(s: str) -> str | None:
    text = (s or "").upper().strip()
    # DNI azul suele imprimir fechas como DD MM AAAA, pero OCR puede devolver /, -, . o espacios.
    m = re.search(r"(?<![0-9])([0-3]?[0-9])[^0-9A-Z]{0,3}([01]?[0-9])[^0-9A-Z]{0,3}((?:19|20)[0-9]{2})(?![0-9])", text)
    if m:
        try:
            dd, mm, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= dd <= 31 and 1 <= mm <= 12:
                return f"{dd:02d} {mm:02d} {yyyy:04d}"
        except Exception:
            pass
    # Respaldo para OCR que compacta la fecha: 18022011. No convertimos letras aquí
    # para evitar que las etiquetas produzcan falsos números.
    for compact in re.findall(r"(?<![0-9])([0-9]{8})(?![0-9])", text):
        try:
            dd, mm, yyyy = int(compact[:2]), int(compact[2:4]), int(compact[4:])
            if 1 <= dd <= 31 and 1 <= mm <= 12 and 1900 <= yyyy <= 2099:
                return f"{dd:02d} {mm:02d} {yyyy:04d}"
        except Exception:
            pass
    return None


def _date_like(s: str) -> bool:
    return _parse_date_value(s) is not None


def _mrz_like(s: str) -> bool:
    compact = re.sub(r"\s+", "", s.upper())
    return compact.count("<") >= 3 or bool(re.fullmatch(r"[A-Z0-9<]{20,}", compact))


def _dni8(s: str) -> str | None:
    compact = re.sub(r"[ .-]", "", s)
    m = re.search(r"(?<!\d)(\d{8})(?!\d)", compact)
    return m.group(1) if m else None


def _digits_from_ocr(s: str) -> str:
    table = str.maketrans({"O":"0","Q":"0","D":"0","I":"1","L":"1","S":"5","B":"8"})
    return re.sub(r"\D", "", (s or "").upper().translate(table))


def _ocr_crop_lines(img: np.ndarray, box: list[int], scale: float = 3.2) -> list[OCRLine]:
    """OCR local sobre un ROI estrecho, con variantes diseñadas para texto pequeño del DNI.

    El DNI azul tiene filas muy próximas. Por eso no agrandamos la ROI: agrandamos únicamente
    los píxeles contenidos en la ROI y generamos variantes de contraste/binarización. Así se evita
    que un campo capture la etiqueta o el valor de la fila vecina.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(CANON_W - 1, x1)); x2 = max(x1 + 1, min(CANON_W, x2))
    y1 = max(0, min(CANON_H - 1, y1)); y2 = max(y1 + 1, min(CANON_H, y2))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    variants = [
        enlarged,
        cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
    ]
    all_lines: list[OCRLine] = []
    for variant in variants:
        try:
            lines, _ = run_ocr(variant)
        except Exception:
            continue
        for ln in lines:
            bx1, by1, bx2, by2 = ln.box
            mapped = [
                int(round(x1 + bx1 / scale)), int(round(y1 + by1 / scale)),
                int(round(x1 + bx2 / scale)), int(round(y1 + by2 / scale)),
            ]
            all_lines.append(OCRLine(ln.text, ln.confidence, mapped))

    # Deduplicar por texto normalizado conservando la lectura de mayor confianza.
    best: dict[str, OCRLine] = {}
    for ln in all_lines:
        key = norm_text(ln.text)
        if not key:
            continue
        if key not in best or ln.confidence > best[key].confidence:
            best[key] = ln
    return list(best.values())

def _tesseract_text_variants(img: np.ndarray, whitelist: str | None = None, psm: int = 6) -> list[str]:
    if pytesseract is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    variants = [gray, cv2.createCLAHE(2.4, (8, 8)).apply(gray)]
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    out: list[str] = []
    cfg = f"--oem 3 --psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    for v in variants:
        try:
            txt = pytesseract.image_to_string(v, config=cfg, lang="eng")
        except Exception:
            continue
        txt = (txt or "").strip()
        if txt:
            out.append(txt)
    return out


def _rows_from_ocr_lines(lines: list[OCRLine]) -> list[str]:
    """Recompone filas de OCR ordenando fragmentos por posición.

    RapidOCR puede dividir la MRZ/cabecera en varios fragmentos. Para los campos
    estructurados nos interesa la fila completa, no cada fragmento aislado.
    """
    if not lines:
        return []
    items = sorted(lines, key=lambda x: ((_center(x.box)[1]), x.box[0]))
    rows: list[list[OCRLine]] = []
    for ln in items:
        cy = _center(ln.box)[1]
        h = max(1.0, ln.box[3] - ln.box[1])
        placed = False
        for row in rows:
            rcy = sum(_center(x.box)[1] for x in row) / len(row)
            rh = sum(max(1.0, x.box[3] - x.box[1]) for x in row) / len(row)
            if abs(cy - rcy) <= max(h, rh) * 0.85:
                row.append(ln)
                placed = True
                break
        if not placed:
            rows.append([ln])
    out: list[str] = []
    for row in rows:
        row.sort(key=lambda x: x.box[0])
        txt = "".join((x.text or "").strip().replace(" ", "") for x in row)
        if txt:
            out.append(txt)
    return out


def _parse_mrz_candidates(texts: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    lines: list[str] = []
    for text in texts:
        for raw in re.split(r"[\r\n|]+", (text or "").upper()):
            compact = re.sub(r"[^A-Z0-9<]", "", raw)
            if compact:
                lines.append(compact)

    # También intentamos sobre una concatenación porque algunos OCR cortan la primera fila.
    scan_lines = lines + (["".join(lines)] if lines else [])

    for line in scan_lines:
        # Forma ideal: I<PER76565960<0...
        patterns = [
            r"PER([0-9OQDISBL]{8})<+([0-9OQDISBL])",
            r"([0-9OQDISBL]{8})<+([0-9OQDISBL])",
            # Si OCR perdió el '<' entre número y dígito, PER sigue siendo un ancla fuerte.
            r"PER([0-9OQDISBL]{8})([0-9OQDISBL])",
        ]
        found = None
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                found = m
                break
        if not found:
            continue
        num = _digits_from_ocr(found.group(1))
        dv = _digits_from_ocr(found.group(2))
        if len(num) == 8 and len(dv) == 1:
            result["dni_number"] = num
            result["verification_digit"] = dv
            break

    # Segunda fila: YYMMDD + dígito + sexo + fecha de caducidad.
    for line in scan_lines:
        m = re.search(r"([0-9OQDISBL]{6})[0-9OQDISBL]([MF])([0-9OQDISBL]{6})", line)
        if not m:
            continue
        birth = _digits_from_ocr(m.group(1))
        if len(birth) == 6:
            try:
                yy, mm, dd = int(birth[:2]), int(birth[2:4]), int(birth[4:6])
                if 1 <= mm <= 12 and 1 <= dd <= 31:
                    year = 2000 + yy if yy <= 30 else 1900 + yy
                    result["birth_date"] = f"{dd:02d} {mm:02d} {year:04d}"
            except Exception:
                pass
        result["sexo"] = m.group(2)
        break

    # Tercera fila: APELLIDO<<NOMBRES. Se usa solo como respaldo.
    for line in lines:
        if any(ch.isdigit() for ch in line) or "<<" not in line:
            continue
        m = re.search(r"([A-Z]{3,})<<([A-Z<]{3,})", line)
        if not m:
            continue
        surname = m.group(1).strip("<")
        given = " ".join(x for x in m.group(2).split("<") if len(x) >= 2)
        if surname:
            result["apellido_paterno"] = surname
        if given:
            result["prenombres"] = given
        break
    return result


def _extract_mrz_identity(img: np.ndarray, full_lines: list[OCRLine] | None = None) -> dict[str, str]:
    mrz_fd = next((x for x in SPEC["front_fields"] if x.get("id") == "mrz"), None)
    if not mrz_fd:
        return {}
    box = _roi_to_box(mrz_fd["roi"])
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    texts: list[str] = []

    crop_lines = _ocr_crop_lines(img, box, scale=2.4)
    texts.extend(x.text for x in crop_lines)
    texts.extend(_rows_from_ocr_lines(crop_lines))

    if full_lines:
        mrz_full = [x for x in full_lines if _inside_score(x.box, box) >= 0.55]
        texts.extend(x.text for x in mrz_full)
        texts.extend(_rows_from_ocr_lines(mrz_full))

    texts.extend(_tesseract_text_variants(crop, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<", psm=6))
    texts.extend(_tesseract_text_variants(crop, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<", psm=11))
    return _parse_mrz_candidates(texts)


def _parse_header_identity_texts(texts: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    candidates: list[str] = []
    for text in texts:
        up = re.sub(r"\s+", "", (text or "").upper())
        if up:
            candidates.append(up)
    if candidates:
        candidates.append("".join(candidates))

    for up in candidates:
        patterns = [
            r"DNI[^0-9OQDISBL]*([0-9OQDISBL]{8})[-<]?([0-9OQDISBL])?",
            r"([0-9OQDISBL]{8})[-<]([0-9OQDISBL])",
        ]
        for pat in patterns:
            m = re.search(pat, up)
            if not m:
                continue
            num = _digits_from_ocr(m.group(1))
            if len(num) != 8:
                continue
            result["dni_number"] = num
            if m.lastindex and m.lastindex >= 2 and m.group(2):
                dv = _digits_from_ocr(m.group(2))
                if len(dv) == 1:
                    result["verification_digit"] = dv
            # Buscar un dígito verificador justo después del número si el patrón genérico no lo capturó.
            if "verification_digit" not in result:
                pos = m.end(1)
                tail = up[pos:pos + 5]
                mdv = re.search(r"[-<]+([0-9OQDISBL])", tail)
                if mdv:
                    dv = _digits_from_ocr(mdv.group(1))
                    if len(dv) == 1:
                        result["verification_digit"] = dv
            return result

    # Respaldo conservador: aceptar un token aislado de exactamente 8 dígitos
    # (no una concatenación de OCR), evitando fabricar un DNI desde ruido textual.
    for up in candidates[:-1] if len(candidates) > 1 else candidates:
        if re.fullmatch(r"[0-9OQDISBL]{8}", up):
            num = _digits_from_ocr(up)
            if len(num) == 8:
                result["dni_number"] = num
                return result
    return result


def _extract_header_identity(img: np.ndarray, full_lines: list[OCRLine] | None = None) -> dict[str, str]:
    # Cabecera superior derecha, por encima del bloque de fechas.
    box = _roi_to_box([0.69, 0.005, 0.998, 0.115])
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    lines = _ocr_crop_lines(img, box, scale=3.4)
    texts = [x.text for x in lines]
    texts.extend(_rows_from_ocr_lines(lines))
    if full_lines:
        top = [x for x in full_lines if _inside_score(x.box, box) >= 0.50]
        texts.extend(x.text for x in top)
        texts.extend(_rows_from_ocr_lines(top))
    texts.extend(_tesseract_text_variants(crop, whitelist="DNI0123456789-<", psm=7))
    texts.extend(_tesseract_text_variants(crop, whitelist="DNI0123456789-<", psm=6))
    return _parse_header_identity_texts(texts)


def _best_crop_value(img: np.ndarray, box: list[int], kind: str = "text") -> OCRLine | None:
    # Los valores pequeños (sexo/estado civil) necesitan mayor ampliación que nombres o fechas.
    scale = 4.2 if kind in {"single_letter", "sex_letter", "civil_letter"} else (3.8 if kind == "ubigeo6" else 3.2)
    lines = _ocr_crop_lines(img, box, scale=scale)

    # Candidatos de fila: RapidOCR puede separar un nombre compuesto o un código en varios tokens.
    row_candidates: list[OCRLine] = []
    if lines:
        ordered = sorted(lines, key=lambda x: (x.box[1], x.box[0]))
        # Como el ROI ya es de una sola línea, agrupamos todos los tokens verticalmente compatibles.
        clusters: list[list[OCRLine]] = []
        for ln in ordered:
            cy = _center(ln.box)[1]
            h = max(1.0, ln.box[3] - ln.box[1])
            target = None
            for cluster in clusters:
                ccy = sum(_center(x.box)[1] for x in cluster) / len(cluster)
                ch = sum(max(1.0, x.box[3] - x.box[1]) for x in cluster) / len(cluster)
                if abs(cy - ccy) <= max(h, ch) * 0.85:
                    target = cluster
                    break
            if target is None:
                clusters.append([ln])
            else:
                target.append(ln)
        for cluster in clusters:
            cluster.sort(key=lambda x: x.box[0])
            text = " ".join((x.text or "").strip() for x in cluster if (x.text or "").strip())
            if not text:
                continue
            x1 = min(x.box[0] for x in cluster); y1 = min(x.box[1] for x in cluster)
            x2 = max(x.box[2] for x in cluster); y2 = max(x.box[3] for x in cluster)
            conf = sum(float(x.confidence) for x in cluster) / len(cluster)
            row_candidates.append(OCRLine(text, conf, [x1, y1, x2, y2]))

    if kind == "date":
        date_candidates: list[tuple[float, OCRLine]] = []
        for ln in lines + row_candidates:
            parsed = _parse_date_value(ln.text)
            if parsed:
                date_box = list(ln.box)
                dh = max(1, date_box[3] - date_box[1])
                # Si OCR agrupó la etiqueta "Nacimiento" con la fecha, conservar la franja
                # inferior donde está impreso el valor.
                if dh > 30:
                    date_box[1] = date_box[1] + int(round(dh * 0.30))
                date_candidates.append((float(ln.confidence), OCRLine(parsed, ln.confidence, date_box)))
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        for text in _tesseract_text_variants(crop, whitelist="0123456789 /.-", psm=7):
            parsed = _parse_date_value(text)
            if parsed:
                date_candidates.append((0.76, OCRLine(parsed, 0.76, box)))
        return max(date_candidates, key=lambda x: x[0])[1] if date_candidates else None

    candidates: list[tuple[float, OCRLine]] = []
    for ln in lines + row_candidates:
        txt = (ln.text or "").strip()
        nt = norm_text(txt)
        if not nt or _is_static_label(txt):
            continue
        normalized_value = txt
        bonus = 0.0

        if kind == "letters":
            # Nombres/apellidos: nunca aceptar fechas, códigos o residuos numéricos.
            if re.search(r"\d", txt):
                continue
            letters = re.sub(r"[^A-Z ]", "", nt)
            compact = letters.replace(" ", "")
            if len(compact) < 2:
                continue
            if len(compact) / max(1, len(nt.replace(" ", ""))) < 0.82:
                continue
            normalized_value = re.sub(r"\s+", " ", letters).strip()
            # Para nombres importa más recuperar la palabra completa que una subcadena con
            # confianza artificialmente alta. Esto penaliza lecturas como "FRANC" frente a
            # "FRANCO IMANOL" cuando ambas aparecen en las variantes del mismo recorte.
            coverage = min(1.0, len(compact) / 12.0)
            score = 0.65 * float(ln.confidence) + 0.35 * coverage
            if " " in normalized_value:
                score += 0.04
            if score >= 0.34:
                candidates.append((score, OCRLine(normalized_value, min(0.99, score), ln.box)))
            continue

        elif kind == "ubigeo6":
            d = _digits_from_ocr(txt)
            if len(d) != 6:
                continue
            normalized_value = d
            bonus += 0.10

        elif kind in {"single_letter", "sex_letter", "civil_letter"}:
            compact = re.sub(r"[^A-Z]", "", nt)
            allowed = {"M", "F"} if kind == "sex_letter" else ({"S", "C", "V", "D"} if kind == "civil_letter" else {"M", "F", "S", "C", "V", "D"})
            if len(compact) != 1 or compact not in allowed:
                continue
            # En el DNI azul el valor de Sexo / Estado civil va debajo de la etiqueta.
            # Si OCR aísla una sola letra dentro de la propia palabra "Sexo" o "Estado",
            # su centro queda demasiado arriba y termina dibujando la caja sobre la etiqueta.
            cy = _center(ln.box)[1]
            rel = max(0.0, min(1.0, (cy - box[1]) / max(1.0, box[3] - box[1])))
            if rel < 0.48:
                continue
            normalized_value = compact
            bonus += 0.12

        score = float(ln.confidence) + bonus
        if kind in {"single_letter", "sex_letter", "civil_letter"}:
            # El valor (M/F/S/...) está impreso debajo de la etiqueta. Favorecer tokens
            # situados en la mitad inferior del recorte y penalizar saltos hacia arriba.
            cy = _center(ln.box)[1]
            rel = max(0.0, min(1.0, (cy - box[1]) / max(1.0, box[3] - box[1])))
            score += 0.20 * rel
        if score >= 0.34:
            candidates.append((score, OCRLine(normalized_value, min(0.99, score), ln.box)))

    # Respaldo Tesseract solo si está disponible localmente; el programa no depende de él.
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    if kind == "ubigeo6":
        for text in _tesseract_text_variants(crop, whitelist="0123456789", psm=7):
            d = _digits_from_ocr(text)
            if len(d) == 6:
                candidates.append((0.80, OCRLine(d, 0.80, box)))
    elif kind in {"single_letter", "sex_letter", "civil_letter"}:
        whitelist = "MF" if kind == "sex_letter" else ("SCVD" if kind == "civil_letter" else "MFSCVD")
        allowed = set(whitelist)
        for text in _tesseract_text_variants(crop, whitelist=whitelist, psm=10):
            compact = re.sub(r"[^A-Z]", "", text.upper())
            if len(compact) == 1 and compact in allowed:
                # PSM 10 solo devuelve el texto, no su bbox. En este formato la letra está
                # debajo de la etiqueta, así que usamos únicamente la zona inferior del recorte
                # en vez de censurar toda la banda etiqueta+valor.
                bh = max(1, y2 - y1)
                tight = [x1, y1 + int(round(bh * 0.42)), x2, y2]
                candidates.append((0.80, OCRLine(compact, 0.80, tight)))
    elif kind == "letters":
        for text in _tesseract_text_variants(crop, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ ", psm=7):
            val = norm_text(text)
            compact = val.replace(" ", "")
            if val and not re.search(r"\d", val) and len(compact) >= 2:
                coverage = min(1.0, len(compact) / 12.0)
                score = 0.72 * 0.76 + 0.28 * coverage + (0.04 if " " in val else 0.0)
                candidates.append((score, OCRLine(val, min(0.99, score), box)))

    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def _snap_box_to_text_band(img: np.ndarray, box: list[int], expand_y: float = 0.045) -> list[int]:
    """Ajusta verticalmente una ROI al renglón de tinta más probable.

    La plantilla aporta la columna correcta, pero pequeñas deformaciones locales de la tarjeta
    pueden desplazar hacia arriba/abajo los valores de las últimas filas. Este refinamiento
    conserva X y busca la banda horizontal de trazos oscuros más cercana al centro esperado.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1)); x2 = max(x1 + 1, min(w, x2))
    base_h = max(8, y2 - y1)
    extra = max(10, int(round(expand_y * h)))
    sy1 = max(0, y1 - extra)
    sy2 = min(h, y2 + extra)
    crop = img[sy1:sy2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return box

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # El texto del DNI es oscuro sobre un fondo azul/celeste. Black-hat realza trazos oscuros
    # sin depender tanto de la iluminación global o de la saturación del documento.
    kx = max(5, min(17, (crop.shape[1] // 20) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Reducir ruido fino vertical del patrón de seguridad y unir letras en la misma línea.
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    row = (bw > 0).sum(axis=1).astype(np.float32)
    if row.max() <= 0:
        return box
    smooth_k = max(3, min(11, (base_h // 4) * 2 + 1))
    smooth = np.convolve(row, np.ones(smooth_k, dtype=np.float32) / smooth_k, mode='same')

    expected = ((y1 + y2) / 2.0) - sy1
    yy = np.arange(len(smooth), dtype=np.float32)
    sigma = max(14.0, base_h * 1.4)
    proximity = np.exp(-0.5 * ((yy - expected) / sigma) ** 2)
    score = smooth * (0.55 + 0.45 * proximity)
    peak = int(np.argmax(score))
    if score[peak] < max(1.0, float(smooth.max()) * 0.22):
        return box

    # Delimitar la banda alrededor del pico. Buscamos una altura compatible con una línea de texto,
    # no con la etiqueta de la fila vecina.
    threshold = max(float(smooth[peak]) * 0.34, float(np.percentile(smooth, 70)))
    top = peak
    bottom = peak
    while top > 0 and smooth[top - 1] >= threshold:
        top -= 1
    while bottom + 1 < len(smooth) and smooth[bottom + 1] >= threshold:
        bottom += 1

    target_h = bottom - top + 1
    min_h = max(10, int(base_h * 0.42))
    max_h = max(min_h + 2, int(base_h * 0.95))
    if target_h < min_h:
        half = min_h // 2
        top = max(0, peak - half)
        bottom = min(len(smooth) - 1, top + min_h - 1)
        top = max(0, bottom - min_h + 1)
    elif target_h > max_h:
        half = max_h // 2
        top = max(0, peak - half)
        bottom = min(len(smooth) - 1, top + max_h - 1)
        top = max(0, bottom - max_h + 1)

    pad = 3
    ny1 = max(0, sy1 + top - pad)
    ny2 = min(h, sy1 + bottom + pad + 1)

    # Guardas: no permitir saltos absurdos causados por la textura de seguridad.
    old_center = (y1 + y2) / 2.0
    new_center = (ny1 + ny2) / 2.0
    if abs(new_center - old_center) > max(extra * 0.95, base_h * 1.15):
        return box
    if ny2 - ny1 < 7:
        return box
    return [x1, int(ny1), x2, int(ny2)]


def _find_label_value(lines: list[OCRLine], aliases: list[str], roi: list[int]) -> OCRLine | None:
    aliases_n = [norm_text(a) for a in aliases]
    labels: list[OCRLine] = []
    for ln in lines:
        nt = norm_text(ln.text)

        def alias_matches(a: str) -> bool:
            if a in nt:
                return True
            tokens = [t for t in a.split() if len(t) >= 3]
            if not tokens:
                return False
            needed = max(1, int(np.ceil(len(tokens) * 0.75)))
            return sum(t in nt for t in tokens) >= needed

        if any(alias_matches(a) for a in aliases_n):
            labels.append(ln)
    candidates: list[tuple[float, OCRLine]] = []
    for lab in labels:
        lx1, ly1, lx2, ly2 = lab.box
        lh = max(1, ly2 - ly1)
        for ln in lines:
            if ln is lab:
                continue
            x1, y1, x2, y2 = ln.box
            same_row = abs((y1 + y2) / 2 - (ly1 + ly2) / 2) < lh * 1.2 and x1 >= lx2 - 15
            below = y1 >= ly2 - 5 and y1 <= ly2 + lh * 3.0 and x2 >= lx1 - 25
            if not (same_row or below):
                continue
            if not _center_in_roi(ln.box, roi, 0.05):
                continue
            loc = _inside_score(ln.box, roi)
            nt = norm_text(ln.text)
            if any(a == nt for a in aliases_n):
                continue
            dist = max(0, x1 - lx2) + 1.5 * max(0, y1 - ly2)
            candidates.append((dist - 120 * loc - 40 * ln.confidence, ln))
    if candidates:
        return min(candidates, key=lambda x: x[0])[1]
    return None


def _is_static_label(text: str) -> bool:
    nt = norm_text(text)
    labels: list[str] = []
    for group in (SPEC["front_fields"], SPEC["back_fields"]):
        for fd in group:
            labels.extend(fd.get("labels") or [])
    for label in labels:
        a = norm_text(label)
        if a and (a in nt or nt in a):
            return True
        tokens = [t for t in a.split() if len(t) >= 4]
        if tokens and sum(t in nt for t in tokens) >= max(1, int(np.ceil(len(tokens) * 0.75))):
            return True
    return False


def _best_value_in_roi(lines: list[OCRLine], roi: list[int], predicate=None) -> OCRLine | None:
    rh = max(1, roi[3] - roi[1])
    cands: list[tuple[float, OCRLine]] = []
    for ln in lines:
        if not _center_in_roi(ln.box, roi, 0.03):
            continue
        loc = _inside_score(ln.box, roi)
        if _is_static_label(ln.text):
            continue
        if predicate and not predicate(ln.text):
            continue
        h = max(1, ln.box[3] - ln.box[1])
        rel_h = min(1.0, h / max(12.0, rh * 0.30))
        rel_y = max(0.0, min(1.0, ((_center(ln.box)[1] - roi[1]) / rh)))
        score = loc * 0.48 + ln.confidence * 0.27 + rel_h * 0.18 + rel_y * 0.07
        cands.append((score, ln))
    return max(cands, key=lambda x: x[0])[1] if cands else None


def _best_line_in_roi(lines: list[OCRLine], roi: list[int], predicate=None) -> OCRLine | None:
    cands: list[tuple[float, OCRLine]] = []
    for ln in lines:
        loc = _inside_score(ln.box, roi)
        if loc < 0.25:
            continue
        if predicate and not predicate(ln.text):
            continue
        score = loc * 0.72 + ln.confidence * 0.28
        cands.append((score, ln))
    return max(cands, key=lambda x: x[0])[1] if cands else None


def extract_fields(img: np.ndarray, lines: list[OCRLine], side: str, registration_confidence: float, reference_used: bool) -> list[SensitiveField]:
    defs = SPEC["front_fields"] if side == "front" else SPEC["back_fields"]
    fields: list[SensitiveField] = []
    mrz_identity = _extract_mrz_identity(img, lines) if side == "front" else {}
    header_identity = _extract_header_identity(img, lines) if side == "front" else {}

    def add_field(fd: dict, box: list[int], value: str, confidence: float, source: str) -> None:
        final_box = list(box)
        if side == "front" and fd.get("snap_to_text") and not source.startswith("OCR ROI"):
            final_box = _snap_box_to_text_band(
                img, final_box, float(fd.get("snap_expand_y", 0.045))
            )
        fields.append(SensitiveField(
            id=fd["id"], name=fd["name"], value=value,
            risk=fd["risk"], recommendation="Censurar antes de compartir si este dato no es necesario.",
            box=final_box, confidence=round(max(0.22, min(0.99, confidence)), 3),
            category=fd.get("category", "personal"),
            default_redact=bool(fd.get("default_redact", True)), source=source
        ))

    for fd in defs:
        roi = _roi_to_box(fd["roi"])
        value_roi = _roi_to_box(fd.get("value_roi", fd["roi"]))
        ocr_roi = _roi_to_box(fd.get("ocr_roi", fd.get("value_roi", fd["roi"])))
        fid = fd["id"]
        pattern = fd.get("pattern")

        # Identity block: header OCR is preferred, MRZ is the strong fallback/validator.
        if side == "front" and fid in {"dni_number", "verification_digit"}:
            hv = header_identity.get(fid)
            mv = mrz_identity.get(fid)
            if hv and mv and hv == mv:
                add_field(fd, value_roi, hv, 0.98 * registration_confidence + 0.01, "OCR cabecera + MRZ")
            elif mv:
                # Value is recovered from MRZ but censoring still targets the printed header position.
                add_field(fd, value_roi, mv, 0.94 * registration_confidence, "MRZ + plantilla")
            elif hv:
                add_field(fd, value_roi, hv, 0.86 * registration_confidence, "OCR cabecera + plantilla")
            elif fd.get("geometry_fallback"):
                add_field(fd, value_roi, "Región esperada por plantilla", 0.86 * registration_confidence, "plantilla")
            continue

        # MRZ itself is always a known geometric sensitive zone. OCR text is useful but not required.
        if side == "front" and fid == "mrz":
            mrz_lines = [x for x in lines if _inside_score(x.box, roi) >= 0.70 and _mrz_like(x.text)]
            if mrz_lines:
                x1 = min(x.box[0] for x in mrz_lines); y1 = min(x.box[1] for x in mrz_lines)
                x2 = max(x.box[2] for x in mrz_lines); y2 = max(x.box[3] for x in mrz_lines)
                add_field(fd, [x1, y1, x2, y2], " / ".join(x.text for x in mrz_lines), 0.90 * registration_confidence, "OCR + plantilla")
            else:
                add_field(fd, roi, "Zona MRZ identificada", 0.90 * registration_confidence, "plantilla")
            continue

        # Strong MRZ fields: date of birth and sex are structurally encoded and are safer than
        # borrowing OCR from neighbouring labels.
        if side == "front" and fid in {"birth_date", "sexo"} and mrz_identity.get(fid):
            mrz_val = mrz_identity[fid]
            kind = "date" if fid == "birth_date" else "sex_letter"
            local = _best_crop_value(img, ocr_roi, kind) if registration_confidence >= 0.80 else None
            local_val = None
            if local is not None:
                local_val = (_parse_date_value(local.text) or local.text) if kind == "date" else norm_text(local.text)
            if local is not None and local_val == mrz_val:
                # El OCR valida el valor, pero para campos minúsculos como Sexo la caja OCR
                # es demasiado inestable. La censura usa siempre la geometría de plantilla.
                add_field(fd, value_roi, mrz_val, 0.96 * registration_confidence, "OCR ROI + MRZ + plantilla")
            else:
                add_field(fd, value_roi, mrz_val, 0.93 * registration_confidence, "MRZ + plantilla")
            continue

        # Tight value ROIs: OCR only inside the value rectangle. If registration is weak, do not
        # invent a value from a displaced crop; prefer the geometric fallback.
        if fd.get("value_roi"):
            kind = "text"
            if fid in {"apellido_paterno", "apellido_materno", "prenombres"}:
                kind = "letters"
            elif pattern == "ubigeo6":
                kind = "ubigeo6"
            elif pattern == "date" or fid == "birth_date":
                kind = "date"
            elif fid == "sexo":
                kind = "sex_letter"
            elif fid == "estado_civil":
                kind = "civil_letter"
            ln = _best_crop_value(img, ocr_roi, kind) if registration_confidence >= 0.80 else None
            if ln is not None:
                val = ln.text.strip()
                if fid in {"apellido_paterno", "apellido_materno", "prenombres"}:
                    val = norm_text(val)
                if pattern == "ubigeo6":
                    val = _digits_from_ocr(val)
                if kind == "date":
                    val = _parse_date_value(val) or val

                # Primer apellido y prenombres también están codificados en la MRZ. Si el OCR de
                # la zona visual devuelve solo un fragmento (p. ej. "FRANC"), la MRZ evita mostrar
                # un valor incompleto. La caja de censura sigue siendo la del campo visual.
                mrz_val = mrz_identity.get(fid) if side == "front" and fid in {"apellido_paterno", "prenombres"} else None
                if mrz_val:
                    local_compact = re.sub(r"[^A-Z]", "", norm_text(val))
                    mrz_compact = re.sub(r"[^A-Z]", "", norm_text(mrz_val))
                    if local_compact == mrz_compact:
                        conf = 0.58 * ln.confidence + 0.42 * registration_confidence
                        add_field(fd, ln.box, mrz_val, conf, "OCR ROI + MRZ + plantilla")
                    elif local_compact and local_compact in mrz_compact and len(local_compact) < max(3, int(len(mrz_compact) * 0.78)):
                        # El OCR visual encontró solo el inicio de un nombre que la MRZ contiene completo.
                        add_field(fd, value_roi, mrz_val, 0.91 * registration_confidence, "MRZ + plantilla")
                    elif len(local_compact) < 3:
                        add_field(fd, value_roi, mrz_val, 0.88 * registration_confidence, "MRZ + plantilla")
                    else:
                        # Si ambas lecturas discrepan pero el recorte visual produjo un nombre completo,
                        # no sustituirlo por una MRZ posiblemente degradada por reflejos/compresión.
                        conf = 0.58 * ln.confidence + 0.42 * registration_confidence
                        add_field(fd, ln.box, val, conf, "OCR ROI + plantilla")
                    continue

                conf = 0.55 * ln.confidence + 0.45 * registration_confidence
                output_box = value_roi if fd.get("fixed_value_box") else ln.box
                add_field(fd, output_box, val, conf, "OCR ROI + plantilla")
                continue
            if side == "front" and fid in {"apellido_paterno", "prenombres"} and mrz_identity.get(fid):
                add_field(fd, value_roi, mrz_identity[fid], 0.88 * registration_confidence, "MRZ + plantilla")
                continue
            if fd.get("geometry_fallback"):
                add_field(fd, value_roi, "Región esperada por plantilla", 0.88 * registration_confidence, "plantilla")
                continue

        # Back side and any legacy fields without a value ROI retain label-aware extraction.
        ln: OCRLine | None = None
        value: str | None = None
        if pattern == "date":
            ln = _best_line_in_roi(lines, roi, _date_like)
        elif pattern == "dni8":
            ln = _best_line_in_roi(lines, roi, lambda text: _dni8(text) is not None)
            if ln:
                value = _dni8(ln.text)
        if ln is None and fd.get("labels"):
            ln = _find_label_value(lines, fd["labels"], roi)
            if ln is not None and _is_static_label(ln.text):
                ln = None
            if ln is None:
                ln = _best_value_in_roi(lines, roi)
        if ln is None and not fd.get("labels") and pattern is None and not fd.get("geometry_fallback"):
            ln = _best_value_in_roi(lines, roi)

        if ln is not None:
            local = _inside_score(ln.box, roi)
            conf = 0.48 * ln.confidence + 0.30 * local + 0.22 * registration_confidence
            add_field(fd, ln.box, value or ln.text, conf, "OCR + plantilla")
            continue

        if fd.get("geometry_fallback"):
            conf = 0.90 * registration_confidence if reference_used else max(0.30, 0.58 * registration_confidence)
            add_field(fd, roi, "Región esperada por plantilla", conf, "plantilla")

    fields.sort(key=lambda f: (f.box[1], f.box[0]))
    return fields

def process_blue_dni(page: np.ndarray, page_index: int, quad: np.ndarray, detection_confidence: float) -> tuple[DocumentResult, str]:
    H_page_to_card = perspective_matrix(quad, CANON_W, CANON_H)
    card = warp_with_matrix(page, H_page_to_card, CANON_W, CANON_H)

    oriented, angle, side, class_conf = choose_orientation_and_side(card)
    H_rot, rotated_size = rotation_homography(CANON_W, CANON_H, angle)
    rw, rh = rotated_size
    if (rw, rh) != (CANON_W, CANON_H):
        sx = CANON_W / rw; sy = CANON_H / rh
        H_scale = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        oriented = cv2.resize(oriented, (CANON_W, CANON_H), interpolation=cv2.INTER_CUBIC)
    else:
        H_scale = np.eye(3, dtype=np.float64)

    aligned, H_reg, reg_conf, ref_used = _register_reference(oriented, side)

    warnings: list[str] = []
    try:
        lines, engine = run_ocr_resilient(aligned)
        structured_defs = SPEC["front_fields"] if side == "front" else SPEC["back_fields"]
        lines = refine_ocr_by_layout(aligned, lines, structured_defs, max_regions=10)
    except Exception:
        lines, engine = [], "Sin OCR local disponible (solo plantilla geométrica)"
        warnings.append("No se pudo ejecutar OCR local; la interfaz mostrará solo regiones basadas en plantilla.")

    fields = extract_fields(aligned, lines, side, reg_conf, ref_used)

    H_page_to_canon = H_reg @ H_scale @ H_rot @ H_page_to_card
    try:
        H_canon_to_page = np.linalg.inv(H_page_to_canon)
    except np.linalg.LinAlgError:
        H_canon_to_page = np.linalg.pinv(H_page_to_canon)

    if detection_confidence < 0.55:
        warnings.append("El contorno del DNI no se detectó con alta confianza; revisa los recuadros.")
    if class_conf < 0.48:
        warnings.append("No se pudo confirmar con alta confianza si es anverso o reverso.")
    if not ref_used:
        warnings.append("No se pudo aplicar la plantilla calibrada; se usa geometría normalizada + OCR.")
    elif reg_conf < 0.60:
        warnings.append("El registro geométrico tiene confianza media; usa zonas manuales si algún campo queda desplazado.")

    side_label = "Anverso" if side == "front" else "Reverso"
    result = DocumentResult(
        id=str(uuid.uuid4()), page_index=page_index, type="dni_azul", side=side,
        side_label=side_label, orientation=angle,
        detection_confidence=round(float(detection_confidence), 3),
        classification_confidence=round(float(class_conf), 3),
        registration_confidence=round(float(reg_conf), 3), reference_used=ref_used,
        width=CANON_W, height=CANON_H, preview=image_to_data_url(aligned), fields=fields, warnings=warnings,
        _page_to_canonical=H_page_to_canon, _canonical_to_page=H_canon_to_page, _image=aligned,
    )
    return result, engine
