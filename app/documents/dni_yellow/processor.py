from __future__ import annotations

import json
import re
import uuid
from collections import Counter, defaultdict
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
    """Parsea la MRZ del DNI amarillo usando voto entre múltiples variantes OCR.

    En tarjetas antiguas/amarillas una sola pasada OCR puede confundir 3/8, 1/I, etc.
    En vez de quedarnos con el primer resultado, reunimos todas las lecturas y elegimos
    la alternativa que más se repite. El dígito inmediatamente posterior al DNI dentro
    de la MRZ se conserva como `mrz_check_digit`: NO es el dígito impreso junto al DNI.
    """
    result: dict[str, str] = {}
    lines: list[str] = []
    for text in texts:
        for raw in re.split(r"[\r\n|]+", (text or "").upper()):
            compact = re.sub(r"[^A-Z0-9<]", "", raw)
            if compact:
                lines.append(compact)

    scan_lines = lines + (["".join(lines)] if lines else [])

    number_votes: list[tuple[str, str]] = []
    for line in scan_lines:
        patterns = [
            r"PER([0-9OQDISBL]{8})<+([0-9OQDISBL])",
            r"([0-9OQDISBL]{8})<+([0-9OQDISBL])",
        ]
        for pat in patterns:
            for m in re.finditer(pat, line):
                num = _digits_from_ocr(m.group(1))
                chk = _digits_from_ocr(m.group(2))
                if len(num) == 8 and len(chk) == 1:
                    number_votes.append((num, chk))
    if number_votes:
        num_counts = Counter(n for n, _ in number_votes)
        best_num, _ = num_counts.most_common(1)[0]
        chk_counts = Counter(c for n, c in number_votes if n == best_num)
        result["dni_number"] = best_num
        if chk_counts:
            result["mrz_check_digit"] = chk_counts.most_common(1)[0][0]

    birth_votes: list[str] = []
    sex_votes: list[str] = []
    for line in scan_lines:
        for m in re.finditer(r"([0-9OQDISBL]{6})[0-9OQDISBL]([MF])([0-9OQDISBL]{6})", line):
            birth = _digits_from_ocr(m.group(1))
            if len(birth) == 6:
                try:
                    yy, mm, dd = int(birth[:2]), int(birth[2:4]), int(birth[4:6])
                    if 1 <= mm <= 12 and 1 <= dd <= 31:
                        year = 2000 + yy if yy <= 30 else 1900 + yy
                        birth_votes.append(f"{dd:02d} {mm:02d} {year:04d}")
                except Exception:
                    pass
            sex_votes.append(m.group(2))
    if birth_votes:
        result["birth_date"] = Counter(birth_votes).most_common(1)[0][0]
    if sex_votes:
        result["sexo"] = Counter(sex_votes).most_common(1)[0][0]

    # Tercera fila: PRIMER_APELLIDO<<PRENOMBRES. Exigimos prenombres no vacíos;
    # así descartamos cadenas corruptas que terminan solo en <<<<<<.
    identity_votes: list[tuple[str, str]] = []
    for line in lines:
        if any(ch.isdigit() for ch in line) or "<<" not in line:
            continue
        for m in re.finditer(r"([A-Z]{3,15})<<([A-Z<]{3,40})", line):
            surname = m.group(1).strip("<")
            given_tokens = [x for x in m.group(2).split("<") if 2 <= len(x) <= 20]
            given = " ".join(given_tokens[:4]).strip()
            if 3 <= len(surname) <= 15 and len(given.replace(" ", "")) >= 3:
                identity_votes.append((surname, given))
    if identity_votes:
        # Votar apellido y prenombres por separado permite conservar una lectura correcta
        # incluso si otra variante OCR corta uno de los nombres.
        sur = Counter(x[0] for x in identity_votes).most_common(1)[0][0]
        giv = Counter(x[1] for x in identity_votes).most_common(1)[0][0]
        result["apellido_paterno"] = sur
        result["prenombres"] = giv
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



def _compact_with_index_map(text: str) -> tuple[str, list[int]]:
    """Compacta una línea OCR conservando el índice de cada carácter útil en el texto original."""
    compact: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate((text or "").upper()):
        if ch.isspace():
            continue
        compact.append(ch)
        index_map.append(i)
    return "".join(compact), index_map


def _subbox_from_text_span(line: OCRLine, raw_text: str, raw_start: int, raw_end: int, pad_x: int = 3, pad_y: int = 3) -> list[int]:
    """Aproxima el bbox de una subcadena dentro del bbox OCR de la línea completa.

    RapidOCR suele entregar toda la cabecera como una sola línea. La altura de la línea es fiable;
    la posición horizontal de una subcadena se aproxima por su rango de caracteres.
    """
    x1, y1, x2, y2 = [int(v) for v in line.box]
    n = max(1, len(raw_text))
    left_ratio = max(0.0, min(1.0, raw_start / n))
    right_ratio = max(left_ratio, min(1.0, raw_end / n))
    w = max(1, x2 - x1)
    sx1 = int(round(x1 + w * left_ratio)) - pad_x
    sx2 = int(round(x1 + w * right_ratio)) + pad_x
    return [max(0, sx1), max(0, y1 - pad_y), min(CANON_W, sx2), min(CANON_H, y2 + pad_y)]


def _parse_header_line_with_boxes(line: OCRLine) -> dict[str, object]:
    """Extrae número/dígito y sus cajas desde una línea OCR del encabezado."""
    raw = (line.text or "").upper()
    if not raw.strip():
        return {}

    # Mapa de cadena compacta -> posiciones del texto OCR original.
    compact, idx_map = _compact_with_index_map(raw)
    if not compact:
        return {}

    patterns = [
        r"DNI[^0-9OQDISBL]*([0-9OQDISBL]{8})[-<]?([0-9OQDISBL])?",
        r"([0-9OQDISBL]{8})[-<]([0-9OQDISBL])",
    ]
    for pat in patterns:
        m = re.search(pat, compact)
        if not m:
            continue
        number = _digits_from_ocr(m.group(1))
        if len(number) != 8:
            continue
        out: dict[str, object] = {"dni_number": number}

        # Convertir el span compacto a un span aproximado dentro del texto original.
        ns, ne = m.span(1)
        if ns < len(idx_map) and ne - 1 < len(idx_map):
            raw_start = idx_map[ns]
            raw_end = idx_map[ne - 1] + 1
            out["dni_number_box"] = _subbox_from_text_span(line, raw, raw_start, raw_end, 4, 3)

        if m.lastindex and m.lastindex >= 2 and m.group(2):
            dv = _digits_from_ocr(m.group(2))
            if len(dv) == 1:
                out["verification_digit"] = dv
                ds, de = m.span(2)
                if ds < len(idx_map) and de - 1 < len(idx_map):
                    out["verification_digit_box"] = _subbox_from_text_span(line, raw, idx_map[ds], idx_map[de - 1] + 1, 5, 3)
        return out
    return {}


def _extract_header_identity_located(img: np.ndarray, full_lines: list[OCRLine] | None = None) -> dict[str, object]:
    """Lee el encabezado amarillo y, cuando es posible, devuelve bbox reales del número y DV.

    La banda se coloca sobre la segunda línea de cabecera (DOCUMENTO ... DNI XXXXXXXX-X), no
    sobre la franja superior de microtexto. Esto evita el desplazamiento vertical observado.
    """
    # Banda deliberadamente más baja que la usada originalmente.
    band = _roi_to_box([0.56, 0.065, 0.998, 0.175])
    candidates: list[OCRLine] = []
    try:
        candidates.extend(_ocr_crop_lines(img, band, scale=3.6))
    except Exception:
        pass

    if full_lines:
        for ln in full_lines:
            cy = _center(ln.box)[1]
            if band[1] - 8 <= cy <= band[3] + 8 and ln.box[2] >= int(CANON_W * 0.58):
                candidates.append(ln)

    # Primero priorizar líneas que contengan DNI; luego tokens numéricos de 8 dígitos.
    ranked = sorted(
        candidates,
        key=lambda ln: (
            1 if "DNI" in norm_text(ln.text) else 0,
            1 if len(_digits_from_ocr(ln.text)) >= 8 else 0,
            float(ln.confidence),
            ln.box[0],
        ),
        reverse=True,
    )

    best: dict[str, object] = {}
    for ln in ranked:
        parsed = _parse_header_line_with_boxes(ln)
        if parsed.get("dni_number"):
            best = parsed
            break

    # Caso común de OCR tokenizado: número y DV vienen en tokens separados.
    if not best.get("dni_number"):
        num_candidates: list[tuple[float, OCRLine, str]] = []
        for ln in candidates:
            digits = _digits_from_ocr(ln.text)
            if len(digits) == 8:
                # El número impreso está en la mitad derecha y en la banda de cabecera.
                cx, cy = _center(ln.box)
                if cx >= CANON_W * 0.68 and band[1] - 8 <= cy <= band[3] + 8:
                    num_candidates.append((float(ln.confidence), ln, digits))
        if num_candidates:
            _, nline, number = max(num_candidates, key=lambda x: x[0])
            best["dni_number"] = number
            best["dni_number_box"] = [max(0, nline.box[0]-4), max(0,nline.box[1]-3), min(CANON_W,nline.box[2]+4), min(CANON_H,nline.box[3]+3)]

            # Buscar un único dígito a la derecha, en la misma fila.
            ncy = _center(nline.box)[1]
            nh = max(8, nline.box[3]-nline.box[1])
            dv_cands: list[tuple[float, OCRLine, str]] = []
            for ln in candidates:
                if ln.box[0] < nline.box[2] - 3:
                    continue
                if abs(_center(ln.box)[1] - ncy) > nh * 1.2:
                    continue
                digits = re.sub(r"\D", "", ln.text or "")
                if len(digits) == 1:
                    dv_cands.append((float(ln.confidence), ln, digits))
            if dv_cands:
                _, dline, dv = max(dv_cands, key=lambda x: x[0])
                best["verification_digit"] = dv
                best["verification_digit_box"] = [max(0,dline.box[0]-5), max(0,dline.box[1]-3), min(CANON_W,dline.box[2]+5), min(CANON_H,dline.box[3]+3)]

    # Valores OCR/Tesseract como respaldo aunque no tengan bbox.
    fallback = _extract_header_identity(img, full_lines)
    for k, v in fallback.items():
        best.setdefault(k, v)
    return best


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


def _label_present(lines: list[OCRLine], labels: list[str]) -> bool:
    joined = " | ".join(norm_text(x.text) for x in lines)
    for label in labels:
        a = norm_text(label)
        if a in joined:
            return True
        tokens = [t for t in a.split() if len(t) >= 3]
        if tokens and sum(1 for t in tokens if t in joined) >= max(1, int(np.ceil(len(tokens) * 0.75))):
            return True
        # En algunas impresiones amarillas OCR degrada "Cuarto" pero conserva "Nivel".
        if a == "CUARTO NIVEL" and "NIVEL" in joined:
            return True
    return False



def _parse_all_dates(text: str) -> list[str]:
    out: list[str] = []
    src = (text or "").upper()
    for m in re.finditer(r"(?<!\d)([0-3]?\d)[\s./-]+([01]?\d)[\s./-]+((?:19|20)\d{2})(?!\d)", src):
        try:
            dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= dd <= 31 and 1 <= mm <= 12:
                out.append(f"{dd:02d} {mm:02d} {yy:04d}")
        except Exception:
            pass
    # Fechas compactadas DDMMYYYY.
    for m in re.finditer(r"(?<!\d)([0-3]\d)([01]\d)((?:19|20)\d{2})(?!\d)", src):
        val = f"{int(m.group(1)):02d} {int(m.group(2)):02d} {int(m.group(3)):04d}"
        if val not in out:
            out.append(val)
    return out


def _find_anchor_line(lines: list[OCRLine], aliases: list[str], y_hint: float | None = None) -> OCRLine | None:
    best: tuple[float, OCRLine] | None = None
    for ln in lines:
        nt = norm_text(ln.text)
        if not nt:
            continue
        score = 0.0
        for alias in aliases:
            a = norm_text(alias)
            tokens = [t for t in a.split() if len(t) >= 3]
            if a and a in nt:
                score = max(score, 4.0)
            elif tokens:
                hits = sum(1 for t in tokens if t in nt)
                score = max(score, 2.2 * hits / len(tokens))
                # Etiquetas degradadas como "... Apellido" siguen siendo útiles si Y concuerda.
                if "APELLIDO" in a and "APELLIDO" in nt:
                    score = max(score, 1.65)
                if "NOMBRES" in a and "NOMB" in nt:
                    score = max(score, 1.65)
                if "NACIMIENTO" in a and "NACIM" in nt:
                    score = max(score, 1.75)
        if score <= 0:
            continue
        if y_hint is not None:
            cy = _center(ln.box)[1] / max(1.0, CANON_H)
            score -= min(1.2, abs(cy - y_hint) * 4.0)
        score += 0.20 * float(ln.confidence)
        if best is None or score > best[0]:
            best = (score, ln)
    return best[1] if best and best[0] >= 1.15 else None


def _box_below_label(label: OCRLine, width: int, height: int, x_offset: int = 0, y_gap: int = -1) -> list[int]:
    x1 = max(0, int(label.box[0] + x_offset))
    y1 = max(0, int(label.box[3] + y_gap))
    return [x1, y1, min(CANON_W, x1 + width), min(CANON_H, y1 + height)]


def _date_triplet_ocr(img: np.ndarray) -> list[tuple[str | None, list[int], float]]:
    """Lee las tres fechas del bloque derecho independientemente de la plantilla."""
    block = _roi_to_box([0.75, 0.08, 0.995, 0.55])
    x1, y1, x2, y2 = block
    crop = img[y1:y2, x1:x2]
    local = _ocr_crop_lines(img, block, scale=3.2)

    # Agrupar detecciones de fecha por fila visual.
    clusters: list[dict] = []
    for ln in local:
        val = _parse_date_value(ln.text)
        if not val:
            continue
        cy = _center(ln.box)[1]
        target = None
        for c in clusters:
            if abs(c["cy"] - cy) <= 14:
                target = c; break
        if target is None:
            clusters.append({"cy": cy, "values": [val], "boxes": [ln.box], "confs": [float(ln.confidence)]})
        else:
            target["values"].append(val); target["boxes"].append(ln.box); target["confs"].append(float(ln.confidence))
            target["cy"] = sum(_center(b)[1] for b in target["boxes"]) / len(target["boxes"])
    clusters.sort(key=lambda c: c["cy"])

    ordinal_votes: list[Counter] = [Counter(), Counter(), Counter()]
    # Cada variante Tesseract aporta una secuencia completa; votar por posición evita
    # que una lectura aislada (p. ej. 16 vs 15) domine el resultado.
    text_variants: list[str] = []
    for psm in (6, 12):
        text_variants.extend(_tesseract_text_variants(crop, whitelist=None, psm=psm))
    for text in text_variants:
        vals = _parse_all_dates(text)
        if len(vals) >= 3:
            for i in range(3): ordinal_votes[i][vals[i]] += 1

    results: list[tuple[str | None, list[int], float]] = []
    fallback_boxes = [
        _roi_to_box([0.80, 0.17, 0.97, 0.235]),
        _roi_to_box([0.80, 0.30, 0.97, 0.365]),
        _roi_to_box([0.80, 0.42, 0.97, 0.49]),
    ]
    for i in range(3):
        val = ordinal_votes[i].most_common(1)[0][0] if ordinal_votes[i] else None
        box = fallback_boxes[i]
        conf = 0.72 if val else 0.45
        if i < len(clusters):
            c = clusters[i]
            cluster_val = Counter(c["values"]).most_common(1)[0][0]
            if val is None:
                val = cluster_val
            # BBox representativa de la fila, no el bounding de la etiqueta.
            best_idx = max(range(len(c["boxes"])), key=lambda j: c["confs"][j])
            box = list(c["boxes"][best_idx])
            conf = max(conf, min(0.97, 0.72 + 0.25 * c["confs"][best_idx]))
        results.append((val, box, conf))
    return results


def _read_single_header_digit(img: np.ndarray) -> tuple[str | None, float]:
    # El dígito verificador impreso en el DNI amarillo NO es el check digit de la MRZ.
    box = _roi_to_box([0.915, 0.025, 0.992, 0.125])
    candidates: list[tuple[float, str]] = []
    for ln in _ocr_crop_lines(img, box, scale=4.2):
        digits = re.sub(r"\D", "", ln.text or "")
        if len(digits) == 1:
            candidates.append((float(ln.confidence), digits))
        elif len(digits) <= 3 and digits:
            candidates.append((float(ln.confidence) * 0.72, digits[-1]))
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    for psm in (10, 7, 13):
        for txt in _tesseract_text_variants(crop, whitelist="0123456789-", psm=psm):
            digits = re.sub(r"\D", "", txt)
            if len(digits) == 1:
                candidates.append((0.80, digits))
            elif 1 < len(digits) <= 3:
                candidates.append((0.62, digits[-1]))
    if not candidates:
        return None, 0.0
    # Voto ponderado por confianza.
    scores: dict[str, float] = defaultdict(float)
    for score, digit in candidates:
        scores[digit] += score
    digit = max(scores, key=scores.get)
    return digit, min(0.95, scores[digit] / max(1.0, len([1 for _, d in candidates if d == digit])))



def _locate_vertical_dni_box(img: np.ndarray, expected_number: str | None = None) -> tuple[list[int] | None, str | None, float]:
    """Localiza el DNI impreso verticalmente en el borde izquierdo del anverso amarillo.

    La tarjeta ya está normalizada. Se recorta solo la banda lateral, se rota para convertir
    los dígitos verticales en texto horizontal y se busca un token de 8 dígitos. Si conocemos
    el DNI por cabecera/MRZ, ese número se usa como validación fuerte.
    """
    h, w = img.shape[:2]
    sx1, sy1 = int(round(0.012 * w)), int(round(0.105 * h))
    sx2, sy2 = int(round(0.135 * w)), int(round(0.685 * h))
    strip = img[sy1:sy2, sx1:sx2]
    if strip.size == 0:
        return None, None, 0.0

    expected = re.sub(r"\D", "", expected_number or "")
    best: tuple[float, list[int], str] | None = None

    for direction in ("cw", "ccw"):
        rotated = cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE if direction == "cw" else cv2.ROTATE_90_COUNTERCLOCKWISE)
        scale = 2.8
        enlarged = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants = [enlarged]
        try:
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            variants.append(cv2.cvtColor(cv2.createCLAHE(2.4, (8, 8)).apply(gray), cv2.COLOR_GRAY2BGR))
        except Exception:
            pass

        for variant in variants:
            try:
                ocr_lines, _ = run_ocr(variant)
            except Exception:
                continue
            for ln in ocr_lines:
                digits = _digits_from_ocr(ln.text)
                # Aceptamos solo secuencias plausibles; con número esperado exigimos coincidencia.
                candidate = None
                if expected and len(expected) == 8 and expected in digits:
                    candidate = expected
                elif not expected:
                    m = re.search(r"(?<!\d)(\d{8})(?!\d)", digits)
                    if m:
                        candidate = m.group(1)
                if not candidate:
                    continue

                rx1, ry1, rx2, ry2 = [v / scale for v in ln.box]
                sh, sw = strip.shape[:2]
                if direction == "cw":
                    # CW: x' = h-1-y, y' = x. Inversa: x=y', y=h-1-x'.
                    ox1, ox2 = ry1, ry2
                    oy1, oy2 = sh - rx2, sh - rx1
                else:
                    # CCW: x'=y, y'=w-1-x. Inversa: x=w-1-y', y=x'.
                    ox1, ox2 = sw - ry2, sw - ry1
                    oy1, oy2 = rx1, rx2

                box = [
                    max(0, int(round(sx1 + ox1 - 4))),
                    max(0, int(round(sy1 + oy1 - 4))),
                    min(w, int(round(sx1 + ox2 + 4))),
                    min(h, int(round(sy1 + oy2 + 4))),
                ]
                bw, bh = box[2] - box[0], box[3] - box[1]
                if bw <= 2 or bh <= 12:
                    continue
                # El código lateral debe ser una banda estrecha y alta.
                geometry = min(1.0, bh / max(1.0, bw * 3.0))
                score = 0.72 * float(ln.confidence) + 0.28 * geometry
                if expected and candidate == expected:
                    score += 0.18
                if best is None or score > best[0]:
                    best = (score, box, candidate)

    if best is None:
        return None, None, 0.0
    return best[1], best[2], min(0.99, best[0])

def _extract_yellow_front_fields(img: np.ndarray, lines: list[OCRLine], registration_confidence: float, reference_used: bool) -> list[SensitiveField]:
    defs = {fd["id"]: fd for fd in SPEC["front_fields"]}
    fields: list[SensitiveField] = []
    mrz = _extract_mrz_identity(img, lines)
    header = _extract_header_identity_located(img, lines)

    # OCR adicional solamente en el bloque textual. Se usa para localizar etiquetas y no
    # para deformar la imagen según una única plantilla.
    rich_lines = list(lines)
    for region in ([0.24, 0.08, 0.73, 0.69], [0.75, 0.08, 0.995, 0.55]):
        try:
            rich_lines.extend(_ocr_crop_lines(img, _roi_to_box(region), scale=2.8))
        except Exception:
            pass

    def add(fid: str, box: list[int], value: str, confidence: float, source: str) -> None:
        fd = defs[fid]
        fields.append(SensitiveField(
            id=fid, name=fd["name"], value=value, risk=fd["risk"],
            recommendation="Censurar antes de compartir si este dato no es necesario.",
            box=[int(v) for v in box], confidence=round(max(0.22, min(0.99, confidence)), 3),
            category=fd.get("category", "personal"), default_redact=bool(fd.get("default_redact", True)),
            source=source,
        ))

    # Número DNI: contenido validado con MRZ, pero ubicación tomada del OCR real de cabecera.
    # La plantilla solo entra si no fue posible localizar físicamente la línea "DNI XXXXXXXX-X".
    number = header.get("dni_number") or mrz.get("dni_number")
    number_box = header.get("dni_number_box") or _roi_to_box([0.705, 0.105, 0.94, 0.158])
    number_source = "OCR cabecera localizado" if header.get("dni_number_box") else "MRZ + zona cabecera"
    if header.get("dni_number") and mrz.get("dni_number") == header.get("dni_number"):
        number_source = "OCR cabecera localizado + MRZ" if header.get("dni_number_box") else "OCR cabecera + MRZ"
    add("dni_number", number_box, number or "No leído con confianza", 0.97 if number and header.get("dni_number_box") else (0.94 if number else 0.55), number_source if number else "zona cabecera")

    # El dígito verificador impreso se localiza en la misma línea y a la derecha del número.
    # No se usa el check digit de la MRZ como sustituto.
    verifier = header.get("verification_digit")
    vconf = 0.94 if verifier and header.get("verification_digit_box") else (0.88 if verifier else 0.0)
    if not verifier:
        verifier, vconf = _read_single_header_digit(img)
    verifier_box = header.get("verification_digit_box") or _roi_to_box([0.945, 0.105, 0.992, 0.158])
    add("verification_digit", verifier_box, verifier or "No leído con confianza", vconf if verifier else 0.50,
        "OCR cabecera localizado" if header.get("verification_digit_box") else ("OCR cabecera" if verifier else "zona cabecera"))

    # Código vertical lateral: es el mismo DNI impreso en vertical. Se localiza con OCR
    # sobre una banda estrecha rotada, usando el número de cabecera/MRZ como validación.
    serial_box, serial_value, serial_conf = _locate_vertical_dni_box(img, number)
    if serial_box is not None:
        add("left_serial", serial_box, serial_value or number or "Código lateral identificado", max(0.82, serial_conf), "OCR vertical + DNI")
    else:
        # Fallback más estrecho y ligeramente desplazado a la izquierda que el ROI anterior.
        add("left_serial", _roi_to_box([0.020, 0.15, 0.090, 0.63]), number or "Región sensible identificada", 0.72, "geometría normalizada")

    # Fotografías: regiones geométricas estables una vez normalizada la tarjeta.
    for fid in ("photo", "mini_photo"):
        fd = defs[fid]
        add(fid, _roi_to_box(fd["roi"]), "Región sensible identificada", 0.91, "geometría normalizada")

    # Nombres: etiqueta OCR como ancla. Si la etiqueta no es fiable, usar la plantilla solo como fallback.
    name_cfg = {
        "apellido_paterno": (["PRIMER APELLIDO"], 0.18, 320),
        "apellido_materno": (["SEGUNDO APELLIDO"], 0.31, 320),
        "prenombres": (["PRE NOMBRES", "PRENOMBRES"], 0.43, 420),
    }
    for fid, (aliases, yhint, width) in name_cfg.items():
        fd = defs[fid]
        label = _find_anchor_line(rich_lines, aliases, yhint)
        box = _roi_to_box(fd["value_roi"])
        if label is not None and (label.box[3] - label.box[1]) <= 36:
            box = _box_below_label(label, width, 36, y_gap=-1)
        else:
            # ROI estrecha: evita mezclar la etiqueta con el valor.
            box = [box[0] - 5, box[1] - 5, box[2] + 5, box[3] + 5]
        local = _best_crop_value(img, box, "letters")
        local_val = norm_text(local.text) if local else ""
        mrz_val = mrz.get(fid) if fid in {"apellido_paterno", "prenombres"} else None
        val = local_val
        source = "OCR anclado"
        conf = 0.55
        if mrz_val:
            a = re.sub(r"[^A-Z]", "", local_val)
            b = re.sub(r"[^A-Z]", "", norm_text(mrz_val))
            if not a or a in b or len(a) < max(3, int(len(b) * 0.78)):
                val = mrz_val
                source = "MRZ + OCR anclado"
                conf = 0.91
            elif a == b:
                val = mrz_val
                source = "OCR anclado + MRZ"
                conf = 0.96
            else:
                conf = 0.82 if local else 0.72
        elif local:
            conf = max(0.72, float(local.confidence))
        if local is not None and label is not None:
            out_box = local.box
        else:
            out_box = _snap_box_to_text_band(img, box, 0.035)
        add(fid, out_box, val or "No leído con confianza", conf, source if val else "zona anclada")

    # Nacimiento/ubigeo: una sola etiqueta fija ambos recuadros inferiores.
    birth_label = _find_anchor_line(rich_lines, ["NACIMIENTO FECHA Y UBIGEO", "NACIMIENTO"], 0.52)
    birth_fd, ubi_fd = defs["birth_date"], defs["ubigeo"]
    birth_box = _roi_to_box(birth_fd["value_roi"])
    ubi_box = _roi_to_box(ubi_fd["value_roi"])
    if birth_label is not None and (birth_label.box[3] - birth_label.box[1]) <= 38:
        birth_box = _box_below_label(birth_label, 155, 34, y_gap=-1)
        ubi_box = _box_below_label(birth_label, 135, 34, x_offset=145, y_gap=-1)
    birth_local = _best_crop_value(img, birth_box, "date")
    birth_val = mrz.get("birth_date") or (_parse_date_value(birth_local.text) if birth_local else None)
    add("birth_date", birth_local.box if birth_local else birth_box, birth_val or "No leído con confianza",
        0.95 if mrz.get("birth_date") else (0.80 if birth_local else 0.52), "MRZ + OCR anclado" if mrz.get("birth_date") else "OCR anclado")
    ubi_local = _best_crop_value(img, ubi_box, "ubigeo6")
    ubi_val = _digits_from_ocr(ubi_local.text) if ubi_local else None
    add("ubigeo", ubi_local.box if ubi_local else ubi_box, ubi_val or "No leído con confianza",
        max(0.50, float(ubi_local.confidence) if ubi_local else 0.0), "OCR anclado" if ubi_local else "zona anclada")

    # Cuarto nivel es opcional; solo existe si OCR detecta explícitamente "Nivel".
    level_label = _find_anchor_line(rich_lines, ["CUARTO NIVEL"], 0.45)
    if level_label is not None:
        level_box = _box_below_label(level_label, 145, 34, y_gap=-1)
        level_local = _best_crop_value(img, level_box, "ubigeo6")
        if level_local:
            add("cuarto_nivel", level_local.box, _digits_from_ocr(level_local.text), max(0.70, float(level_local.confidence)), "OCR etiqueta + valor")

    # Sexo: MRZ para el valor, etiqueta visual para ubicar el recuadro justo debajo.
    sex_label = _find_anchor_line(rich_lines, ["SEXO"], 0.60)
    sex_box = _roi_to_box(defs["sexo"]["value_roi"])
    if sex_label is not None and (sex_label.box[3] - sex_label.box[1]) <= 32:
        sex_box = _box_below_label(sex_label, 58, 30, y_gap=-1)
    sex_val = mrz.get("sexo")
    sex_local = _best_crop_value(img, sex_box, "sex_letter")
    if not sex_val and sex_local:
        sex_val = norm_text(sex_local.text)
    add("sexo", sex_box, sex_val or "No leído con confianza", 0.95 if mrz.get("sexo") else (0.72 if sex_local else 0.50), "MRZ + etiqueta OCR" if mrz.get("sexo") else "OCR anclado")

    # Fechas del panel derecho: OCR del bloque completo, no tres ROIs rígidas.
    triplet = _date_triplet_ocr(img)
    for fid, item in zip(("fecha_inscripcion", "fecha_emision", "fecha_caducidad"), triplet):
        value, box, conf = item
        add(fid, box, value or "No leído con confianza", conf, "OCR bloque de fechas")

    # MRZ: la caja de censura SIEMPRE queda limitada al bloque inferior de tres líneas.
    # El OCR se ejecuta únicamente dentro de ese recorte para evitar que una línea grande
    # del OCR global expanda el bbox hasta abarcar todo el DNI.
    mrz_fd = defs["mrz"]
    mrz_box = _roi_to_box(mrz_fd["roi"])
    try:
        mrz_crop_lines = _ocr_crop_lines(img, mrz_box, scale=2.35)
    except Exception:
        mrz_crop_lines = []
    mrz_text_lines = [x.text for x in mrz_crop_lines if _mrz_like(x.text)]
    if not mrz_text_lines:
        # Recompone filas aunque RapidOCR las haya dividido en varios tokens.
        rows = _rows_from_ocr_lines(mrz_crop_lines)
        mrz_text_lines = [x for x in rows if _mrz_like(x)]
    mrz_value = " / ".join(mrz_text_lines[:3]) if mrz_text_lines else "Zona MRZ identificada"
    add("mrz", mrz_box, mrz_value, 0.93 if mrz_text_lines else 0.88,
        "OCR MRZ acotado + geometría" if mrz_text_lines else "geometría normalizada")

    fields.sort(key=lambda f: (f.box[1], f.box[0]))
    return fields



def _tight_digits_box(line: OCRLine, digits: str) -> list[int]:
    """Aproxima dentro del bbox OCR el tramo que corresponde a los 8 dígitos."""
    x1, y1, x2, y2 = [int(v) for v in line.box]
    text = (line.text or "").strip()
    if not text or not digits:
        return [x1, y1, x2, y2]
    compact = re.sub(r"\s+", "", text.upper())
    pos = compact.find(digits)
    if pos < 0:
        # Si OCR leyó DNI/LE junto al número, los dígitos suelen ocupar la mitad derecha.
        if "DNI" in compact or "LE" in compact:
            return [int(round(x1 + (x2-x1)*0.43)), y1, x2, y2]
        return [x1, y1, x2, y2]
    n=max(1,len(compact))
    sx=x1+(x2-x1)*(pos/n)
    ex=x1+(x2-x1)*((pos+len(digits))/n)
    pad=max(2,int((y2-y1)*0.12))
    return [max(0,int(sx)-pad), max(0,y1-2), min(CANON_W,int(ex)+pad), min(CANON_H,y2+2)]


def _family_anchor_boxes(anchor: OCRLine | None, role: str) -> tuple[list[list[int]], list[int]]:
    """Devuelve tres bandas de nombre y la banda DNI/LE ancladas en Madre/Padre."""
    if role == "mother":
        fallback_rows=[
            _roi_to_box([0.15,0.07,0.48,0.13]),
            _roi_to_box([0.15,0.115,0.48,0.175]),
            _roi_to_box([0.15,0.16,0.52,0.22]),
        ]
        fallback_doc=_roi_to_box([0.205,0.205,0.39,0.255])
    else:
        fallback_rows=[
            _roi_to_box([0.15,0.255,0.48,0.315]),
            _roi_to_box([0.15,0.30,0.48,0.36]),
            _roi_to_box([0.15,0.345,0.52,0.405]),
        ]
        fallback_doc=_roi_to_box([0.205,0.39,0.39,0.44])
    if anchor is None:
        return fallback_rows, fallback_doc

    # La primera línea del nombre va inmediatamente a la derecha de la palabra ancla.
    cy=int(round(_center(anchor.box)[1]))
    x1=max(int(anchor.box[2])+8, int(CANON_W*0.14))
    x2=int(CANON_W*0.46)
    centers=[cy, cy+23, cy+46]
    rows=[]
    for c in centers:
        rows.append([x1,max(0,c-10),x2,min(CANON_H,c+11)])
    doc_c=cy+70
    doc=[max(int(anchor.box[2])+4, int(CANON_W*0.17)),max(0,doc_c-12),int(CANON_W*0.42),min(CANON_H,doc_c+14)]
    return rows,doc


def _merge_lines_as_rows(lines: list[OCRLine], y_tol: int = 12) -> list[OCRLine]:
    rows: list[list[OCRLine]] = []
    ordered = sorted(lines, key=lambda x: (_center(x.box)[1], x.box[0]))
    for ln in ordered:
        cy = _center(ln.box)[1]
        h = max(1.0, ln.box[3] - ln.box[1])
        target = None
        for row in rows:
            rcy = sum(_center(x.box)[1] for x in row) / len(row)
            rh = sum(max(1.0, x.box[3] - x.box[1]) for x in row) / len(row)
            if abs(cy - rcy) <= max(float(y_tol), max(h, rh) * 0.8):
                target = row
                break
        if target is None:
            rows.append([ln])
        else:
            target.append(ln)
    merged: list[OCRLine] = []
    for row in rows:
        row.sort(key=lambda x: x.box[0])
        text = " ".join((x.text or "").strip() for x in row if (x.text or "").strip())
        if not text:
            continue
        box = [min(x.box[0] for x in row), min(x.box[1] for x in row), max(x.box[2] for x in row), max(x.box[3] for x in row)]
        conf = sum(float(x.confidence) for x in row) / len(row)
        merged.append(OCRLine(text, conf, box))
    merged.sort(key=lambda x: (_center(x.box)[1], x.box[0]))
    return merged


def _normalize_name_text(text: str) -> str:
    letters = re.sub(r"[^A-Z ]", " ", norm_text(text))
    return re.sub(r"\s+", " ", letters).strip()


def _family_rows_from_anchors(lines: list[OCRLine], anchor: OCRLine | None, role: str) -> tuple[list[OCRLine], OCRLine | None]:
    if anchor is None:
        return [], None
    ax2 = int(anchor.box[2])
    ay = _center(anchor.box)[1]
    x1 = max(ax2 + 4, int(CANON_W * 0.14))
    x2 = int(CANON_W * 0.50)
    y1 = max(0, int(ay - 16))
    y2 = min(CANON_H, int(ay + (88 if role == "mother" else 92)))

    family_lines: list[OCRLine] = []
    for ln in lines:
        lx1, ly1, lx2, ly2 = [int(v) for v in ln.box]
        cx, cy = _center(ln.box)
        if cy < y1 or cy > y2:
            continue
        if lx2 < x1 or lx1 > x2:
            continue
        # Nos quedamos con el bloque a la derecha de Madre/Padre.
        if cx < x1:
            continue
        family_lines.append(ln)

    rows = _merge_lines_as_rows(family_lines, y_tol=11)
    name_rows: list[OCRLine] = []
    doc_row: OCRLine | None = None
    for row in rows:
        nt = norm_text(row.text)
        if not nt:
            continue
        digits = re.sub(r"\D", "", nt)
        if ("DNI" in nt or "LE" in nt or len(digits) >= 8):
            if doc_row is None or abs(_center(row.box)[1] - (ay + 70)) < abs(_center(doc_row.box)[1] - (ay + 70)):
                doc_row = row
            continue
        if _is_static_label(row.text):
            continue
        normalized = _normalize_name_text(row.text)
        compact = normalized.replace(" ", "")
        if len(compact) < 2:
            continue
        if re.search(r"\d", row.text):
            continue
        name_rows.append(OCRLine(normalized, row.confidence, list(row.box)))

    # Conservar sólo las tres primeras filas reales: apellido paterno, materno y prenombres.
    name_rows.sort(key=lambda x: _center(x.box)[1])
    return name_rows[:3], doc_row


def _read_family_document(img: np.ndarray, box: list[int]) -> tuple[str | None, list[int] | None, float]:
    candidates=[]
    try:
        local=_ocr_crop_lines(img,box,scale=3.4)
    except Exception:
        local=[]
    for ln in local:
        compact=re.sub(r"\D","",ln.text or "")
        m=re.search(r"(?<!\d)(\d{8})(?!\d)",compact)
        if m:
            digits=m.group(1)
            candidates.append((float(ln.confidence),digits,_tight_digits_box(ln,digits)))
    # El OCR a veces conserva DNI/LE y los dígitos separados en dos tokens. Reconstruir por fila.
    if not candidates and local:
        rows=[]
        for ln in sorted(local,key=lambda x:(_center(x.box)[1],x.box[0])):
            cy=_center(ln.box)[1]
            row=next((r for r in rows if abs(r['cy']-cy)<13),None)
            if row is None:
                row={'cy':cy,'items':[ln]};rows.append(row)
            else:
                row['items'].append(ln);row['cy']=sum(_center(z.box)[1] for z in row['items'])/len(row['items'])
        for row in rows:
            items=sorted(row['items'],key=lambda z:z.box[0])
            txt=' '.join(z.text for z in items)
            digits=re.sub(r"\D","",txt)
            m=re.search(r"(\d{8})",digits)
            if m:
                value=m.group(1)
                # caja solo de tokens numéricos cuando sea posible
                nums=[z for z in items if re.search(r"\d",z.text or '')]
                if nums:
                    bb=[min(z.box[0] for z in nums),min(z.box[1] for z in nums),max(z.box[2] for z in nums),max(z.box[3] for z in nums)]
                else: bb=box
                candidates.append((sum(float(z.confidence) for z in items)/len(items),value,bb))
    if not candidates:
        return None,None,0.0
    conf,val,bb=max(candidates,key=lambda x:x[0])
    return val,bb,float(conf)


def _extract_yellow_back_fields(img: np.ndarray, lines: list[OCRLine], registration_confidence: float, reference_used: bool) -> list[SensitiveField]:
    defs={fd['id']:fd for fd in SPEC['back_fields']}
    fields=[]

    def add(fid, box, value, confidence, source):
        fd=defs[fid]
        fields.append(SensitiveField(
            id=fid,name=fd['name'],value=value,risk=fd['risk'],
            recommendation='Censurar antes de compartir si este dato no es necesario.',
            box=[int(v) for v in box],confidence=round(max(0.22,min(0.99,float(confidence))),3),
            category=fd.get('category','personal'),default_redact=bool(fd.get('default_redact',True)),source=source
        ))

    mother_anchor=_find_anchor_line(lines,['MADRE'],y_hint=0.09)
    father_anchor=_find_anchor_line(lines,['PADRE'],y_hint=0.28)

    for role,anchor,prefix in [('mother',mother_anchor,'mother'),('father',father_anchor,'father')]:
        row_boxes,doc_box=_family_anchor_boxes(anchor,role)
        ids=[f'{prefix}_surname1',f'{prefix}_surname2',f'{prefix}_given_names']
        anchored_rows, anchored_doc = _family_rows_from_anchors(lines, anchor, role)
        for idx,(fid,box) in enumerate(zip(ids,row_boxes)):
            row = anchored_rows[idx] if idx < len(anchored_rows) else None
            if row is not None:
                val = _normalize_name_text(row.text)
                out_box = list(row.box)
                conf = 0.66*float(row.confidence)+0.28*max(registration_confidence,0.55)+0.06
                add(fid,out_box,val,conf,'OCR anclado a '+('Madre' if role=='mother' else 'Padre'))
                continue
            local=_best_crop_value(img,box,'letters')
            if local is not None:
                val=_normalize_name_text(local.text)
                out_box=list(local.box)
                conf=0.62*float(local.confidence)+0.30*max(registration_confidence,0.55)+0.08
                add(fid,out_box,val,conf,'OCR fila + ancla '+('Madre' if role=='mother' else 'Padre'))
            else:
                add(fid,box,'Región esperada por estructura',max(0.52,0.72*registration_confidence),'ancla + geometría')

        doc_fid=f'{prefix}_document'
        if anchored_doc is not None:
            digits = re.search(r'(?<!\d)(\d{8})(?!\d)', re.sub(r'\D','', anchored_doc.text or ''))
            if digits:
                value = digits.group(1)
                bb = _tight_digits_box(anchored_doc, value)
                conf = 0.68*float(anchored_doc.confidence)+0.28*max(registration_confidence,0.55)+0.04
                add(doc_fid,bb,value,conf,'OCR DNI/LE anclado a '+('Madre' if role=='mother' else 'Padre'))
                continue
        value,bb,ocr_conf=_read_family_document(img,doc_box)
        if value and bb:
            add(doc_fid,bb,value,0.68*ocr_conf+0.28*max(registration_confidence,0.55)+0.04,'OCR DNI/LE + ancla')
        else:
            add(doc_fid,doc_box,'Región esperada por estructura',max(0.50,0.70*registration_confidence),'ancla + geometría')

    handled={
        'mother_surname1','mother_surname2','mother_given_names','mother_document',
        'father_surname1','father_surname2','father_given_names','father_document'
    }
    # El resto del reverso mantiene extracción por etiqueta/geometría, pero sin volver a unir los nombres.
    for fd in SPEC['back_fields']:
        if fd['id'] in handled:
            continue
        fid=fd['id']; roi=_roi_to_box(fd['roi'])
        ln=None
        if fd.get('labels'):
            ln=_find_label_value(lines,fd['labels'],roi)
            if ln is not None and _is_static_label(ln.text): ln=None
            if ln is None: ln=_best_value_in_roi(lines,roi)
        elif not fd.get('geometry_fallback'):
            ln=_best_value_in_roi(lines,roi)
        if ln is not None:
            loc=_inside_score(ln.box,roi)
            conf=0.50*float(ln.confidence)+0.28*loc+0.22*max(registration_confidence,0.45)
            add(fid,ln.box,ln.text,conf,'OCR + estructura')
        elif fd.get('geometry_fallback'):
            conf=0.88*registration_confidence if reference_used else max(0.42,0.60*registration_confidence)
            add(fid,roi,'Región esperada por plantilla',conf,'plantilla')

    fields.sort(key=lambda f:(f.box[1],f.box[0]))
    return fields


def extract_fields(img: np.ndarray, lines: list[OCRLine], side: str, registration_confidence: float, reference_used: bool) -> list[SensitiveField]:
    if side == "front":
        return _extract_yellow_front_fields(img, lines, registration_confidence, reference_used)
    if side == "back":
        return _extract_yellow_back_fields(img, lines, registration_confidence, reference_used)
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

        if fd.get("optional_label_required") and not _label_present(lines, fd.get("labels") or []):
            continue
        if fd.get("optional_label_required"):
            kind = "ubigeo6" if pattern == "ubigeo6" else "text"
            # El campo opcional existe solo si su etiqueta fue observada; por eso sí podemos
            # intentar OCR local aunque el matching de referencia sea débil.
            local = _best_crop_value(img, ocr_roi, kind)
            if local is not None:
                val = _digits_from_ocr(local.text) if kind == "ubigeo6" else local.text.strip()
                add_field(fd, value_roi, val, max(0.58, 0.55 * local.confidence + 0.35 * registration_confidence), "OCR ROI + etiqueta")
            else:
                add_field(fd, value_roi, "Región opcional detectada por etiqueta", max(0.48, registration_confidence), "etiqueta + plantilla")
            continue

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

def process_yellow_dni(page: np.ndarray, page_index: int, quad: np.ndarray, detection_confidence: float) -> tuple[DocumentResult, str]:
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

    registered, H_reg_candidate, reg_conf, ref_used = _register_reference(oriented, side)
    # DNI amarillo anverso: la plantilla sirve como señal de clasificación/confianza, pero
    # NO deforma el documento. Las variantes reales tienen diferencias de interlineado y
    # campos opcionales (Cuarto Nivel); para ellas OCR + etiquetas es más preciso.
    if side == "front":
        aligned = oriented
        H_reg = np.eye(3, dtype=np.float64)
    elif ref_used and reg_conf >= 0.88:
        aligned = registered
        H_reg = H_reg_candidate
    else:
        aligned = oriented
        H_reg = np.eye(3, dtype=np.float64)
        ref_used = False
        reg_conf = max(float(reg_conf), min(0.86, 0.84 * float(detection_confidence)))

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
        warnings.append("La plantilla solo se usa como referencia; la ubicación fina se resolvió con geometría normalizada + OCR.")
    elif reg_conf < 0.60:
        warnings.append("El registro geométrico tiene confianza media; usa zonas manuales si algún campo queda desplazado.")

    side_label = "Anverso" if side == "front" else "Reverso"
    result = DocumentResult(
        id=str(uuid.uuid4()), page_index=page_index, type="dni_amarillo", side=side,
        side_label=side_label, orientation=angle,
        detection_confidence=round(float(detection_confidence), 3),
        classification_confidence=round(float(class_conf), 3),
        registration_confidence=round(float(reg_conf), 3), reference_used=ref_used,
        width=CANON_W, height=CANON_H, preview=image_to_data_url(aligned), fields=fields, warnings=warnings,
        _page_to_canonical=H_page_to_canon, _canonical_to_page=H_canon_to_page, _image=aligned,
    )
    return result, engine
