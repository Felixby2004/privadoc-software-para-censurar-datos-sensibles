from __future__ import annotations

import re
import uuid

import cv2
import numpy as np

from app.core.imaging import image_to_data_url
from app.core.models import DocumentResult, OCRLine, SensitiveField
from app.core.ocr import norm_text, run_ocr, run_ocr_resilient


LABEL_KEYWORDS = [
    "ENVIADO A",
    "DESTINATARIO",
    "DESDE",
    "REMITENTE",
    "ORDENANTE",
    "TITULAR",
    "COMISION",
    "NUMERO DE OPERACION",
    "CODIGO DE OPERACION",
    "NUMERO DE TRANSACCION",
    "CODIGO DE TRANSACCION",
    "ID DE TRANSACCION",
    "FECHA Y HORA",
]


def _clamp_box(box: list[int], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return [x1, y1, x2, y2]


def _expand(box: list[int], px: int = 8, py: int = 6, w: int | None = None, h: int | None = None) -> list[int]:
    x1, y1, x2, y2 = box
    out = [x1 - px, y1 - py, x2 + px, y2 + py]
    if w is not None and h is not None:
        return _clamp_box(out, w, h)
    return out


def _cy(line: OCRLine) -> float:
    return (line.box[1] + line.box[3]) / 2.0


def _cx(line: OCRLine) -> float:
    return (line.box[0] + line.box[2]) / 2.0


def _text_has_alpha(text: str) -> bool:
    return bool(re.search(r"[A-ZÁÉÍÓÚÜÑ]", text or "", re.I))


def _digits_count(text: str) -> int:
    return sum(ch.isdigit() for ch in text or "")


def _is_label(text: str) -> bool:
    nt = norm_text(text)
    return any(k in nt for k in LABEL_KEYWORDS)


def _contains_any(nt: str, items: list[str]) -> bool:
    return any(x in nt for x in items)


def _find_label(lines: list[OCRLine], keywords: list[str]) -> OCRLine | None:
    best: tuple[float, OCRLine] | None = None
    for line in lines:
        nt = norm_text(line.text)
        score = 0.0
        for kw in keywords:
            nkw = norm_text(kw)
            if nkw == nt:
                score = max(score, 3.0)
            elif nkw in nt:
                score = max(score, 2.0)
            else:
                tokens = [t for t in nkw.split() if len(t) >= 3]
                hit = sum(1 for t in tokens if t in nt)
                if tokens and hit / len(tokens) >= 0.66:
                    score = max(score, 1.0 + hit / len(tokens))
        if score > 0:
            score += max(0.0, min(1.0, line.confidence)) * 0.4
            if best is None or score > best[0]:
                best = (score, line)
    return best[1] if best else None


def _next_section_y(lines: list[OCRLine], current: OCRLine, others: list[OCRLine | None], default_h: int) -> int:
    y = default_h
    for other in others:
        if other is None:
            continue
        if other.box[1] > current.box[1]:
            y = min(y, other.box[1] - 8)
    return y


def _find_name_line(lines: list[OCRLine], label: OCRLine, section_bottom: int, page_w: int) -> OCRLine | None:
    best: tuple[float, OCRLine] | None = None
    ly = _cy(label)
    for line in lines:
        if line is label:
            continue
        nt = norm_text(line.text)
        if not _text_has_alpha(line.text):
            continue
        if _contains_any(nt, ["MONEDA", "SOL", "YAPE", "PLIN", "GRATIS", "CUENTAS", "AHORRO", "OPERACION", "FECHA", "COMISION", "CONSTANCIA"]):
            continue
        cy = _cy(line)
        if cy < label.box[1] - 25 or cy > section_bottom:
            continue
        letters = len(re.findall(r"[A-Z]", nt))
        digits = _digits_count(nt)
        score = letters * 1.4 - digits * 3.0
        if line.box[0] > label.box[2] + 60:
            score += 18
        if abs(cy - ly) <= 45:
            score += 12
        if line.box[1] >= label.box[3] - 10:
            score += 8
        if line.box[0] >= page_w * 0.25:
            score += 4
        if best is None or score > best[0]:
            best = (score, line)
    return best[1] if best else None


def _find_product_line(lines: list[OCRLine], label: OCRLine, section_bottom: int) -> OCRLine | None:
    best: tuple[float, OCRLine] | None = None
    for line in lines:
        if line is label:
            continue
        nt = norm_text(line.text)
        cy = _cy(line)
        if cy < label.box[1] - 20 or cy > section_bottom:
            continue
        if not _contains_any(nt, ["CUENTA", "CUENTAS", "AHORRO", "CORRIENTE", "TARJETA"]):
            continue
        score = len(nt)
        if line.box[0] > label.box[2] + 80:
            score += 8
        if best is None or score > best[0]:
            best = (score, line)
    return best[1] if best else None


def _has_nearby_wallet_hint(lines: list[OCRLine], line: OCRLine, section_bottom: int) -> bool:
    """Detecta Yape/Plin cerca de una línea numérica, incluso si OCR los separó en otra fila."""
    for other in lines:
        if other is line:
            continue
        nt = norm_text(other.text)
        if not _contains_any(nt, ["YAPE", "PLIN"]):
            continue
        cy = _cy(other)
        if cy < line.box[1] - 12 or cy > min(section_bottom, line.box[3] + 95):
            continue
        # Debe estar razonablemente alineado con el valor detectado.
        if other.box[2] < line.box[0] - 80 or other.box[0] > line.box[2] + 130:
            continue
        return True
    return False


def _find_numeric_line(lines: list[OCRLine], y1: int, y2: int, x_min: int = 0) -> OCRLine | None:
    best: tuple[float, OCRLine] | None = None
    for line in lines:
        nt = norm_text(line.text)
        if _is_label(line.text):
            continue
        cy = _cy(line)
        if cy < y1 or cy > y2:
            continue
        if line.box[0] < x_min:
            continue
        digits = _digits_count(line.text)
        if digits < 3 and "*" not in line.text:
            continue
        score = digits * 2.0 + line.confidence * 2.0
        if _contains_any(nt, ["YAPE", "PLIN"]):
            score += 3
        if best is None or score > best[0]:
            best = (score, line)
    return best[1] if best else None


def _map_crop_line(line: OCRLine, roi: list[int], scale: float) -> OCRLine:
    x1, y1, _, _ = roi
    bx1, by1, bx2, by2 = line.box
    return OCRLine(
        line.text,
        line.confidence,
        [
            int(round(x1 + bx1 / scale)),
            int(round(y1 + by1 / scale)),
            int(round(x1 + bx2 / scale)),
            int(round(y1 + by2 / scale)),
        ],
    )


def _targeted_ocr(page: np.ndarray, roi: list[int]) -> list[OCRLine]:
    """OCR de rescate para valores pequeños que el OCR general puede omitir.

    Sólo se ejecuta cuando falta un dato esperado, por lo que no penaliza el flujo normal.
    Prueba el recorte ampliado y una variante de alto contraste.
    """
    h, w = page.shape[:2]
    x1, y1, x2, y2 = _clamp_box(roi, w, h)
    crop = page[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    scale = 2.4
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    variants = [enlarged, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)]
    found: list[OCRLine] = []
    for variant in variants:
        try:
            local, _ = run_ocr(variant)
        except Exception:
            local = []
        for line in local:
            mapped = _map_crop_line(line, [x1, y1, x2, y2], scale)
            if not mapped.text.strip():
                continue
            duplicate = False
            for old in found:
                if norm_text(old.text) == norm_text(mapped.text) and abs(_cy(old) - _cy(mapped)) < 12:
                    duplicate = True
                    if mapped.confidence > old.confidence:
                        old.text = mapped.text
                        old.confidence = mapped.confidence
                        old.box = mapped.box
                    break
            if not duplicate:
                found.append(mapped)
    return found


def _find_phone_near_name(page: np.ndarray, lines: list[OCRLine], label: OCRLine, name_line: OCRLine, section_bottom: int) -> OCRLine | None:
    """Busca el teléfono bajo/alrededor del nombre del destinatario.

    Está orientado a vouchers como Plin/Yape donde el número suele ir en la fila
    inmediatamente inferior y RapidOCR puede omitirlo en el análisis de página completa.
    """
    h, w = page.shape[:2]
    y1 = max(label.box[1], name_line.box[3] - 6)
    y2 = min(section_bottom, name_line.box[3] + max(90, int(h * 0.075)))
    x1 = max(0, min(label.box[0], name_line.box[0]) - 12)
    x2 = min(w, max(name_line.box[2] + int(w * 0.18), int(w * 0.62)))
    candidates = [ln for ln in lines if y1 <= _cy(ln) <= y2 and ln.box[2] >= x1 and ln.box[0] <= x2]
    # Si el OCR global no trajo una línea numérica suficiente, reintentar sólo el bloque esperado.
    if not any(_digits_count(ln.text) >= 7 or "*" in (ln.text or "") for ln in candidates):
        candidates.extend(_targeted_ocr(page, [x1, y1, x2, y2]))

    best: tuple[float, OCRLine] | None = None
    for line in candidates:
        text = line.text or ""
        nt = norm_text(text)
        digits = re.sub(r"\D", "", text)
        stars = text.count("*")
        wallet_hint = _contains_any(nt, ["YAPE", "PLIN"]) or _has_nearby_wallet_hint(lines + candidates, line, section_bottom)
        looks_peruvian_mobile = len(digits) == 9 and digits.startswith("9")
        looks_masked_mobile = stars >= 2 and len(digits) >= 3 and wallet_hint
        if not (looks_peruvian_mobile or looks_masked_mobile):
            continue
        # Evitar fechas y códigos de operación: el teléfono debe estar pegado al nombre.
        if re.search(r"\b(?:19|20)\d{2}\b", text):
            continue
        cy = _cy(line)
        score = min(12, len(digits)) * 2.2 + float(line.confidence) * 4.0
        if name_line.box[3] - 8 <= line.box[1] <= name_line.box[3] + 75:
            score += 14.0
        if abs(line.box[0] - name_line.box[0]) <= max(80, int(w * 0.12)):
            score += 7.0
        if _contains_any(nt, ["YAPE", "PLIN"]):
            score += 9.0
        if _has_nearby_wallet_hint(lines + candidates, line, section_bottom):
            score += 7.0
        if best is None or score > best[0]:
            best = (score, line)
    return best[1] if best else None


def _char_cut_ratio(text: str, keep_count: int, *, digits_only: bool = False) -> float:
    """Aproxima el punto horizontal después de N letras/dígitos dentro de una línea OCR."""
    if not text:
        return 0.20
    seen = 0
    for i, ch in enumerate(text):
        valid = ch.isdigit() if digits_only else ch.isalpha()
        if valid:
            seen += 1
            if seen >= keep_count:
                # +1 para colocar el inicio de la máscara después del carácter conservado.
                return max(0.06, min(0.88, (i + 1) / max(1, len(text))))
    return 0.18


def _partial_mask_box(box: list[int], text: str, w: int, h: int) -> list[int]:
    """Nombre: conserva exactamente el inicio equivalente a las primeras 3 letras."""
    bw = max(10, box[2] - box[0])
    ratio = _char_cut_ratio(text, 3, digits_only=False)
    keep = int(round(bw * ratio))
    keep = min(bw - 5, max(18, keep))
    return _clamp_box([box[0] + keep, box[1], box[2], box[3]], w, h)


def _partial_phone_mask_box(box: list[int], text: str, w: int, h: int) -> list[int]:
    """Teléfono: conserva sólo los primeros 3 dígitos cuando son visibles en el comprobante."""
    # Si el comprobante ya oculta el inicio con asteriscos, no exponemos los dígitos finales.
    stripped = (text or "").strip()
    first_digit = next((i for i, ch in enumerate(stripped) if ch.isdigit()), None)
    first_star = stripped.find("*")
    if first_star >= 0 and (first_digit is None or first_star < first_digit):
        return _clamp_box([box[0], box[1], box[2], box[3]], w, h)
    bw = max(10, box[2] - box[0])
    ratio = _char_cut_ratio(stripped, 3, digits_only=True)
    keep = int(round(bw * ratio))
    keep = min(bw - 4, max(20, keep))
    return _clamp_box([box[0] + keep, box[1], box[2], box[3]], w, h)


def _sensitive_segment_box(line: OCRLine, w: int, h: int) -> tuple[list[int], str]:
    text = line.text.strip()
    nt = norm_text(text)
    x1, y1, x2, y2 = line.box
    full_w = max(10, x2 - x1)

    if "-" in text and _contains_any(nt, ["YAPE", "PLIN"]):
        left = text.split("-", 1)[0].rstrip()
        ratio = max(0.32, min(0.82, len(left) / max(1, len(text))))
        seg_x2 = x1 + int(full_w * ratio) + 10
        return _clamp_box([x1 - 4, y1 - 4, seg_x2, y2 + 4], w, h), left.strip()

    digits = re.findall(r"\d+", text)
    if len("".join(digits)) >= 8:
        return _clamp_box([x1 - 4, y1 - 4, x2 + 6, y2 + 4], w, h), text

    # Cuando el OCR sólo ve los últimos 4 dígitos, ampliar hacia la izquierda para cubrir los asteriscos omitidos.
    expand_left = max(90, int(full_w * 1.9))
    label_value = f"**** {digits[-1]}" if digits else text
    return _clamp_box([x1 - expand_left, y1 - 6, x2 + 8, y2 + 6], w, h), label_value.strip()


def _add_field(fields: list[SensitiveField], name: str, value: str, box: list[int], confidence: float, *, risk: str = "Alto", category: str = "financial", source: str = "OCR + estructura", recommendation: str = "total", mask_box: list[int] | None = None, default_redact: bool = True):
    fields.append(SensitiveField(
        id=str(uuid.uuid4()),
        name=name,
        value=value,
        risk=risk,
        recommendation=recommendation,
        box=box,
        confidence=float(max(0.0, min(0.99, confidence))),
        category=category,
        default_redact=default_redact,
        source=source,
        mask_box=mask_box,
    ))


def _extract_inline_digits_box(line: OCRLine, min_digits: int = 5) -> tuple[str, list[int]] | None:
    text = line.text or ""
    matches = list(re.finditer(r"(?<!\d)([A-Z0-9-]*\d[A-Z0-9-]{%d,})(?!\d)" % max(1, min_digits - 1), norm_text(text)))
    if not matches:
        return None
    m = max(matches, key=lambda x: sum(ch.isdigit() for ch in x.group(1)))
    value = m.group(1).strip(" -")
    if sum(ch.isdigit() for ch in value) < min_digits:
        return None
    x1, y1, x2, y2 = line.box
    bw = max(1, x2 - x1)
    # norm_text conserva razonablemente la longitud visual para este tipo de línea.
    nt = norm_text(text)
    idx = nt.find(value)
    if idx < 0:
        return value, list(line.box)
    sx = x1 + int(bw * (idx / max(1, len(nt))))
    ex = x1 + int(bw * ((idx + len(value)) / max(1, len(nt))))
    return value, [sx, y1, max(sx + 8, ex), y2]


def _find_operation_value(lines: list[OCRLine], label: OCRLine | None, page_w: int, page_h: int) -> tuple[str, list[int], float] | None:
    if label is None:
        return None
    inline = _extract_inline_digits_box(label, 5)
    if inline is not None:
        value, box = inline
        return value, _expand(box, 5, 3, page_w, page_h), float(label.confidence)

    ly = _cy(label)
    best: tuple[float, OCRLine, str] | None = None
    for line in lines:
        if line is label:
            continue
        cy = _cy(line)
        # En BCP suele estar en la misma fila a la derecha; en Plin, en la fila siguiente.
        if cy < label.box[1] - 18 or cy > label.box[3] + 95:
            continue
        nt = norm_text(line.text)
        if _contains_any(nt, ["FECHA", "HORA", "COMISION", "GRATIS"]):
            continue
        digits = re.sub(r"\D", "", line.text or "")
        if len(digits) < 5:
            continue
        # Evitar montos/fechas: buscamos valores compactos de operación.
        if re.search(r"[/:]", line.text or "") and len(digits) <= 8:
            continue
        score = len(digits) * 2.0 + float(line.confidence) * 4.0
        if line.box[0] > label.box[2] + 20:
            score += 10.0
        if abs(cy - ly) <= 35:
            score += 8.0
        if line.box[1] >= label.box[3] - 4:
            score += 4.0
        if best is None or score > best[0]:
            best = (score, line, digits)
    if best is None:
        return None
    _, line, digits = best
    return digits, _expand(line.box, 5, 3, page_w, page_h), float(line.confidence)


def _looks_like_person_name(text: str) -> bool:
    nt = norm_text(text)
    if not _text_has_alpha(text):
        return False
    if _contains_any(nt, ["CUENTA", "CUENTAS", "AHORRO", "CORRIENTE", "TARJETA", "YAPE", "PLIN", "BANCO", "MONEDA", "SOL", "GRATIS", "OPERACION", "COMISION"]):
        return False
    letters = len(re.findall(r"[A-Z]", nt))
    digits = _digits_count(nt)
    words = [x for x in nt.split() if x]
    return letters >= 4 and digits == 0 and 1 <= len(words) <= 8


def process_receipt(page: np.ndarray, page_index: int) -> tuple[DocumentResult, str]:
    h, w = page.shape[:2]
    lines, engine = run_ocr_resilient(page)
    lines = sorted(lines, key=lambda ln: (ln.box[1], ln.box[0]))

    label_to = _find_label(lines, ["Enviado a", "Destinatario", "Beneficiario", "Recibe"])
    label_from = _find_label(lines, ["Desde"])
    label_origin_name = _find_label(lines, ["Remitente", "Ordenante", "Titular", "Enviado por", "Nombre del remitente", "Nombre de origen"])
    label_fee = _find_label(lines, ["Comisión"])
    label_op = _find_label(lines, ["Número de operación", "Código de operación", "Número de transacción", "Código de transacción", "ID de transacción", "Nro de operación", "Nro de transacción"])
    label_date = _find_label(lines, ["Fecha y hora"])

    fields: list[SensitiveField] = []
    warnings: list[str] = []

    # Nombre del destinatario/titular (censura parcial).
    if label_to is not None:
        section_bottom = _next_section_y(lines, label_to, [label_fee, label_from, label_op, label_date], h)
        name_line = _find_name_line(lines, label_to, section_bottom, w)
        if name_line is not None:
            name_box = _expand(name_line.box, 6, 4, w, h)
            _add_field(
                fields,
                "Nombre del destinatario",
                name_line.text,
                name_box,
                min(0.97, 0.55 + name_line.confidence * 0.45),
                risk="Medio",
                category="personal",
                source="OCR estructural",
                recommendation="partial",
                mask_box=_partial_mask_box(name_box, name_line.text, w, h),
            )

            contact_line = _find_numeric_line(lines, name_line.box[1] - 8, section_bottom, x_min=max(label_to.box[0], name_line.box[0] - 25))
            rescue_phone = _find_phone_near_name(page, lines, label_to, name_line, section_bottom)
            # Un teléfono cercano al nombre tiene prioridad sobre una línea numérica genérica.
            if rescue_phone is not None:
                contact_line = rescue_phone
            if contact_line is not None:
                sensitive_box, sensitive_value = _sensitive_segment_box(contact_line, w, h)
                contact_digits = re.sub(r"\D", "", sensitive_value)
                wallet_hint = _contains_any(norm_text(contact_line.text), ["YAPE", "PLIN"]) or _has_nearby_wallet_hint(lines, contact_line, section_bottom)
                is_phone = (
                    rescue_phone is not None
                    or (len(contact_digits) == 9 and contact_digits.startswith("9"))
                    or ("*" in (contact_line.text or "") and wallet_hint)
                )
                _add_field(
                    fields,
                    "Celular destino" if is_phone else "Cuenta / tarjeta destino",
                    sensitive_value,
                    sensitive_box,
                    min(0.97, 0.54 + contact_line.confidence * 0.44),
                    risk="Medio" if is_phone else "Alto",
                    category="contact" if is_phone else "financial",
                    source="OCR focal + estructura" if rescue_phone is not None else "OCR estructural",
                    recommendation="partial_phone" if is_phone else "total",
                    mask_box=_partial_phone_mask_box(sensitive_box, sensitive_value, w, h) if is_phone else None,
                )
        else:
            warnings.append("No se pudo aislar con suficiente confianza el nombre del destinatario. Puedes añadir una zona manual si lo necesitas.")

    # Cuenta/tarjeta de origen (censura total).
    if label_from is not None:
        section_bottom = _next_section_y(lines, label_from, [label_op], h)
        product_line = _find_product_line(lines, label_from, section_bottom)
        search_x = product_line.box[0] if product_line is not None else int(w * 0.45)
        account_line = _find_numeric_line(lines, label_from.box[1] - 5, section_bottom + 15, x_min=search_x)
        if account_line is not None:
            account_box, account_value = _sensitive_segment_box(account_line, w, h)
            _add_field(
                fields,
                "Cuenta / tarjeta origen",
                account_value,
                account_box,
                min(0.94, 0.50 + account_line.confidence * 0.45),
                risk="Alto",
                category="financial",
                source="OCR estructural",
                recommendation="total",
            )
        elif product_line is not None:
            px1 = max(int(w * 0.73), int(product_line.box[0] + (product_line.box[2] - product_line.box[0]) * 0.60))
            py1 = min(h - 8, product_line.box[3] + 8)
            py2 = min(h, py1 + max(34, int((product_line.box[3] - product_line.box[1]) * 1.15)))
            fallback_box = _clamp_box([px1, py1, min(w - 8, product_line.box[2] + 12), py2], w, h)
            _add_field(
                fields,
                "Cuenta / tarjeta origen",
                "****",
                fallback_box,
                0.66,
                risk="Alto",
                category="financial",
                source="estructura",
                recommendation="total",
            )

    # Ancla extra para comprobantes que sólo muestran cuenta destino debajo del nombre.
    if label_to is not None and not any(f.name == "Cuenta / tarjeta destino" for f in fields) and not any(f.name == "Celular destino" for f in fields):
        section_bottom = _next_section_y(lines, label_to, [label_fee, label_from, label_op, label_date], h)
        x_min = int(w * 0.60)
        digits_line = _find_numeric_line(lines, label_to.box[1], section_bottom, x_min=x_min)
        if digits_line is not None:
            b, v = _sensitive_segment_box(digits_line, w, h)
            _add_field(fields, "Cuenta / tarjeta destino", v, b, min(0.92, 0.50 + digits_line.confidence * 0.42), risk="Alto", category="financial", source="OCR estructural", recommendation="total")

    # Nombre del remitente/origen, si el comprobante lo muestra como persona.
    origin_anchor = label_origin_name or label_from
    if origin_anchor is not None:
        origin_bottom = _next_section_y(lines, origin_anchor, [label_fee, label_op, label_date], h)
        origin_line = _find_name_line(lines, origin_anchor, origin_bottom, w)
        if origin_line is not None and _looks_like_person_name(origin_line.text):
            origin_box = _expand(origin_line.box, 6, 4, w, h)
            _add_field(
                fields,
                "Nombre del remitente / origen",
                origin_line.text,
                origin_box,
                min(0.96, 0.54 + origin_line.confidence * 0.44),
                risk="Medio",
                category="personal",
                source="OCR estructural",
                recommendation="partial",
                mask_box=_partial_mask_box(origin_box, origin_line.text, w, h),
            )

    # Número/código de operación o transacción: se detecta para revisión, pero se mantiene visible.
    op_value = _find_operation_value(lines, label_op, w, h)
    if op_value is not None:
        value, op_box, op_conf = op_value
        _add_field(
            fields,
            "Número / código de operación",
            value,
            op_box,
            min(0.98, 0.58 + op_conf * 0.40),
            risk="Contexto",
            category="context",
            source="OCR estructural",
            recommendation="visible",
            default_redact=False,
        )

    if not fields:
        warnings.append("No se identificaron regiones sensibles con suficiente confianza. Usa la selección manual para complementar el comprobante.")

    anchor_hits = sum(1 for x in [label_to, label_from, label_origin_name, label_fee, label_op, label_date] if x is not None)
    cls_conf = min(0.99, 0.56 + anchor_hits * 0.08)
    doc = DocumentResult(
        id=str(uuid.uuid4()),
        page_index=page_index,
        type="comprobante",
        side="single",
        side_label="Comprobante digital",
        orientation=0,
        detection_confidence=0.98,
        classification_confidence=cls_conf,
        registration_confidence=0.96,
        reference_used=False,
        width=w,
        height=h,
        preview=image_to_data_url(page),
        fields=fields,
        warnings=warnings,
        _page_to_canonical=np.eye(3, dtype=np.float64),
        _canonical_to_page=np.eye(3, dtype=np.float64),
        _image=page,
    )
    return doc, engine
