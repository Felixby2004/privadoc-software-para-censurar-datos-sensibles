from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.geometry import order_points

CARD_RATIO = 85.60 / 53.98


@dataclass
class CardCandidate:
    quad: np.ndarray
    confidence: float
    area_ratio: float


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1 = a.min(axis=0); ax2, ay2 = a.max(axis=0)
    bx1, by1 = b.min(axis=0); bx2, by2 = b.max(axis=0)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(1.0, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter / ua)


def _candidate_from_rect(rect, area: float, image_area: float, scale: float, *, min_fill: float = 0.28) -> CardCandidate | None:
    (_, _), (rw, rh), _ = rect
    if min(rw, rh) < 55:
        return None
    ratio = max(rw, rh) / max(1.0, min(rw, rh))
    ratio_error = abs(ratio - CARD_RATIO) / CARD_RATIO
    if ratio_error > 0.34:
        return None
    area_ratio = area / max(1.0, image_area)
    rect_area = max(1.0, float(rw * rh))
    fill = max(0.0, min(1.0, area / rect_area))
    if fill < min_fill:
        return None
    ratio_score = max(0.0, 1.0 - ratio_error * 2.0)
    size_score = min(1.0, area_ratio / 0.22)
    score = 0.56 * ratio_score + 0.28 * fill + 0.16 * size_score
    quad = order_points(cv2.boxPoints(rect).astype(np.float32)) / scale
    return CardCandidate(quad, float(max(0.30, min(0.98, score))), float(area_ratio))


def _dedupe_candidates(items: list[CardCandidate], max_cards: int) -> list[CardCandidate]:
    kept: list[CardCandidate] = []
    for cand in sorted(items, key=lambda x: (x.confidence, x.area_ratio), reverse=True):
        cx1, cy1 = cand.quad.min(axis=0); cx2, cy2 = cand.quad.max(axis=0)
        c_area = max(1.0, float((cx2 - cx1) * (cy2 - cy1)))
        duplicate = False
        for k in kept:
            if _bbox_iou(cand.quad, k.quad) > 0.60:
                duplicate = True
                break
            kx1, ky1 = k.quad.min(axis=0); kx2, ky2 = k.quad.max(axis=0)
            ix1, iy1 = max(cx1, kx1), max(cy1, ky1)
            ix2, iy2 = min(cx2, kx2), min(cy2, ky2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            smaller = min(c_area, max(1.0, float((kx2 - kx1) * (ky2 - ky1))))
            if inter / smaller > 0.82:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(cand)
        if len(kept) >= max_cards:
            break
    kept.sort(key=lambda c: (float(c.quad[:, 1].min()), float(c.quad[:, 0].min())))
    return kept


def _canny_candidates(img: np.ndarray, *, max_side_target: int = 2000) -> list[CardCandidate]:
    h0, w0 = img.shape[:2]
    max_side = max(h0, w0)
    scale = min(1.0, float(max_side_target) / max(1.0, max_side))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(small.shape[0] * small.shape[1])
    out: list[CardCandidate] = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:140]:
        area = float(cv2.contourArea(c))
        aratio = area / max(1.0, image_area)
        if aratio < 0.012 or aratio > 0.97:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            p = order_points(approx.reshape(4, 2).astype(np.float32))
            ew = max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3]))
            eh = max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1]))
            if min(ew, eh) < 40:
                continue
            ratio = max(ew, eh) / max(1.0, min(ew, eh))
            ratio_error = abs(ratio - CARD_RATIO) / CARD_RATIO
            if ratio_error > 0.34:
                continue
            rectangularity = min(1.0, area / max(1.0, ew * eh))
            confidence = max(0.0, min(1.0, 1.0 - ratio_error * 1.6)) * (0.52 + 0.48 * rectangularity)
            out.append(CardCandidate(p / scale, float(confidence), float(aratio)))
        else:
            rect = cv2.minAreaRect(c)
            cand = _candidate_from_rect(rect, area, image_area, scale)
            if cand is not None:
                out.append(cand)
    return out


def _mask_candidates(mask: np.ndarray, scale: float, image_area: float, *, min_area_ratio: float, max_area_ratio: float, min_fill: float = 0.35) -> list[CardCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[CardCandidate] = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        area = float(cv2.contourArea(c))
        area_ratio = area / max(1.0, image_area)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        rect = cv2.minAreaRect(c)
        cand = _candidate_from_rect(rect, area, image_area, scale, min_fill=min_fill)
        if cand is not None:
            out.append(cand)
    return out


def _color_fallback_candidates(img: np.ndarray, color_hint: str | None) -> list[CardCandidate]:
    h0, w0 = img.shape[:2]
    max_side = max(h0, w0)
    scale = min(1.0, 2200.0 / max_side)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    image_area = float(small.shape[0] * small.shape[1])
    out: list[CardCandidate] = []
    if color_hint == "yellow":
        mask = cv2.inRange(hsv, np.array([7, 22, 40], dtype=np.uint8), np.array([42, 255, 255], dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        out.extend(_mask_candidates(mask, scale, image_area, min_area_ratio=0.025, max_area_ratio=0.995, min_fill=0.34))
    elif color_hint == "electronic":
        sat_mask = ((hsv[:, :, 1] >= 28) & (hsv[:, :, 2] >= 45)).astype(np.uint8) * 255
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        out.extend(_mask_candidates(sat_mask, scale, image_area, min_area_ratio=0.02, max_area_ratio=0.97, min_fill=0.32))
        hue_mask = cv2.inRange(hsv, np.array([5, 12, 45], dtype=np.uint8), np.array([78, 230, 255], dtype=np.uint8))
        hue_mask = cv2.morphologyEx(hue_mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=1)
        hue_mask = cv2.morphologyEx(hue_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        out.extend(_mask_candidates(hue_mask, scale, image_area, min_area_ratio=0.02, max_area_ratio=0.97, min_fill=0.30))
    else:
        mask = cv2.inRange(hsv, np.array([78, 42, 28], dtype=np.uint8), np.array([118, 255, 255], dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        out.extend(_mask_candidates(mask, scale, image_area, min_area_ratio=0.02, max_area_ratio=0.92, min_fill=0.34))
    return out


def detect_cards(img: np.ndarray, max_cards: int = 6, color_hint: str | None = None) -> list[CardCandidate]:
    h0, w0 = img.shape[:2]
    all_candidates: list[CardCandidate] = []

    # Doble pasada para páginas PDF/escaneos donde puede haber anverso y reverso en una sola hoja.
    for target in (1800, 2600):
        all_candidates.extend(_canny_candidates(img, max_side_target=target))

    kept = _dedupe_candidates(all_candidates, max_cards=max_cards)

    # Si sólo apareció una tarjeta pequeña o ninguna, complementar con segmentación por color.
    if len(kept) < 2 or (kept and max(c.area_ratio for c in kept) < 0.18):
        all_candidates.extend(_color_fallback_candidates(img, color_hint))
        kept = _dedupe_candidates(all_candidates, max_cards=max_cards)

    if kept:
        return kept

    # Fallback: si la imagen completa ya se parece a una tarjeta, usarla como una sola.
    page_ratio = max(w0, h0) / max(1, min(w0, h0))
    if abs(page_ratio - CARD_RATIO) / CARD_RATIO < 0.22:
        quad = np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]], dtype=np.float32)
        return [CardCandidate(quad, 0.58, 1.0)]

    quad = np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]], dtype=np.float32)
    return [CardCandidate(quad, 0.20, 1.0)]
