from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None


def image_to_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("No se pudo codificar la imagen.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _resize_for_memory(img: np.ndarray) -> np.ndarray:
    max_side = int(
        os.getenv(
            "PRIVADOC_MAX_IMAGE_SIDE",
            "2600" if os.getenv("PRIVADOC_LOW_MEMORY") == "1" else "0",
        )
    )
    if max_side <= 0 or max(img.shape[:2]) <= max_side:
        return img
    height, width = img.shape[:2]
    scale = max_side / max(height, width)
    return cv2.resize(
        img,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def decode_upload(data: bytes, filename: str, content_type: str | None, max_pdf_pages: int = 10) -> tuple[list[np.ndarray], bool]:
    is_pdf = (content_type or "").lower() == "application/pdf" or filename.lower().endswith(".pdf")
    if is_pdf:
        if pdfium is None:
            raise ValueError("Falta pypdfium2 para procesar archivos PDF.")
        pdf = pdfium.PdfDocument(data)
        if len(pdf) < 1:
            raise ValueError("El PDF no contiene páginas.")
        pages: list[np.ndarray] = []
        configured_pages = int(
            os.getenv(
                "PRIVADOC_MAX_PDF_PAGES",
                "6" if os.getenv("PRIVADOC_LOW_MEMORY") == "1" else str(max_pdf_pages),
            )
        )
        page_limit = max(1, min(max_pdf_pages, configured_pages))
        render_scale = float(
            os.getenv(
                "PRIVADOC_PDF_RENDER_SCALE",
                "2.0" if os.getenv("PRIVADOC_LOW_MEMORY") == "1" else "3.0",
            )
        )
        for i in range(min(len(pdf), page_limit)):
            page = pdf[i]
            bitmap = page.render(scale=render_scale)
            pil = bitmap.to_pil().convert("RGB")
            arr = np.array(pil)
            pages.append(_resize_for_memory(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)))
        return pages, True

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("El archivo no es una imagen compatible.")
    return [_resize_for_memory(img)], False


def image_bytes(img: np.ndarray, fmt: str = "png") -> bytes:
    ext = ".jpg" if fmt.lower() in {"jpg", "jpeg"} else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("No se pudo generar la imagen censurada.")
    return buf.tobytes()


def images_to_pdf(images: Iterable[np.ndarray]) -> bytes:
    pil_images: list[Image.Image] = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))
    if not pil_images:
        raise ValueError("No hay páginas para generar el PDF.")
    bio = io.BytesIO()
    first, rest = pil_images[0], pil_images[1:]
    first.save(bio, format="PDF", save_all=True, append_images=rest, resolution=150.0)
    return bio.getvalue()
